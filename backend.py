import os
import sys
import shutil
import cv2
import json
import numpy as np
import time
import traceback
from typing import List, Dict, Tuple, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager
try:
    if not hasattr(np, 'float_'): np.float_ = np.float64
    if not hasattr(np, 'float'): np.float = float
    if not hasattr(np, 'int_'): np.int_ = np.int64
    if not hasattr(np, 'int'): np.int = int
    if not hasattr(np, 'bool'): np.bool = bool
    if not hasattr(np, 'bool8'): np.bool8 = np.bool
except Exception as e:
    print(f"Patch warning: {e}")
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
from ultralytics import YOLO
import google.generativeai as genai

sys.path.append(os.getcwd())

try:
    from utils.inference import model2annotations
except ImportError:
    model2annotations = None
    print("CRITICAL WARNING: inference.py не найден.")

try:
    from manga_ocr import MangaOcr
except ImportError:
    MangaOcr = None
    print("CRITICAL WARNING: manga_ocr библиотека не установлена.")

try:
    if not hasattr(np, 'float_'): np.float_ = np.float64
    if not hasattr(np, 'float'): np.float = float
    if not hasattr(np, 'int_'): np.int_ = np.int64
    if not hasattr(np, 'int'): np.int = int
    if not hasattr(np, 'bool'): np.bool = bool
except Exception:
    pass


@dataclass
class Config:
    bubble_detector_path: str = r'models/best.pt'
    text_detector_path: str = r'models/comictextdetector.pt'
    ocr_model_path: str = r'models/my_manga_ocr_model'
    poppler_path: str = r'poppler/Library/bin'
    font_path: str = 'arial.ttf'
    temp_dir: str = r'data/temp_gui'
    output_dir: str = r'data/results'
    debug_dir: str = r'data/debug'
    gemini_model: str = "gemini-2.5-flash"
    config_file: str = "config.txt"
    grouping_distance: int = 15
    render_padding: int = 15
    line_spacing: float = 1.2

@dataclass
class ProgressInfo:
    progress: int
    message: str
    current_step: int
    total_steps: int

class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    page: int

@dataclass
class DetectionResult:
    pages: List[Image.Image]
    boxes: List[BoundingBox]
    total_blocks: int


def get_optimal_font_size(box, text, font_path, line_spacing_multiplier):
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    if box_width <= 0 or box_height <= 0: return 8
    
    words = text.split()
    if not words: return 8
    
    font_size = 100
    min_font_size = 8
    
    while font_size > min_font_size:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
            return 12
            
        max_word_width = max(font.getbbox(word)[2] for word in words)
        if max_word_width > box_width:
            font_size -= 2
            continue
            
        lines = []
        current_line = ""
        
        for word in words:
            if not current_line:
                current_line = word
            else:
                test_line = current_line + " " + word
                if font.getbbox(test_line)[2] <= box_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
        lines.append(current_line)
        
        ascent, descent = font.getmetrics()
        line_height = (ascent + descent) * line_spacing_multiplier
        total_height = len(lines) * line_height
        
        if total_height < box_height:
            return font_size
            
        font_size -= 2
        
    return min_font_size

def draw_text_in_box(draw, box, text, font, line_spacing_multiplier):
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    if box_width <= 0 or box_height <= 0: return
    
    words = text.split()
    if not words: return
    
    lines = []
    current_line = ""
    for word in words:
        if not current_line:
            current_line = word
        else:
            test_line = current_line + " " + word
            if font.getbbox(test_line)[2] <= box_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
    lines.append(current_line)
    
    ascent, descent = font.getmetrics()
    line_height = (ascent + descent) * line_spacing_multiplier
    total_height = len(lines) * line_height
    y_text = box[1] + (box_height - total_height) / 2
    
    for line in lines:
        line_bbox = font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        x_text = box[0] + (box_width - line_width) / 2
        draw.text((x_text, y_text), line, font=font, fill="black")
        y_text += line_height

def get_bounding_rect(poly):
    return cv2.boundingRect(np.array(poly, dtype=np.int32))

def are_boxes_close(box1, box2, threshold):
    x1, y1, w1, h1 = box1; x2, y2, w2, h2 = box2
    center_dist_x = abs((x1 + w1/2) - (x2 + w2/2))
    center_dist_y = abs((y1 + h1/2) - (y2 + h2/2))
    return (center_dist_x < (w1/2 + w2/2 + threshold)) and \
           (center_dist_y < (h1/2 + h2/2 + threshold))

def group_text_lines(lines, threshold):
    if not lines: return []
    rects = [get_bounding_rect(line) for line in lines]
    adj = {i: [] for i in range(len(rects))}
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if are_boxes_close(rects[i], rects[j], threshold):
                adj[i].append(j); adj[j].append(i)
    groups = []; visited = set()
    for i in range(len(rects)):
        if i not in visited:
            component = []; q = [i]; visited.add(i)
            while q:
                u = q.pop(0); component.append(lines[u])
                for v in adj[u]:
                    if v not in visited: visited.add(v); q.append(v)
            groups.append(component)
    return groups

def split_merged_bubbles(polygon, page_shape):
    mask = np.zeros(page_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    threshold_value = 0.8 
    _, sure_fg = cv2.threshold(dist_transform, threshold_value * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    num_labels, markers = cv2.connectedComponents(sure_fg)
    if num_labels <= 2: return [polygon]
    markers = markers + 1
    unknown = cv2.subtract(mask, sure_fg)
    markers[unknown == 255] = 0
    img_for_watershed = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_for_watershed, markers)
    split_polygons = []
    for i in range(2, num_labels + 1):
        component_mask = np.zeros(page_shape[:2], dtype="uint8")
        component_mask[markers == i] = 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            split_polygons.append(c)
    return split_polygons

def is_box_inside(inner_box, outer_box):
    """Проверяет, находится ли центр inner_box внутри outer_box"""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box 
    cx = (ix1 + ix2) / 2
    cy = (iy1 + iy2) / 2
    return (ox1 <= cx <= ox2) and (oy1 <= cy <= oy2)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou


class MangaTranslator:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._progress_callback = None
        self._is_cancelled = False
        self.bubble_detector = None
        self.mocr = None
        self.models_loaded = False
        
        self.temp_img_dir = os.path.join(self.config.temp_dir, 'images')
        self.temp_txt_dir = os.path.join(self.config.temp_dir, 'text_results')
        
        os.makedirs(self.config.debug_dir, exist_ok=True)

    def set_progress_callback(self, callback):
        self._progress_callback = callback

    def cancel(self):
        self._is_cancelled = True

    def _check_cancel(self):
        if self._is_cancelled:
            raise Exception("Отменено пользователем")

    def _emit_progress(self, progress: int, message: str, step: int, total: int):
        print(f"[PROGRESS {progress}%] {message}")
        if self._progress_callback:
            self._progress_callback(ProgressInfo(progress, message, step, total))

    def _setup_dirs(self):
        for d in [self.config.temp_dir, self.temp_img_dir, self.temp_txt_dir, self.config.output_dir, self.config.debug_dir]:
            if d != self.config.debug_dir and os.path.exists(d): 
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    def check_requirements(self) -> Dict[str, bool]:
        return {
            "Bubble Model": os.path.exists(self.config.bubble_detector_path),
            "Text Model": os.path.exists(self.config.text_detector_path),
            "Font": os.path.exists(self.config.font_path),
            "Inference Script": model2annotations is not None
        }

    @contextmanager
    def _env_proxy_context(self):
        try:
            if not os.path.exists(self.config.config_file):
                yield
                return
            with open(self.config.config_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                proxy_line = lines[0].strip() if len(lines) > 0 else ""
                api_key = lines[1].strip() if len(lines) > 1 else ""
        except:
            yield
            return

        if not api_key:
            yield
            return

        parts = proxy_line.split(':')
        proxy_url = f"http://{parts[2]}:{parts[3]}@{parts[0]}:{parts[1]}" if len(parts) == 4 else proxy_line
        
        old_http = os.environ.get('http_proxy')
        old_https = os.environ.get('https_proxy')
        old_api = os.environ.get('GOOGLE_API_KEY')
        
        if proxy_url:
            os.environ['http_proxy'] = proxy_url
            os.environ['https_proxy'] = proxy_url
            print(f"[PROXY] Установлен прокси: {proxy_url}")
        os.environ['GOOGLE_API_KEY'] = api_key
        
        try:
            yield
        finally:
            if old_http: os.environ['http_proxy'] = old_http
            else: os.environ.pop('http_proxy', None)
            if old_https: os.environ['https_proxy'] = old_https
            else: os.environ.pop('https_proxy', None)
            if old_api: os.environ['GOOGLE_API_KEY'] = old_api
            else: os.environ.pop('GOOGLE_API_KEY', None)
            print("[PROXY] Настройки сброшены")

    def _load_models_if_needed(self):
        if self.models_loaded: return
        self._emit_progress(5, "Загрузка моделей...", 0, 0)
        
        if os.path.exists(self.config.bubble_detector_path):
            self.bubble_detector = YOLO(self.config.bubble_detector_path)
            print("[MODELS] YOLO загружен")
        
        if MangaOcr is None:
            raise Exception("Библиотека manga_ocr не установлена!")
            
        try:
            if os.path.exists(self.config.ocr_model_path):
                self.mocr = MangaOcr(pretrained_model_name_or_path=self.config.ocr_model_path)
                print("[MODELS] MangaOCR (Local) загружен")
            else:
                self.mocr = MangaOcr()
                print("[MODELS] MangaOCR (HuggingFace) загружен")
        except Exception as e:
            raise Exception(f"Ошибка загрузки MangaOCR: {e}")
            
        self.models_loaded = True

    def _convert_input_to_images(self, input_paths: List[str]) -> List[Image.Image]:
        images = []
        for path in input_paths:
            self._check_cancel()
            print(f"[INPUT] Обработка файла: {path}")
            if path.lower().endswith('.pdf'):
                try:
                    pages = convert_from_path(path, poppler_path=self.config.poppler_path)
                    images.extend(pages)
                    print(f"[INPUT] PDF конвертирован: {len(pages)} страниц")
                except Exception as e:
                    print(f"[ERROR] PDF Error: {e}")
            else:
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"[ERROR] Image Error: {e}")
        return images

    def _run_text_detection(self, images: List[Image.Image]):
        if model2annotations is None:
            raise Exception("Ошибка: inference.py не найден или поврежден.")
            
        self._setup_dirs()
        for i, img in enumerate(images):
            img.save(os.path.join(self.temp_img_dir, f"page_{i:04d}.png"), 'PNG')
            
        print("[DETECTION] Запуск model2annotations...")
        model2annotations(
            self.config.text_detector_path,
            self.temp_img_dir,
            self.temp_txt_dir,
            save_json=False
        )
        print("[DETECTION] Детекция завершена")

    def _save_yolo_debug(self, image: Image.Image, physical_bubbles: List[dict], page_idx: int):
        try:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            overlay = img_cv.copy()
            for pb in physical_bubbles:
                poly = pb['poly']
                cv2.polylines(img_cv, [poly], True, (0, 0, 255), 2)
                cv2.fillPoly(overlay, [poly], (0, 0, 255))
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0, img_cv)
            ts = int(time.time())
            filename = os.path.join(self.config.debug_dir, f"segmentation_page_{page_idx+1}_ts{ts}.png")
            cv2.imwrite(filename, img_cv)
            print(f"[DEBUG] Сегментация сохранена: {filename}")
        except Exception as e:
            print(f"[DEBUG ERROR] {e}")

    def detect_blocks_multiple(self, input_paths: List[str], output_path: str = None, save_visualization: bool = False) -> DetectionResult:
        self._is_cancelled = False
        self._load_models_if_needed()
        self._emit_progress(10, "Загрузка изображений...", 1, 4)
        
        images = self._convert_input_to_images(input_paths)
        if not images: raise Exception("Нет изображений для обработки")
        
        self._emit_progress(30, "Детекция текста...", 2, 4)
        self._run_text_detection(images)
        
        all_gui_boxes = []
        total_blocks = 0
        
        self._emit_progress(70, "Анализ результатов...", 3, 4)
        for i, img in enumerate(images):
            self._check_cancel()
            filename_base = f"page_{i:04d}"
            txt_path = os.path.join(self.temp_txt_dir, f'line-{filename_base}.txt')
            
            text_lines = []
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f.readlines():
                        coords = list(map(int, line.strip().split()))
                        if len(coords) >= 8:
                            text_lines.append(np.array(coords, dtype=np.int32).reshape(4, 2))
            
            groups = group_text_lines(text_lines, self.config.grouping_distance)
            print(f"[PAGE {i+1}] Найдено {len(groups)} текстовых блоков")
            
            for grp in groups:
                pts = np.vstack(grp)
                x, y, w, h = cv2.boundingRect(pts)
                all_gui_boxes.append(BoundingBox(x, y, x+w, y+h, i))
                total_blocks += 1
                
        return DetectionResult(images, all_gui_boxes, total_blocks)

    def process_multiple(self, input_paths: List[str], output_path: str = None) -> str:
        self._is_cancelled = False
        self._load_models_if_needed()
        
        if self.mocr is None:
             raise Exception("OCR модель не загружена!")

        self._emit_progress(10, "Загрузка файлов...", 1, 6)
        images = self._convert_input_to_images(input_paths)
        
        self._emit_progress(20, "Детекция текста...", 2, 6)
        self._run_text_detection(images)
        
        all_text_groups_for_translation = []
        final_render_data = {}
        
        total_pages = len(images)
        for i, page_pil in enumerate(images):
            self._check_cancel()
            self._emit_progress(30 + int((i/total_pages)*20), f"OCR страницы {i+1}...", 3, 6)
            
            filename_base = f"page_{i:04d}"
            txt_path = os.path.join(self.temp_txt_dir, f'line-{filename_base}.txt')
            text_lines = []
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f.readlines():
                        c = list(map(int, line.strip().split()))
                        if len(c) >= 8: text_lines.append(np.array(c, dtype=np.int32).reshape(4, 2))
            
            groups = group_text_lines(text_lines, self.config.grouping_distance)
            logical_bubbles = []
            
            print(f"[OCR PAGE {i+1}] Обработка {len(groups)} групп текста...")
            
            for j, grp in enumerate(groups):
                grp.sort(key=lambda p: -np.mean(p, axis=0)[0])
                full_text = ""
                for poly in grp:
                    x, y, w, h = cv2.boundingRect(poly)
                    crop = page_pil.crop((x, y, x+w, y+h))
                    if self.mocr:
                        full_text += self.mocr(crop)
                
                g_box = cv2.boundingRect(np.vstack(grp))
                uid = f"{filename_base}_g{j}"
                logical_bubbles.append({
                    'unique_id': uid, 'text': full_text,
                    'box': (g_box[0], g_box[1], g_box[0]+g_box[2], g_box[1]+g_box[3]),
                    'polys': grp
                })
            
            all_text_groups_for_translation.extend(logical_bubbles)
            
            phys_bubbles = []
            if self.bubble_detector:
                res = self.bubble_detector(page_pil, verbose=False)[0]
                if res.masks is not None:
                    for poly in res.masks.xy:
                        splits = split_merged_bubbles(np.array(poly, dtype=np.int32), np.array(page_pil).shape)
                        for sp in splits:
                            phys_bubbles.append({'poly': sp, 'box': cv2.boundingRect(sp)})
            
            print(f"[YOLO PAGE {i+1}] Найдено {len(phys_bubbles)} пузырей")
            self._save_yolo_debug(page_pil, phys_bubbles, i)
                            
            page_render = []
            used_indices = set()
            
            for lb in logical_bubbles:
                best_idx = -1
                
                for idx, pb in enumerate(phys_bubbles):
                    if idx in used_indices: continue
                    
                    px, py, pw, ph = pb['box']
                    bubble_rect = (px, py, px+pw, py+ph)
                    
                    if is_box_inside(lb['box'], bubble_rect):
                        best_idx = idx
                        break 
                
                if best_idx == -1:
                    best_iou = 0.0
                    for idx, pb in enumerate(phys_bubbles):
                        if idx in used_indices: continue
                        px, py, pw, ph = pb['box']
                        iou = calculate_iou(lb['box'], (px, py, px+pw, py+ph))
                        if iou > best_iou: 
                            best_iou = iou
                            best_idx = idx
                    
                    if best_iou < 0.01: 
                        best_idx = -1

                if best_idx != -1:
                    erase_poly = phys_bubbles[best_idx]['poly']
                    used_indices.add(best_idx)
                else:
                    all_pts = np.vstack(lb['polys'])
                    erase_poly = cv2.convexHull(all_pts)
                    
                bx, by, bw, bh = lb['box'][0], lb['box'][1], lb['box'][2]-lb['box'][0], lb['box'][3]-lb['box'][1]
                pad = self.config.render_padding
                render_box = (
                    max(0, bx - pad), max(0, by - pad),
                    min(page_pil.width, bx + bw + pad), min(page_pil.height, by + bh + pad)
                )
                page_render.append({'unique_id': lb['unique_id'], 'erase': erase_poly, 'render_box': render_box})
            
            final_render_data[i] = page_render

        self._emit_progress(60, "AI Перевод...", 4, 6)
        translation_map = {}
        
        with self._env_proxy_context():
            genai.configure()
            model = genai.GenerativeModel(self.config.gemini_model)
            
            batch_size = 75 
            items = [item for item in all_text_groups_for_translation if item['text'].strip()]
            print(f"[TRANSLATION] Всего реплик: {len(items)}. Батч: {batch_size}")
            
            for k in range(0, len(items), batch_size):
                self._check_cancel()
                batch = items[k:k+batch_size]
                payload = [{"id": x['unique_id'], "text": x['text']} for x in batch]
                
                print(f"[GEMINI] Отправка батча {k//batch_size + 1}...")
                
                max_retries = 3
                success = False
                
                for attempt in range(max_retries):
                    try:
                        prompt = f"""Translate to Russian JSON list: [{{"id": "...", "translation": "..."}}]. JSON: {json.dumps(payload, ensure_ascii=False)}"""
                        resp = model.generate_content(prompt)
                        clean = resp.text.strip().replace("```json", "").replace("```", "")
                        data = json.loads(clean)
                        for d in data: translation_map[d['id']] = d['translation']
                        success = True
                        print(f"[GEMINI] Батч {k//batch_size + 1} ОК")
                        break 
                    except Exception as e:
                        print(f"[GEMINI ERROR] {e}")
                        if "429" in str(e) or "Resource" in str(e):
                            time.sleep(5 * (attempt + 1))
                        else:
                            time.sleep(2)
                
                if not success: print(f"[GEMINI FAIL] Батч пропущен")
                time.sleep(2)

        self._emit_progress(80, "Рендеринг...", 5, 6)
        final_pages = []

        for i, page_pil in enumerate(images):
            img_cv = cv2.cvtColor(np.array(page_pil), cv2.COLOR_RGB2BGR)
            render_items = final_render_data.get(i, [])
            
            for item in render_items:
                translated_text = translation_map.get(item['unique_id'], "")
                if translated_text and translated_text.strip():
                    cv2.fillPoly(img_cv, [item['erase']], (255, 255, 255))
            
            pil_clean = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_clean)
            
            font_sizes = []
            for item in render_items:
                translated_text = translation_map.get(item['unique_id'], "")
                if translated_text:
                    size = get_optimal_font_size(item['render_box'], translated_text, self.config.font_path, self.config.line_spacing)
                    font_sizes.append(size)
            
            page_font_size = int(np.median(font_sizes)) if font_sizes else 20
            try: page_font = ImageFont.truetype(self.config.font_path, page_font_size)
            except: page_font = ImageFont.load_default()
            
            for item in render_items:
                text = translation_map.get(item['unique_id'], "")
                if text:
                    draw_text_in_box(draw, item['render_box'], text, page_font, self.config.line_spacing)
            
            final_pages.append(pil_clean)

        self._emit_progress(95, "Сохранение...", 6, 6)
        if not output_path: output_path = os.path.join(self.config.output_dir, "translated.pdf")
        if output_path.lower().endswith('.pdf'):
            final_pages[0].save(output_path, "PDF", resolution=150.0, save_all=True, append_images=final_pages[1:])
        else:
            final_pages[0].save(output_path)
        
        print(f"[DONE] Сохранено: {output_path}")    
        return output_path

    def process_with_custom_boxes(self, input_paths: List[str], manual_boxes: List[Any], output_path: str) -> str:
        """Перевод кастомных областей с подробным логированием"""
        print(f"[BACKEND] Start process_with_custom_boxes")
        print(f"[BACKEND] input_paths: {input_paths}")
        print(f"[BACKEND] manual_boxes count: {len(manual_boxes)}")
        for mb in manual_boxes:
            print(f"  Box: page={mb.page_index}, coords=({mb.x1},{mb.y1})-({mb.x2},{mb.y2})")
        
        self._is_cancelled = False
        self._load_models_if_needed()
        
        os.makedirs(self.config.debug_dir, exist_ok=True)
        crops_dir = os.path.join(self.config.temp_dir, 'crops_img')
        crops_txt_dir = os.path.join(self.config.temp_dir, 'crops_txt')
        for d in [crops_dir, crops_txt_dir]:
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        self._emit_progress(5, "Подготовка областей...", 1, 6)
        
        full_images = self._convert_input_to_images(input_paths)
        if not full_images:
            raise Exception("Не удалось загрузить изображения")
        
        print(f"[BACKEND] Загружено {len(full_images)} страниц")
        for i, img in enumerate(full_images):
            print(f"  Page {i}: size={img.size}")
        
        result_images = [img.copy() for img in full_images]
        
        boxes_by_page = {}
        for mb in manual_boxes:
            if mb.page_index not in boxes_by_page: 
                boxes_by_page[mb.page_index] = []
            boxes_by_page[mb.page_index].append(mb)
        
        print(f"[BACKEND] Боксы по страницам: {list(boxes_by_page.keys())}")

        crop_metadata = []
        
        for page_idx, img in enumerate(full_images):
            if page_idx not in boxes_by_page: 
                continue
            
            print(f"[BACKEND] Страница {page_idx}")
            
            for box_idx, mb in enumerate(boxes_by_page[page_idx]):
                x1 = max(0, min(mb.x1, img.width))
                y1 = max(0, min(mb.y1, img.height))
                x2 = max(0, min(mb.x2, img.width))
                y2 = max(0, min(mb.y2, img.height))
                
                print(f"  Box {box_idx}: ({mb.x1},{mb.y1})-({mb.x2},{mb.y2}) -> ({x1},{y1})-({x2},{y2})")
                
                if x2 <= x1 + 5 or y2 <= y1 + 5: 
                    print(f"  Box {box_idx}: SKIP - too small")
                    continue
                
                crop_img = img.crop((x1, y1, x2, y2))
                print(f"  Box {box_idx}: crop size = {crop_img.size}")
                
                ts = int(time.time())
                debug_path = os.path.join(self.config.debug_dir, f"crop_input_p{page_idx}_b{box_idx}_{ts}.png")
                try:
                    crop_img.save(debug_path)
                except: pass
                
                crop_name = f"p{page_idx}_b{box_idx}.png"
                crop_path = os.path.join(crops_dir, crop_name)
                
                try:
                    crop_img.save(crop_path)
                    crop_metadata.append({
                        'crop_path': crop_path,
                        'txt_path': os.path.join(crops_txt_dir, f"line-p{page_idx}_b{box_idx}.txt"),
                        'page_idx': page_idx,
                        'box_coords': (x1, y1, x2, y2),
                        'crop_pil': crop_img.copy(),
                        'id': f"p{page_idx}_b{box_idx}"
                    })
                except Exception as e:
                    print(f"[ERROR] Saving crop: {e}")

        if not crop_metadata:
            print("[BACKEND] No crops, saving originals")
            if len(result_images) == 1:
                result_images[0].save(output_path, "PDF", resolution=150.0)
            else:
                result_images[0].save(output_path, "PDF", resolution=150.0, save_all=True, append_images=result_images[1:])
            return output_path

        print(f"[BACKEND] Создано {len(crop_metadata)} кропов")

        self._emit_progress(20, "Детекция...", 2, 6)
        if model2annotations:
            try:
                model2annotations(
                    self.config.text_detector_path,
                    crops_dir,
                    crops_txt_dir,
                    save_json=False
                )
                print("[BACKEND] Text detection OK")
            except Exception as e:
                print(f"[BACKEND] Detector error: {e}")

        self._emit_progress(40, "OCR...", 3, 6)
        
        items_to_translate = []
        crops_render_data = {} 

        for item in crop_metadata:
            self._check_cancel()
            
            print(f"[BACKEND] Processing: {item['id']}")
            
            text_lines = []
            if os.path.exists(item['txt_path']):
                with open(item['txt_path'], 'r') as f:
                    for line in f.readlines():
                        c = list(map(int, line.strip().split()))
                        if len(c) >= 8: 
                            text_lines.append(np.array(c, dtype=np.int32).reshape(4, 2))
            
            print(f"  Text lines: {len(text_lines)}")
            
            if not text_lines:
                full_text = ""
                if self.mocr:
                    try:
                        full_text = self.mocr(item['crop_pil'])
                        print(f"  Full OCR: '{full_text[:50]}...'" if len(full_text) > 50 else f"  Full OCR: '{full_text}'")
                    except Exception as e:
                        print(f"  OCR error: {e}")
                
                if full_text.strip():
                    uid = f"{item['id']}_full"
                    items_to_translate.append({'unique_id': uid, 'text': full_text})
                    crop_w, crop_h = item['crop_pil'].size
                    crops_render_data[item['id']] = [{
                        'unique_id': uid,
                        'erase': np.array([[0,0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]]),
                        'render_box': (0, 0, crop_w, crop_h)
                    }]
                continue

            groups = group_text_lines(text_lines, self.config.grouping_distance)
            print(f"  Groups: {len(groups)}")
            
            phys_bubbles = []
            if self.bubble_detector:
                try:
                    res = self.bubble_detector(item['crop_pil'], verbose=False)[0]
                    if res.masks is not None:
                        for poly in res.masks.xy:
                            splits = split_merged_bubbles(np.array(poly, dtype=np.int32), np.array(item['crop_pil']).shape)
                            for sp in splits:
                                phys_bubbles.append({'poly': sp, 'box': cv2.boundingRect(sp)})
                    print(f"  YOLO bubbles: {len(phys_bubbles)}")
                except Exception as e: 
                    print(f"  YOLO error: {e}")

            crop_render_items = []
            used_indices = set()
            
            for grp_idx, grp in enumerate(groups):
                grp.sort(key=lambda p: -np.mean(p, axis=0)[0])
                
                full_text = ""
                for poly in grp:
                    x, y, w, h = cv2.boundingRect(poly)
                    sub_crop = item['crop_pil'].crop((x, y, x+w, y+h))
                    if self.mocr:
                        try:
                            full_text += self.mocr(sub_crop)
                        except: pass
                
                uid = f"{item['id']}_g{grp_idx}"
                if full_text.strip():
                    items_to_translate.append({'unique_id': uid, 'text': full_text})
                
                pts = np.vstack(grp)
                g_box = cv2.boundingRect(pts)
                l_rect = (g_box[0], g_box[1], g_box[0]+g_box[2], g_box[1]+g_box[3])
                
                best_idx = -1
                for p_idx, pb in enumerate(phys_bubbles):
                    if p_idx in used_indices: continue
                    px, py, pw, ph = pb['box']
                    p_rect = (px, py, px+pw, py+ph)
                    if is_box_inside(l_rect, p_rect):
                        best_idx = p_idx
                        break
                
                if best_idx == -1:
                    best_iou = 0.0
                    for p_idx, pb in enumerate(phys_bubbles):
                        if p_idx in used_indices: continue
                        px, py, pw, ph = pb['box']
                        iou = calculate_iou(l_rect, (px, py, px+pw, py+ph))
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = p_idx
                    if best_iou < 0.01: best_idx = -1

                if best_idx != -1:
                    erase_poly = phys_bubbles[best_idx]['poly']
                    used_indices.add(best_idx)
                else:
                    erase_poly = cv2.convexHull(np.vstack(grp))

                pad = self.config.render_padding
                crop_w, crop_h = item['crop_pil'].size
                render_box = (
                    max(0, l_rect[0] - pad), max(0, l_rect[1] - pad),
                    min(crop_w, l_rect[2] + pad), min(crop_h, l_rect[3] + pad)
                )
                
                crop_render_items.append({
                    'unique_id': uid, 'erase': erase_poly, 'render_box': render_box
                })
            
            crops_render_data[item['id']] = crop_render_items

        print(f"[BACKEND] Items to translate: {len(items_to_translate)}")

        self._emit_progress(60, "Перевод...", 4, 6)
        translation_map = {}
        
        if items_to_translate:
            with self._env_proxy_context():
                try:
                    genai.configure()
                    model = genai.GenerativeModel(self.config.gemini_model)
                    batch_size = 40
                    
                    for k in range(0, len(items_to_translate), batch_size):
                        batch = items_to_translate[k:k+batch_size]
                        payload = [{"id": x['unique_id'], "text": x['text']} for x in batch]
                        prompt = f"Translate to Russian JSON list: [{{\"id\": \"...\", \"translation\": \"...\"}}]. JSON: {json.dumps(payload, ensure_ascii=False)}"
                        
                        try:
                            resp = model.generate_content(prompt)
                            clean = resp.text.replace("```json", "").replace("```", "").strip()
                            if "[" in clean and "]" in clean:
                                clean = clean[clean.find("["):clean.rfind("]")+1]
                            data = json.loads(clean)
                            for d in data: 
                                translation_map[d['id']] = d['translation']
                                print(f"  Translated: {d['id']}")
                        except Exception as e:
                            print(f"[BACKEND] Gemini error: {e}")
                            time.sleep(1)
                except Exception as e:
                    print(f"[BACKEND] Translation error: {e}")

        print(f"[BACKEND] Translations: {len(translation_map)}")

        self._emit_progress(80, "Сборка...", 5, 6)
        
        font_path = self.config.font_path
        if not os.path.exists(font_path) and sys.platform == "win32":
            font_path = "C:\\Windows\\Fonts\\arial.ttf"

        for item in crop_metadata:
            render_items = crops_render_data.get(item['id'], [])
            
            if not render_items:
                continue
            
            img_cv = cv2.cvtColor(np.array(item['crop_pil']), cv2.COLOR_RGB2BGR)
            
            has_translation = False
            for r_item in render_items:
                trans = translation_map.get(r_item['unique_id'], "")
                if trans:
                    has_translation = True
                    cv2.fillPoly(img_cv, [r_item['erase']], (255, 255, 255))
            
            if not has_translation:
                continue
            
            pil_clean = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_clean)
            
            font_sizes = []
            for r_item in render_items:
                trans = translation_map.get(r_item['unique_id'], "")
                if trans:
                    size = get_optimal_font_size(r_item['render_box'], trans, font_path, self.config.line_spacing)
                    font_sizes.append(size)
            
            median_size = int(np.median(font_sizes)) if font_sizes else 14
            try: 
                font = ImageFont.truetype(font_path, median_size)
            except: 
                font = ImageFont.load_default()
            
            for r_item in render_items:
                trans = translation_map.get(r_item['unique_id'], "")
                if trans:
                    draw_text_in_box(draw, r_item['render_box'], trans, font, self.config.line_spacing)
            
            ts = int(time.time())
            debug_path = os.path.join(self.config.debug_dir, f"crop_output_{item['id']}_{ts}.png")
            try:
                pil_clean.save(debug_path)
            except: pass
            
            target_page = result_images[item['page_idx']]
            x1, y1, x2, y2 = item['box_coords']
            print(f"[BACKEND] Paste {item['id']} to page {item['page_idx']} at ({x1},{y1})")
            target_page.paste(pil_clean, (x1, y1))

        self._emit_progress(95, "Сохранение...", 6, 6)
        print(f"[BACKEND] Saving to {output_path}")
        
        if len(result_images) == 1:
            result_images[0].save(output_path, "PDF", resolution=150.0)
        else:
            result_images[0].save(output_path, "PDF", resolution=150.0, save_all=True, append_images=result_images[1:])
            
        print(f"[BACKEND] Done!")
        return output_path