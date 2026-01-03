import os
import sys
import glob
import cv2
import json
import shutil
import textwrap
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from pdf2image import convert_from_path

# === NumPy патчи ===
try:
    if not hasattr(np, 'float_'): np.float_ = np.float64
    if not hasattr(np, 'float'): np.float = float
    if not hasattr(np, 'int_'): np.int_ = np.int64
    if not hasattr(np, 'int'): np.int = int
    if not hasattr(np, 'bool'): np.bool = bool
    if not hasattr(np, 'bool8'): np.bool8 = np.bool_
except Exception as e:
    print(f"Patch warning: {e}")

sys.path.append(os.getcwd())

# Импорт inference
try:
    from inference import model2annotations
    INFERENCE_AVAILABLE = True
except ImportError:
    print("⚠️ inference.py не найден")
    model2annotations = None
    INFERENCE_AVAILABLE = False

# Импорт MangaOCR
try:
    from manga_ocr import MangaOcr
    MANGA_OCR_AVAILABLE = True
except ImportError:
    print("⚠️ manga_ocr не установлен")
    MangaOcr = None
    MANGA_OCR_AVAILABLE = False


# ================= КОНФИГУРАЦИЯ =================
@dataclass
class Config:
    """Конфигурация приложения"""
    CONFIG_FILE: str = 'config.txt'
    MODEL_NAME: str = "gemini-2.5-flash"
    POPPLER_PATH: str = r'poppler/Library/bin'
    FONT_PATH: str = 'data/fonts/arial.ttf'
    MODEL_PATH: str = r'models/comictextdetector.pt'
    LOCAL_OCR_MODEL: str = 'models/my_manga_ocr_model'
    TEMP_DIR: Path = field(default_factory=lambda: Path('temp'))
    OUTPUT_DIR: Path = field(default_factory=lambda: Path('output'))
    
    def __post_init__(self):
        self.TEMP_DIR = Path(self.TEMP_DIR)
        self.OUTPUT_DIR = Path(self.OUTPUT_DIR)


# ================= СТАТУСЫ =================
class TaskStatus:
    IDLE = "idle"
    PENDING = "pending"
    CONVERTING = "converting"
    DETECTING = "detecting"
    OCR = "ocr"
    TRANSLATING = "translating"
    RENDERING = "rendering"
    SAVING = "saving"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ProgressInfo:
    """Информация о прогрессе"""
    status: str = TaskStatus.IDLE
    progress: int = 0
    message: str = ""
    current_step: int = 0
    total_steps: int = 6


# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================
@contextmanager
def env_proxy(proxy_str: str):
    """Контекстный менеджер для установки прокси"""
    api_key, proxy = proxy_str.split('|')
    old_env = {
        'http_proxy': os.environ.get('http_proxy'),
        'https_proxy': os.environ.get('https_proxy'),
        'GOOGLE_API_KEY': os.environ.get('GOOGLE_API_KEY')
    }
    
    os.environ['http_proxy'] = proxy
    os.environ['https_proxy'] = proxy
    os.environ['GOOGLE_API_KEY'] = api_key
    
    try:
        yield
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_config(filepath: str) -> Optional[str]:
    """Загрузка конфигурации прокси"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            proxy_line = f.readline().strip()
            api_key = f.readline().strip()
        
        if not proxy_line or not api_key:
            return None
        
        parts = proxy_line.split(':')
        if len(parts) == 4:
            proxy_url = f"http://{parts[2]}:{parts[3]}@{parts[0]}:{parts[1]}"
        else:
            proxy_url = proxy_line
        
        return f"{api_key}|{proxy_url}"
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
        return None


def draw_text_in_box(draw: ImageDraw, box: tuple, text: str, font_path: str):
    """Рендеринг текста с автоподбором размера"""
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    
    if box_width <= 0 or box_height <= 0:
        return
    
    font_size = 80
    font = None
    lines = []
    line_height = 20
    
    while font_size > 8:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
            break
        
        # Оценка ширины символа
        test_bbox = font.getbbox('А')
        avg_char_width = test_bbox[2] if test_bbox[2] > 0 else 10
        chars_per_line = max(1, int(box_width * 0.85 / avg_char_width))
        
        # Перенос текста
        wrapped_text = textwrap.fill(text, width=chars_per_line)
        lines = wrapped_text.split('\n')
        
        # Высота строки
        line_height = font.getbbox('Ay')[3] + 4
        text_height = len(lines) * line_height
        
        if text_height < box_height * 0.9:
            break
        
        font_size -= 2
    
    if not font or not lines:
        return
    
    # Рендеринг
    text_height = len(lines) * line_height
    y_offset = box[1] + (box_height - text_height) / 2
    
    for line in lines:
        bbox = font.getbbox(line)
        line_width = bbox[2] - bbox[0]
        x_offset = box[0] + (box_width - line_width) / 2
        
        draw.text((x_offset, y_offset), line, font=font, fill="black")
        y_offset += line_height


# ================= ОСНОВНОЙ КЛАСС ПЕРЕВОДЧИКА =================
class MangaTranslator:
    """Основной класс для перевода манги"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.mocr = None
        self._cancelled = False
        self._progress_callback: Optional[Callable[[ProgressInfo], None]] = None
        
        # Создание директорий
        self.config.TEMP_DIR.mkdir(exist_ok=True)
        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    def set_progress_callback(self, callback: Callable[[ProgressInfo], None]):
        """Установка callback для обновления прогресса"""
        self._progress_callback = callback
    
    def cancel(self):
        """Отмена обработки"""
        self._cancelled = True
    
    def _update_progress(self, status: str, progress: int, message: str, step: int = 0):
        """Обновление прогресса"""
        if self._progress_callback:
            info = ProgressInfo(
                status=status,
                progress=progress,
                message=message,
                current_step=step,
                total_steps=6
            )
            self._progress_callback(info)
    
    def _check_cancelled(self):
        """Проверка отмены"""
        if self._cancelled:
            raise InterruptedError("Операция отменена пользователем")
    
    def check_requirements(self) -> dict:
        """Проверка наличия всех требований"""
        return {
            'font': os.path.exists(self.config.FONT_PATH),
            'model': os.path.exists(self.config.MODEL_PATH),
            'config': os.path.exists(self.config.CONFIG_FILE),
            'inference': INFERENCE_AVAILABLE,
            'manga_ocr': MANGA_OCR_AVAILABLE,
            'poppler': os.path.exists(self.config.POPPLER_PATH)
        }
    
    def _init_ocr(self):
        """Инициализация OCR"""
        if self.mocr is None:
            if os.path.exists(self.config.LOCAL_OCR_MODEL):
                self.mocr = MangaOcr(pretrained_model_name_or_path=self.config.LOCAL_OCR_MODEL)
            else:
                self.mocr = MangaOcr()
    
    def _translate_with_gemini(self, text_data: list) -> dict:
        """Перевод текста через Gemini API"""
        genai.configure()
        model = genai.GenerativeModel(self.config.MODEL_NAME)
        
        lines_to_send = [
            {"id": f"{item['file']}_block_{item['id']}", "text": item['text']} 
            for item in text_data
        ]
        
        prompt = f"""Ты профессиональный переводчик манги с японского на русский. 
Переведи список реплик, сохраняя контекст и стиль речи персонажей.

Вот реплики в JSON:
{json.dumps(lines_to_send, ensure_ascii=False, indent=2)}

ТРЕБОВАНИЯ:
1. Верни ответ ТОЛЬКО в формате JSON
2. Формат: список объектов с полями "id" и "translation"
3. Сохраняй эмоциональность и стиль оригинала
4. Не добавляй никакого дополнительного текста"""
        
        try:
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            raw_text = raw_text.replace("```json", "").replace("```", "")
            translated_list = json.loads(raw_text)
            return {item['id']: item['translation'] for item in translated_list}
        except Exception as e:
            print(f"Ошибка Gemini: {e}")
            return {}
    
    def _perform_ocr(self, temp_img_dir: Path, temp_res_dir: Path) -> list:
        """Выполнение OCR на всех страницах"""
        image_files = sorted(temp_img_dir.glob('*.png'))
        batch_data = []
        total_files = len(image_files)
        
        for idx, img_path in enumerate(image_files):
            self._check_cancelled()
            
            # Обновляем прогресс внутри OCR
            sub_progress = 50 + int((idx / total_files) * 15)
            self._update_progress(
                TaskStatus.OCR, 
                sub_progress, 
                f"OCR страницы {idx + 1}/{total_files}",
                3
            )
            
            filename = img_path.name
            img = cv2.imread(str(img_path))
            h_img, w_img = img.shape[:2]
            
            txt_path = temp_res_dir / f"{img_path.stem}.txt"
            if not txt_path.exists():
                continue
            
            with open(txt_path, 'r', encoding='utf-8') as f:
                blocks_coords = []
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        blocks_coords.append(list(map(float, parts[1:5])))
                    except:
                        continue
                
                blocks_coords.sort(key=lambda b: b[1])
            
            for i, (x_c, y_c, w, h) in enumerate(blocks_coords):
                x1 = max(0, int((x_c - w/2) * w_img))
                y1 = max(0, int((y_c - h/2) * h_img))
                x2 = min(w_img, int((x_c + w/2) * w_img))
                y2 = min(h_img, int((y_c + h/2) * h_img))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop = Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                text = self.mocr(crop)
                
                batch_data.append({
                    'file': filename,
                    'id': i + 1,
                    'text': text,
                    'box': (x1, y1, x2, y2)
                })
        
        return batch_data
    
    def _render_translation(self, pages_pil: list, batch_data: list, translation_map: dict) -> list:
        """Рендеринг переведённого текста на страницы"""
        # Группировка по страницам
        pages_data = {}
        for item in batch_data:
            if item['file'] not in pages_data:
                pages_data[item['file']] = []
            pages_data[item['file']].append(item)
        
        final_pages = []
        total_pages = len(pages_pil)
        
        for i, original_pil_page in enumerate(pages_pil):
            self._check_cancelled()
            
            # Обновляем прогресс
            sub_progress = 80 + int((i / total_pages) * 15)
            self._update_progress(
                TaskStatus.RENDERING, 
                sub_progress, 
                f"Рендеринг страницы {i + 1}/{total_pages}",
                5
            )
            
            filename = f"page_{i:04d}.png"
            
            img_cv = cv2.cvtColor(np.array(original_pil_page), cv2.COLOR_RGB2BGR)
            
            if filename in pages_data:
                # Закрашиваем области
                for item in pages_data[filename]:
                    box = item['box']
                    unique_id = f"{filename}_block_{item['id']}"
                    if translation_map.get(unique_id):
                        cv2.rectangle(
                            img_cv, 
                            (box[0], box[1]), 
                            (box[2], box[3]), 
                            (255, 255, 255), 
                            -1
                        )
                
                # Рендерим текст
                img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                
                for item in pages_data[filename]:
                    box = item['box']
                    unique_id = f"{filename}_block_{item['id']}"
                    translated_text = translation_map.get(unique_id, "")
                    
                    if translated_text:
                        draw_text_in_box(draw, box, translated_text, self.config.FONT_PATH)
                
                final_pages.append(img_pil)
            else:
                final_pages.append(original_pil_page)
        
        return final_pages
    
    def process(self, pdf_path: str, output_path: str = None) -> Optional[str]:
        """
        Основной метод обработки PDF
        
        Args:
            pdf_path: Путь к входному PDF файлу
            output_path: Путь для сохранения результата (опционально)
            
        Returns:
            Путь к результирующему файлу или None при ошибке
        """
        self._cancelled = False
        pdf_path = Path(pdf_path)
        
        # Определяем выходной путь
        if output_path is None:
            output_path = self.config.OUTPUT_DIR / f"{pdf_path.stem}_translated.pdf"
        else:
            output_path = Path(output_path)
        
        # Временные директории для этой задачи
        task_id = pdf_path.stem
        temp_img_dir = self.config.TEMP_DIR / f"{task_id}_images"
        temp_res_dir = self.config.TEMP_DIR / f"{task_id}_results"
        
        try:
            # Очистка и создание временных директорий
            for d in [temp_img_dir, temp_res_dir]:
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True)
            
            # Загрузка конфигурации прокси
            proxy_config = load_config(self.config.CONFIG_FILE)
            if not proxy_config:
                raise ValueError("Ошибка загрузки конфигурации прокси")
            
            with env_proxy(proxy_config):
                # ШАГ 1: Конвертация PDF
                self._check_cancelled()
                self._update_progress(TaskStatus.CONVERTING, 5, "Конвертация PDF в изображения...", 1)
                
                pages_pil = convert_from_path(
                    str(pdf_path), 
                    poppler_path=self.config.POPPLER_PATH
                )
                
                total_pages = len(pages_pil)
                for i, page in enumerate(pages_pil):
                    self._check_cancelled()
                    page.save(temp_img_dir / f"page_{i:04d}.png", 'PNG')
                    progress = 5 + int((i / total_pages) * 20)
                    self._update_progress(
                        TaskStatus.CONVERTING, 
                        progress, 
                        f"Сохранение страницы {i + 1}/{total_pages}",
                        1
                    )
                
                # ШАГ 2: Детекция текста
                self._check_cancelled()
                self._update_progress(TaskStatus.DETECTING, 30, "Детекция текстовых блоков...", 2)
                
                if model2annotations:
                    model2annotations(
                        self.config.MODEL_PATH, 
                        str(temp_img_dir), 
                        str(temp_res_dir), 
                        save_json=False
                    )
                else:
                    raise RuntimeError("Модуль inference недоступен")
                
                # ШАГ 3: OCR
                self._check_cancelled()
                self._update_progress(TaskStatus.OCR, 50, "Инициализация OCR...", 3)
                
                self._init_ocr()
                batch_data = self._perform_ocr(temp_img_dir, temp_res_dir)
                
                if not batch_data:
                    raise ValueError("Текст не найден в документе")
                
                # ШАГ 4: Перевод
                self._check_cancelled()
                self._update_progress(
                    TaskStatus.TRANSLATING, 
                    70, 
                    f"Перевод {len(batch_data)} текстовых блоков...",
                    4
                )
                
                translation_map = self._translate_with_gemini(batch_data)
                
                if not translation_map:
                    raise RuntimeError("Ошибка получения перевода от Gemini")
                
                # ШАГ 5: Рендеринг
                self._check_cancelled()
                self._update_progress(TaskStatus.RENDERING, 80, "Рендеринг перевода...", 5)
                
                final_pages = self._render_translation(pages_pil, batch_data, translation_map)
                
                # ШАГ 6: Сохранение
                self._check_cancelled()
                self._update_progress(TaskStatus.SAVING, 95, "Сохранение PDF...", 6)
                
                if final_pages:
                    final_pages[0].save(
                        output_path, 
                        "PDF", 
                        resolution=150.0, 
                        save_all=True, 
                        append_images=final_pages[1:]
                    )
                
                self._update_progress(TaskStatus.COMPLETED, 100, "Готово!", 6)
                return str(output_path)
                
        except InterruptedError:
            self._update_progress(TaskStatus.CANCELLED, 0, "Операция отменена", 0)
            return None
        except Exception as e:
            self._update_progress(TaskStatus.ERROR, 0, str(e), 0)
            raise
        finally:
            # Очистка временных файлов
            for d in [temp_img_dir, temp_res_dir]:
                if d.exists():
                    shutil.rmtree(d)


# ================= ТЕСТОВЫЙ ЗАПУСК =================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manga Translator CLI")
    parser.add_argument("input", help="Путь к PDF файлу")
    parser.add_argument("-o", "--output", help="Путь для сохранения результата")
    args = parser.parse_args()
    
    def progress_callback(info: ProgressInfo):
        print(f"[{info.progress:3d}%] {info.message}")
    
    translator = MangaTranslator()
    translator.set_progress_callback(progress_callback)
    
    # Проверка требований
    reqs = translator.check_requirements()
    print("Проверка требований:")
    for key, value in reqs.items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}")
    
    if not all(reqs.values()):
        print("\n⚠️ Не все требования выполнены!")
        sys.exit(1)
    
    try:
        result = translator.process(args.input, args.output)
        if result:
            print(f"\n✅ Результат сохранён: {result}")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)