
import sys
import os
import shutil
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum, auto

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog, QFrame,
    QMessageBox, QTextEdit, QScrollArea, QCheckBox,
    QSizePolicy, QSpacerItem, QToolTip, QMenu
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer, QPoint, QRect
from PySide6.QtGui import (
    QFont, QPalette, QColor, 
    QPixmap, QImage, QPainter, QPen, QBrush, QMouseEvent, QAction
)

from backend import MangaTranslator, Config, ProgressInfo, TaskStatus, DetectionResult, BoundingBox

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from PIL import Image

STYLESHEET = """
* {
    font-family: 'Segoe UI', 'SF Pro Display', Arial, sans-serif;
}

QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
        stop:0 #0f0f1a, stop:0.5 #1a1a2e, stop:1 #16213e);
}

QScrollArea {
    background: transparent;
    border: none;
}

QScrollBar:vertical {
    border: none;
    background: rgba(255, 255, 255, 0.05);
    width: 8px;
    margin: 0;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: rgba(102, 126, 234, 0.6);
    min-height: 30px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: rgba(102, 126, 234, 0.9);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

QScrollBar:horizontal {
    height: 8px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: rgba(102, 126, 234, 0.6);
    min-width: 30px;
    border-radius: 4px;
}

QLabel {
    color: #e0e0e0;
}

QLabel#title {
    font-size: 24px;
    font-weight: 700;
    color: #667eea;
    padding: 15px 0;
}

QLabel#sectionTitle {
    font-size: 11px;
    font-weight: 600;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 8px 0 4px 0;
}

QLabel#statusLabel {
    font-size: 12px;
    color: #64ffda;
    padding: 5px;
}

QLabel#loadingLabel {
    font-size: 11px;
    color: #f093fb;
    padding: 3px;
}

QLabel#boxCountLabel {
    font-size: 12px;
    color: #ffd93d;
    padding: 5px;
    font-weight: 600;
}

QPushButton#modeBtn {
    background: rgba(45, 45, 68, 0.8);
    color: #a0a0c0;
    border: 2px solid rgba(102, 126, 234, 0.3);
    padding: 12px 16px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 13px;
    text-align: left;
}
QPushButton#modeBtn:hover {
    background: rgba(102, 126, 234, 0.2);
    border-color: rgba(102, 126, 234, 0.6);
    color: white;
}
QPushButton#modeBtn:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #667eea, stop:1 #764ba2);
    border-color: transparent;
    color: white;
}
QPushButton#modeBtn:disabled {
    background: rgba(30, 30, 46, 0.5);
    border-color: rgba(60, 60, 80, 0.3);
    color: #404060;
}

QPushButton {
    background: rgba(53, 53, 80, 0.9);
    color: white;
    border: 1px solid rgba(74, 74, 106, 0.5);
    padding: 10px 16px;
    border-radius: 10px;
    font-weight: 500;
}
QPushButton:hover {
    background: rgba(74, 74, 106, 0.9);
    border-color: rgba(102, 126, 234, 0.5);
}
QPushButton:pressed {
    background: rgba(40, 40, 60, 0.9);
}

QPushButton#actionBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #4facfe, stop:1 #00f2fe);
    border: none;
    color: white;
    font-weight: 600;
}
QPushButton#actionBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #00f2fe, stop:1 #4facfe);
}
QPushButton#actionBtn:disabled {
    background: rgba(60, 60, 80, 0.5);
    color: #505070;
}

QPushButton#translateBoxBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #667eea, stop:1 #764ba2);
    border: none;
    color: white;
    font-weight: 600;
    padding: 10px;
}
QPushButton#translateBoxBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #764ba2, stop:1 #667eea);
}
QPushButton#translateBoxBtn:disabled {
    background: rgba(60, 60, 80, 0.5);
    color: #505070;
}

QPushButton#revertBoxBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #f093fb, stop:1 #f5576c);
    border: none;
    color: white;
    font-weight: 600;
    padding: 10px;
}
QPushButton#revertBoxBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #f5576c, stop:1 #f093fb);
}
QPushButton#revertBoxBtn:disabled {
    background: rgba(60, 60, 80, 0.5);
    color: #505070;
}

QPushButton#clearBoxesBtn {
    background: rgba(255, 87, 87, 0.3);
    border: 1px solid rgba(255, 87, 87, 0.5);
    color: #ff5757;
    font-weight: 500;
    padding: 8px;
}
QPushButton#clearBoxesBtn:hover {
    background: rgba(255, 87, 87, 0.5);
}

QPushButton#clearBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #f093fb, stop:1 #f5576c);
    border: none;
}
QPushButton#clearBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #f5576c, stop:1 #f093fb);
}

QPushButton#downloadBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #11998e, stop:1 #38ef7d);
    border: none;
    color: white;
    font-weight: 700;
    padding: 14px;
    font-size: 14px;
}
QPushButton#downloadBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #38ef7d, stop:1 #11998e);
}
QPushButton#downloadBtn:disabled {
    background: rgba(47, 54, 64, 0.5);
    color: #505060;
}

QFrame#thumbnail {
    background: rgba(45, 45, 68, 0.6);
    border: 2px solid rgba(61, 61, 92, 0.8);
    border-radius: 10px;
}
QFrame#thumbnail:hover {
    border-color: rgba(102, 126, 234, 0.8);
    background: rgba(55, 55, 78, 0.8);
}
QFrame#thumbnail[selected="true"] {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.15);
}
QFrame#thumbnail[checked="true"] {
    border-color: #38ef7d;
    background: rgba(56, 239, 125, 0.1);
}

QCheckBox {
    spacing: 6px;
    color: #c0c0c0;
}
QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border: 2px solid rgba(74, 74, 106, 0.8);
    border-radius: 6px;
    background: rgba(45, 45, 68, 0.8);
}
QCheckBox::indicator:hover {
    border-color: #667eea;
}
QCheckBox::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
        stop:0 #38ef7d, stop:1 #11998e);
    border-color: #38ef7d;
}

QProgressBar {
    border: none;
    border-radius: 6px;
    background: rgba(45, 45, 68, 0.8);
    height: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #667eea, stop:1 #764ba2);
    border-radius: 6px;
}

QTextEdit {
    background: rgba(30, 30, 46, 0.9);
    border: 1px solid rgba(61, 61, 92, 0.5);
    border-radius: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 11px;
    color: #a0a0c0;
    padding: 8px;
}
QTextEdit:focus {
    border-color: rgba(102, 126, 234, 0.5);
}

QFrame#sidebar {
    background: rgba(22, 33, 62, 0.95);
    border-right: 1px solid rgba(102, 126, 234, 0.2);
}

QFrame#rightSidebar {
    background: rgba(22, 33, 62, 0.95);
    border-left: 1px solid rgba(102, 126, 234, 0.2);
}

QFrame#pageCard {
    background: rgba(30, 30, 46, 0.7);
    border: 1px solid rgba(61, 61, 92, 0.5);
    border-radius: 16px;
}
QFrame#pageCard:hover {
    border-color: rgba(102, 126, 234, 0.4);
}

QFrame#toolsFrame {
    background: rgba(102, 126, 234, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 10px;
    padding: 10px;
}

QToolTip {
    background: rgba(30, 30, 46, 0.95);
    color: #e0e0e0;
    border: 1px solid rgba(102, 126, 234, 0.5);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}
"""


class ViewMode(Enum):
    ORIGINAL = auto()
    DETECTION = auto()
    TRANSLATION = auto()

@dataclass
class PageState:
    """Состояние страницы"""
    original_pil: Image.Image
    name: str
    original_pixmap: Optional[QPixmap] = None
    detected_boxes: List[QRect] = field(default_factory=list)
    translated_image: Optional[Image.Image] = None
    translated_pixmap: Optional[QPixmap] = None
    working_image: Optional[Image.Image] = None
    working_pixmap: Optional[QPixmap] = None
    is_detected: bool = False
    is_translated: bool = False
    current_view_mode: ViewMode = ViewMode.ORIGINAL

@dataclass
class ManualBox:
    x1: int
    y1: int
    x2: int
    y2: int
    page_index: int


def pil_to_pixmap(pil_image: Image.Image) -> Optional[QPixmap]:
    """Конвертирует PIL Image в QPixmap"""
    if pil_image is None:
        return None
    
    try:
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        width = pil_image.width
        height = pil_image.height
        
        data = pil_image.tobytes("raw", "RGB")
        bytes_per_line = width * 3
        
        qimage = QImage(data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        if qimage.isNull():
            return None
        
        qimage = qimage.copy()
        pixmap = QPixmap.fromImage(qimage)
        
        if pixmap.isNull():
            return None
        
        return pixmap
        
    except Exception as e:
        print(f"[pil_to_pixmap] ERROR: {e}")
        return None

class PdfImportWorker(QThread):
    page_ready = Signal(object, str, int, int)
    finished = Signal()
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, pdf_paths: List[str], poppler_path: str):
        super().__init__()
        self.pdf_paths = pdf_paths
        self.poppler_path = poppler_path
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if not PDF2IMAGE_AVAILABLE:
                self.error.emit("pdf2image не установлен")
                self.finished.emit()
                return
                
            abs_poppler = None
            if self.poppler_path:
                abs_poppler = os.path.abspath(self.poppler_path)
                if not os.path.exists(abs_poppler):
                    abs_poppler = os.path.join(os.getcwd(), self.poppler_path)

            for pdf_path in self.pdf_paths:
                if self._is_cancelled:
                    break
                    
                pdf_name = Path(pdf_path).stem
                self.progress.emit(f"📄 Импорт: {Path(pdf_path).name}")
                
                try:
                    from pdf2image.pdf2image import pdfinfo_from_path
                    try:
                        info = pdfinfo_from_path(pdf_path, poppler_path=abs_poppler)
                        total_pages = info.get('Pages', 0)
                    except:
                        total_pages = 0
                    
                    page_num = 1
                    while not self._is_cancelled:
                        try:
                            pages = convert_from_path(
                                pdf_path, 
                                poppler_path=abs_poppler,
                                first_page=page_num,
                                last_page=page_num
                            )
                            if not pages:
                                break
                            
                            pil_image = pages[0].convert('RGB')
                            name = f"{pdf_name}_p{page_num}"
                            
                            self.page_ready.emit(pil_image, name, page_num, 
                                                total_pages if total_pages > 0 else page_num)
                            
                            page_num += 1
                            
                            if total_pages > 0 and page_num > total_pages:
                                break
                                
                        except Exception as e:
                            if "page" in str(e).lower() or page_num > 1000:
                                break
                            raise e
                            
                except Exception as e:
                    self.error.emit(f"Ошибка {Path(pdf_path).name}: {e}")
                    
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"Критическая ошибка: {e}")
            self.finished.emit()


class TranslatorWorker(QThread):
    progress_updated = Signal(ProgressInfo)
    finished = Signal(list, list)
    error = Signal(str)
    
    def __init__(self, translator: MangaTranslator, input_paths: List[str], 
                 indices: List[int], output_path: str):
        super().__init__()
        self.translator = translator
        self.input_paths = list(input_paths)
        self.indices = list(indices)
        self.output_path = output_path
        
    def run(self):
        try:
            if not self.input_paths:
                self.error.emit("Нет файлов для перевода")
                return
            
            valid_paths = []
            valid_indices = []
            for path, idx in zip(self.input_paths, self.indices):
                if os.path.exists(path):
                    valid_paths.append(path)
                    valid_indices.append(idx)
                    
            if not valid_paths:
                self.error.emit("Все файлы недоступны")
                return
            
            fresh_translator = MangaTranslator(self.translator.config)
            fresh_translator.set_progress_callback(self._on_progress)
            
            result_path = fresh_translator.process_multiple(valid_paths, self.output_path)
            
            if not result_path or not os.path.exists(result_path):
                self.error.emit("Не удалось создать результат")
                return
            
            images = self._load_result(result_path)
            
            if images:
                self.finished.emit(images, valid_indices)
            else:
                self.error.emit("Не удалось загрузить результат")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Ошибка: {str(e)}")
            
    def _load_result(self, result_path: str) -> List[Image.Image]:
        try:
            ext = Path(result_path).suffix.lower()
            
            if ext == '.pdf':
                if PDF2IMAGE_AVAILABLE:
                    poppler = self.translator.config.poppler_path
                    abs_poppler = None
                    if poppler:
                        abs_poppler = os.path.abspath(poppler)
                        if not os.path.exists(abs_poppler):
                            abs_poppler = os.path.join(os.getcwd(), poppler)
                    
                    pages = convert_from_path(result_path, poppler_path=abs_poppler)
                    result = []
                    for p in pages:
                        result.append(p.convert('RGB'))
                    return result
            else:
                img = Image.open(result_path).convert('RGB')
                return [img]
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
        return []
            
    def _on_progress(self, info: ProgressInfo):
        self.progress_updated.emit(info)


class DetectionWorker(QThread):
    progress_updated = Signal(ProgressInfo)
    finished = Signal(object, list)
    error = Signal(str)
    
    def __init__(self, translator: MangaTranslator, input_paths: List[str], indices: List[int]):
        super().__init__()
        self.translator = translator
        self.input_paths = list(input_paths)
        self.indices = list(indices)
        
    def run(self):
        try:
            if not self.input_paths:
                self.error.emit("Нет файлов для детекции")
                return
            
            valid_paths = []
            valid_indices = []
            for path, idx in zip(self.input_paths, self.indices):
                if os.path.exists(path):
                    valid_paths.append(path)
                    valid_indices.append(idx)
                    
            if not valid_paths:
                self.error.emit("Все файлы недоступны")
                return
            
            self.translator.set_progress_callback(self._on_progress)
            result = self.translator.detect_blocks_multiple(valid_paths, None)
            
            if result is None:
                self.error.emit("Пустой результат детекции")
                return
                
            self.finished.emit(result, valid_indices)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Ошибка: {str(e)}")
            
    def _on_progress(self, info: ProgressInfo):
        self.progress_updated.emit(info)


class ProcessBoxesWorker(QThread):
    """Worker для перевода конкретных областей"""
    progress_updated = Signal(ProgressInfo)
    finished = Signal(list, list) 
    error = Signal(str)
    
    def __init__(self, translator: MangaTranslator, input_paths: List[str], 
                 manual_boxes: List[ManualBox], indices: List[int], output_path: str):
        super().__init__()
        self.translator = translator
        self.input_paths = list(input_paths)
        self.manual_boxes = list(manual_boxes)
        self.indices = list(indices)
        self.output_path = output_path
        
    def run(self):
        try:
            if not self.input_paths:
                self.error.emit("Нет файлов")
                return
                
            fresh_translator = MangaTranslator(self.translator.config)
            fresh_translator.set_progress_callback(self._on_progress)
            
            result_path = fresh_translator.process_with_custom_boxes(
                self.input_paths, self.manual_boxes, self.output_path
            )
            
            if not result_path or not os.path.exists(result_path):
                self.error.emit("Не удалось создать результат")
                return
            
            images = self._load_result(result_path)
            if images:
                self.finished.emit(images, self.indices)
            else:
                self.error.emit("Не удалось загрузить результат")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Ошибка: {str(e)}")
            
    def _load_result(self, result_path: str) -> List[Image.Image]:
        try:
            ext = Path(result_path).suffix.lower()
            if ext == '.pdf':
                if PDF2IMAGE_AVAILABLE:
                    poppler = self.translator.config.poppler_path
                    abs_poppler = None
                    if poppler:
                        abs_poppler = os.path.abspath(poppler)
                        if not os.path.exists(abs_poppler):
                            abs_poppler = os.path.join(os.getcwd(), poppler)
                    pages = convert_from_path(result_path, poppler_path=abs_poppler)
                    return [p.convert('RGB') for p in pages]
            else:
                img = Image.open(result_path).convert('RGB')
                return [img]
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
        return []
            
    def _on_progress(self, info: ProgressInfo):
        self.progress_updated.emit(info)


class ThumbnailWidget(QFrame):
    clicked = Signal(int)
    selection_changed = Signal()
    
    def __init__(self, index: int, pixmap: QPixmap, name: str = ""):
        super().__init__()
        self.index = index
        self.setObjectName("thumbnail")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(130, 170)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.checkbox = QCheckBox()
        self.checkbox.stateChanged.connect(self._on_check_changed)
        header.addStretch()
        header.addWidget(self.checkbox)
        layout.addLayout(header)
        
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(118, 118)
        self._update_pixmap(pixmap)
        layout.addWidget(self.img_label, alignment=Qt.AlignCenter)
        
        display_name = name if name else f"Стр. {index + 1}"
        if len(display_name) > 15:
            display_name = display_name[:12] + "..."
        self.name_label = QLabel(display_name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("color: #8892b0; font-size: 10px; font-weight: 500;")
        self.name_label.setToolTip(name)
        layout.addWidget(self.name_label)
    
    def _update_pixmap(self, pixmap: QPixmap):
        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(scaled)
        
    def update_thumbnail(self, pixmap: QPixmap):
        self._update_pixmap(pixmap)
        
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(e)
        
    def set_active(self, active: bool):
        self.setProperty("selected", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_checked(self, checked: bool):
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(checked)
        self.checkbox.blockSignals(False)
        self._update_style()

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def _on_check_changed(self):
        self._update_style()
        self.selection_changed.emit()
        
    def _update_style(self):
        self.setProperty("checked", self.checkbox.isChecked())
        self.style().unpolish(self)
        self.style().polish(self)


class PageCardWidget(QFrame):
    box_drawn = Signal(int, QRect)
    boxes_changed = Signal(int)
    
    def __init__(self, page_index: int):
        super().__init__()
        self.page_index = page_index
        self.setObjectName("pageCard")
        
        self.original_pixmap: Optional[QPixmap] = None
        self.display_pixmap: Optional[QPixmap] = None
        self.scale_factor: float = 1.0
        
        self.drawing_enabled = False
        self.is_drawing = False
        self.start_point: Optional[QPoint] = None
        self.current_rect: Optional[QRect] = None
        self.boxes: List[QRect] = []
        self.show_boxes = True
        
        self.setMouseTracking(True)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        
        self.image_container = QWidget()
        self.image_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        container_layout = QVBoxLayout(self.image_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image_label.setMouseTracking(True)
        container_layout.addWidget(self.image_label)
        
        layout.addWidget(self.image_container)
        
        self.name_label = QLabel()
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("color: #8892b0; font-size: 12px; font-weight: 500;")
        layout.addWidget(self.name_label)

    def set_pixmap(self, pixmap: QPixmap, name: str = ""):
        self.original_pixmap = pixmap
        self.name_label.setText(name)
        self._update_display()
        
    def _update_display(self):
        if not self.original_pixmap or self.original_pixmap.isNull():
            return
        
        view_w = 900
        view_h = 1200
        
        orig_w = self.original_pixmap.width()
        orig_h = self.original_pixmap.height()
        
        scale_w = view_w / orig_w if orig_w > view_w else 1.0
        scale_h = view_h / orig_h if orig_h > view_h else 1.0
        self.scale_factor = min(scale_w, scale_h)
        
        if self.scale_factor < 1.0:
            new_w = int(orig_w * self.scale_factor)
            new_h = int(orig_h * self.scale_factor)
            self.display_pixmap = self.original_pixmap.scaled(
                new_w, new_h, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        else:
            self.display_pixmap = self.original_pixmap
            self.scale_factor = 1.0
        
        self._render_overlays()

    def _render_overlays(self):
        if not self.display_pixmap:
            return
            
        result = self.display_pixmap.copy()
        
        if self.show_boxes:
            painter = QPainter(result)
            painter.setRenderHint(QPainter.Antialiasing)
            
            for i, box in enumerate(self.boxes):
                sx = int(box.x() * self.scale_factor)
                sy = int(box.y() * self.scale_factor)
                sw = int(box.width() * self.scale_factor)
                sh = int(box.height() * self.scale_factor)
                scaled_rect = QRect(sx, sy, sw, sh)
                
                painter.setBrush(QBrush(QColor(102, 126, 234, 40)))
                painter.setPen(QPen(QColor(102, 126, 234), 2))
                painter.drawRoundedRect(scaled_rect, 4, 4)
                
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.setFont(QFont("Arial", 10, QFont.Bold))
                painter.drawText(scaled_rect.topLeft() + QPoint(5, 15), str(i + 1))
            
            if self.is_drawing and self.current_rect:
                painter.setBrush(QBrush(QColor(255, 165, 0, 50)))
                painter.setPen(QPen(QColor(255, 165, 0), 2, Qt.DashLine))
                painter.drawRoundedRect(self.current_rect, 4, 4)
                
            painter.end()
        
        self.image_label.setFixedSize(result.size())
        self.image_label.setPixmap(result)

    def _get_img_coords(self, global_pos):
        """
        Преобразует ГЛОБАЛЬНЫЕ координаты мыши в координаты внутри картинки.
        Это устраняет проблему смещения из-за отступов (margins) карточки.
        """
        if not self.display_pixmap:
            return None
            
        lbl_pos = self.image_label.mapFromGlobal(global_pos)
        
        x = lbl_pos.x()
        y = lbl_pos.y()
        
        if 0 <= x < self.display_pixmap.width() and 0 <= y < self.display_pixmap.height():
            return QPoint(x, y)
        return None

    def mousePressEvent(self, e: QMouseEvent):
        if self.drawing_enabled and e.button() == Qt.LeftButton and self.display_pixmap:
            pt = self._get_img_coords(e.globalPosition().toPoint())
            if pt:
                self.is_drawing = True
                self.start_point = pt
                self.current_rect = QRect(pt, pt)
                self._render_overlays()
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent):
        if self.is_drawing and self.start_point and self.display_pixmap:
            lbl_pos = self.image_label.mapFromGlobal(e.globalPosition().toPoint())
            x = max(0, min(lbl_pos.x(), self.display_pixmap.width()))
            y = max(0, min(lbl_pos.y(), self.display_pixmap.height()))
            
            self.current_rect = QRect(self.start_point, QPoint(x, y)).normalized()
            self._render_overlays()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent):
        if self.is_drawing and e.button() == Qt.LeftButton:
            self.is_drawing = False
            if self.current_rect and self.current_rect.width() > 5 and self.current_rect.height() > 5:
                real_x = int(self.current_rect.x() / self.scale_factor)
                real_y = int(self.current_rect.y() / self.scale_factor)
                real_w = int(self.current_rect.width() / self.scale_factor)
                real_h = int(self.current_rect.height() / self.scale_factor)
                
                orig_rect = QRect(real_x, real_y, real_w, real_h)
                
                self.boxes.append(orig_rect)
                self.box_drawn.emit(self.page_index, orig_rect)
                self.boxes_changed.emit(len(self.boxes))
            
            self.current_rect = None
            self._render_overlays()
            return
        super().mouseReleaseEvent(e)

    def set_boxes(self, boxes: List[QRect]):
        self.boxes = boxes.copy() if boxes else []
        self._render_overlays()
        self.boxes_changed.emit(len(self.boxes))

    def clear_boxes(self):
        self.boxes.clear()
        self._render_overlays()
        self.boxes_changed.emit(0)


    
    def get_manual_boxes(self) -> List[ManualBox]:
        """
        Возвращаем координаты в системе исходного изображения.
        x2, y2 — правая и нижняя граница (EXCLUSIVE), как ожидает PIL.crop.
        """
        result = []
        for b in self.boxes:
            x1 = b.x()
            y1 = b.y()
            x2 = x1 + b.width()
            y2 = y1 + b.height()
            result.append(ManualBox(x1, y1, x2, y2, self.page_index))
        return result
    
    def remove_last_box(self):
        if self.boxes:
            self.boxes.pop()
            self._render_overlays()
            self.boxes_changed.emit(len(self.boxes))

class ImageViewer(QScrollArea):
    boxes_count_changed = Signal(int)
    
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.layout = QVBoxLayout(self.container)
        self.layout.setSpacing(30)
        self.layout.setContentsMargins(30, 30, 30, 30)
        self.layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.setWidget(self.container)
        
        self.page_cards: List[PageCardWidget] = []
        
    def add_page(self, pixmap: QPixmap, name: str = "") -> PageCardWidget:
        index = len(self.page_cards)
        card = PageCardWidget(index)
        card.set_pixmap(pixmap, name)
        card.boxes_changed.connect(self._on_boxes_changed)
        self.page_cards.append(card)
        self.layout.addWidget(card)
        return card
    
    def _on_boxes_changed(self, count: int):
        total = sum(len(c.boxes) for c in self.page_cards)
        self.boxes_count_changed.emit(total)
        
    def clear_all(self):
        for card in self.page_cards:
            self.layout.removeWidget(card)
            card.deleteLater()
        self.page_cards.clear()

    def set_drawing_mode(self, enabled: bool):
        for card in self.page_cards:
            card.drawing_enabled = enabled
            card.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
            card.show_boxes = True
            card._render_overlays()

    def get_all_manual_boxes(self) -> List[ManualBox]:
        result = []
        for card in self.page_cards:
            result.extend(card.get_manual_boxes())
        return result
    
    def get_total_boxes_count(self) -> int:
        return sum(len(c.boxes) for c in self.page_cards)
    
    def clear_all_boxes(self):
        for card in self.page_cards:
            card.clear_boxes()
    
    def scroll_to_page(self, index: int):
        if 0 <= index < len(self.page_cards):
            self.ensureWidgetVisible(self.page_cards[index], 50, 50)

    def get_visible_page_index(self) -> int:
        if not self.page_cards:
            return -1
        viewport_center = self.verticalScrollBar().value() + (self.viewport().height() / 2)
        for card in self.page_cards:
            geo = card.geometry()
            if geo.top() <= viewport_center <= geo.bottom():
                return card.page_index
        return 0

    def update_page_pixmap(self, index: int, pixmap: QPixmap, name: str = None):
        if 0 <= index < len(self.page_cards):
            card = self.page_cards[index]
            current_name = name if name else card.name_label.text()
            card.set_pixmap(pixmap, current_name)
            card.update()
            card.repaint()
            QApplication.processEvents()

    def set_page_boxes(self, index: int, boxes: List[QRect]):
        if 0 <= index < len(self.page_cards):
            self.page_cards[index].set_boxes(boxes)
            self.page_cards[index].show_boxes = True
            self.page_cards[index]._render_overlays()

    def clear_page_boxes(self, index: int):
        if 0 <= index < len(self.page_cards):
            self.page_cards[index].boxes.clear()
            self.page_cards[index].show_boxes = False
            self.page_cards[index]._render_overlays()


class RightSidebar(QFrame):
    page_selected = Signal(int)
    selection_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.setObjectName("rightSidebar")
        self.setFixedWidth(165)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(8)
        
        title = QLabel("📑 Страницы")
        title.setStyleSheet("font-size: 13px; font-weight: 600; color: #667eea;")
        layout.addWidget(title)
        
        self.select_all_btn = QPushButton("☐ Выбрать всё")
        self.select_all_btn.setStyleSheet("""
            QPushButton { font-size: 11px; padding: 6px; background: rgba(102, 126, 234, 0.15); border: 1px solid rgba(102, 126, 234, 0.3); }
            QPushButton:hover { background: rgba(102, 126, 234, 0.25); }
        """)
        self.select_all_btn.setCheckable(True)
        self.select_all_btn.clicked.connect(self._toggle_select_all)
        layout.addWidget(self.select_all_btn)
        
        self.loading_label = QLabel("")
        self.loading_label.setObjectName("loadingLabel")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.hide()
        layout.addWidget(self.loading_label)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)
        
        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.thumbs_layout = QVBoxLayout(self.container)
        self.thumbs_layout.setSpacing(8)
        self.thumbs_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.scroll.setWidget(self.container)
        
        layout.addWidget(self.scroll)
        
        self.counter_label = QLabel("0 страниц")
        self.counter_label.setStyleSheet("font-size: 10px; color: #8892b0;")
        self.counter_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.counter_label)
        
        self.thumbnails: List[ThumbnailWidget] = []
        
    def add_thumbnail(self, pixmap: QPixmap, name: str = "") -> ThumbnailWidget:
        index = len(self.thumbnails)
        thumb = ThumbnailWidget(index, pixmap, name)
        thumb.clicked.connect(self._on_thumb_clicked)
        thumb.selection_changed.connect(self._on_selection_changed)
        self.thumbnails.append(thumb)
        self.thumbs_layout.addWidget(thumb)
        self._update_counter()
        return thumb
    
    def update_thumbnail(self, index: int, pixmap: QPixmap):
        if 0 <= index < len(self.thumbnails):
            self.thumbnails[index].update_thumbnail(pixmap)
    
    def clear_all(self):
        for thumb in self.thumbnails:
            self.thumbs_layout.removeWidget(thumb)
            thumb.deleteLater()
        self.thumbnails.clear()
        self.select_all_btn.setChecked(False)
        self.select_all_btn.setText("☐ Выбрать всё")
        self._update_counter()
        self.hide_loading()

    def show_loading(self, current: int, total: int):
        self.loading_label.setText(f"⏳ {current}/{total}")
        self.loading_label.show()
        
    def hide_loading(self):
        self.loading_label.hide()

    def _on_thumb_clicked(self, index: int):
        for thumb in self.thumbnails:
            thumb.set_active(thumb.index == index)
        self.page_selected.emit(index)

    def _toggle_select_all(self, checked: bool):
        self.select_all_btn.setText("☑ Снять" if checked else "☐ Выбрать всё")
        for thumb in self.thumbnails:
            thumb.set_checked(checked)
        self.selection_changed.emit()

    def _on_selection_changed(self):
        self.selection_changed.emit()
        all_checked = all(t.is_checked() for t in self.thumbnails) if self.thumbnails else False
        self.select_all_btn.blockSignals(True)
        self.select_all_btn.setChecked(all_checked)
        self.select_all_btn.setText("☑ Снять" if all_checked else "☐ Выбрать всё")
        self.select_all_btn.blockSignals(False)

    def get_selected_indices(self) -> List[int]:
        return [t.index for t in self.thumbnails if t.is_checked()]
        
    def _update_counter(self):
        self.counter_label.setText(f"{len(self.thumbnails)} стр.")


class LeftSidebar(QFrame):
    open_files = Signal()
    clear_all = Signal()
    start_full_translation = Signal()
    translate_boxes = Signal()
    revert_boxes = Signal()
    clear_boxes = Signal()
    download_result = Signal()
    mode_changed = Signal(ViewMode)
    drawing_toggled = Signal(bool)
    
    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(300)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        
        title = QLabel("🎌 Manga Translator")
        title.setObjectName("title")
        layout.addWidget(title)
        
        self.add_btn = QPushButton("📂  Добавить файлы")
        self.add_btn.setObjectName("actionBtn")
        self.add_btn.clicked.connect(self.open_files.emit)
        layout.addWidget(self.add_btn)
        
        self.clear_btn = QPushButton("🗑️  Очистить всё")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self.clear_all.emit)
        layout.addWidget(self.clear_btn)
        
        layout.addSpacing(5)
        layout.addWidget(QLabel("РЕЖИМ ПРОСМОТРА", objectName="sectionTitle"))
        
        self.btn_original = QPushButton("📷  Оригинал")
        self.btn_original.setObjectName("modeBtn")
        self.btn_original.setCheckable(True)
        self.btn_original.setChecked(True)
        self.btn_original.clicked.connect(lambda: self._set_mode(ViewMode.ORIGINAL))
        layout.addWidget(self.btn_original)
        
        self.btn_detection = QPushButton("📦  Детекция")
        self.btn_detection.setObjectName("modeBtn")
        self.btn_detection.setCheckable(True)
        self.btn_detection.setEnabled(False)
        self.btn_detection.clicked.connect(lambda: self._set_mode(ViewMode.DETECTION))
        layout.addWidget(self.btn_detection)
        
        self.btn_translation = QPushButton("✨  Полный перевод")
        self.btn_translation.setObjectName("modeBtn")
        self.btn_translation.setCheckable(True)
        self.btn_translation.setEnabled(False)
        self.btn_translation.clicked.connect(lambda: self._set_mode(ViewMode.TRANSLATION))
        layout.addWidget(self.btn_translation)
        
        layout.addSpacing(5)
        layout.addWidget(QLabel("РУЧНОЕ ВЫДЕЛЕНИЕ", objectName="sectionTitle"))
        
        self.tools_frame = QFrame()
        self.tools_frame.setObjectName("toolsFrame")
        tools_layout = QVBoxLayout(self.tools_frame)
        tools_layout.setContentsMargins(10, 10, 10, 10)
        tools_layout.setSpacing(8)
        
        self.draw_btn = QPushButton("✏️  Ручное выделение")
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self._on_drawing_toggled)
        tools_layout.addWidget(self.draw_btn)
        
        self.box_count_label = QLabel("Выделено областей: 0")
        self.box_count_label.setObjectName("boxCountLabel")
        self.box_count_label.setAlignment(Qt.AlignCenter)
        tools_layout.addWidget(self.box_count_label)
        
        actions_layout = QHBoxLayout()
        
        self.translate_box_btn = QPushButton("✨ Перевести")
        self.translate_box_btn.setObjectName("translateBoxBtn")
        self.translate_box_btn.setEnabled(False)
        self.translate_box_btn.setToolTip("Перевести выделенные области")
        self.translate_box_btn.clicked.connect(self.translate_boxes.emit)
        actions_layout.addWidget(self.translate_box_btn)
        
        self.revert_box_btn = QPushButton("↩️ Отменить")
        self.revert_box_btn.setObjectName("revertBoxBtn")
        self.revert_box_btn.setEnabled(False)
        self.revert_box_btn.setToolTip("Вернуть оригинал в выделенных областях")
        self.revert_box_btn.clicked.connect(self.revert_boxes.emit)
        actions_layout.addWidget(self.revert_box_btn)
        
        tools_layout.addLayout(actions_layout)
        
        self.clear_boxes_btn = QPushButton("🗑️ Очистить выделения")
        self.clear_boxes_btn.setObjectName("clearBoxesBtn")
        self.clear_boxes_btn.setEnabled(False)
        self.clear_boxes_btn.clicked.connect(self.clear_boxes.emit)
        tools_layout.addWidget(self.clear_boxes_btn)
        
        layout.addWidget(self.tools_frame)
        
        layout.addStretch()
        
        self.download_btn = QPushButton("💾  СКАЧАТЬ РЕЗУЛЬТАТ")
        self.download_btn.setObjectName("downloadBtn")
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self.download_result.emit)
        layout.addWidget(self.download_btn)
        
        layout.addSpacing(5)
        
        self.status_label = QLabel("⏳ Готов")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)
        
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        layout.addWidget(self.log)

    def _set_mode(self, mode: ViewMode):
        self.btn_original.setChecked(mode == ViewMode.ORIGINAL)
        self.btn_detection.setChecked(mode == ViewMode.DETECTION)
        self.btn_translation.setChecked(mode == ViewMode.TRANSLATION)
        self.mode_changed.emit(mode)
        
    def _on_drawing_toggled(self, checked: bool):
        self.drawing_toggled.emit(checked)

    def set_mode_buttons_state(self, enabled: bool):
        self.btn_detection.setEnabled(enabled)
        self.btn_translation.setEnabled(enabled)

    def set_status(self, text: str):
        self.status_label.setText(text)

    def set_progress(self, value: int):
        self.progress.setValue(value)

    def append_log(self, text: str):
        self.log.append(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def enable_download(self, enabled: bool):
        self.download_btn.setEnabled(enabled)
        
    def update_box_count(self, count: int):
        self.box_count_label.setText(f"Выделено областей: {count}")
        has_boxes = count > 0
        self.translate_box_btn.setEnabled(has_boxes)
        self.clear_boxes_btn.setEnabled(has_boxes)
        
    def enable_revert(self, enabled: bool):
        self.revert_box_btn.setEnabled(enabled)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎌 Manga Translator AI")
        self.resize(1500, 950)
        self.setMinimumSize(1100, 700)
        
        self.translator = MangaTranslator()
        self.worker: Optional[QThread] = None
        self.pdf_worker: Optional[PdfImportWorker] = None
        
        self.pages: List[PageState] = []
        self.current_mode = ViewMode.ORIGINAL
        
        self._setup_ui()
        self._connect_signals()
        self._check_requirements()
        
    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.sidebar = LeftSidebar()
        layout.addWidget(self.sidebar)
        
        self.viewer = ImageViewer()
        layout.addWidget(self.viewer, stretch=1)
        
        self.right_sidebar = RightSidebar()
        layout.addWidget(self.right_sidebar)
        
    def _connect_signals(self):
        self.sidebar.open_files.connect(self._on_open_files)
        self.sidebar.clear_all.connect(self._on_clear_all)
        self.sidebar.mode_changed.connect(self._on_mode_changed)
        self.sidebar.drawing_toggled.connect(self._on_drawing_toggled)
        self.sidebar.translate_boxes.connect(self._on_translate_boxes)
        self.sidebar.revert_boxes.connect(self._on_revert_boxes)
        self.sidebar.clear_boxes.connect(self._on_clear_boxes)
        self.sidebar.download_result.connect(self._on_download)
        
        self.viewer.boxes_count_changed.connect(self.sidebar.update_box_count)
        
        self.right_sidebar.page_selected.connect(self.viewer.scroll_to_page)
        
    def _check_requirements(self):
        reqs = self.translator.check_requirements()
        missing = [k for k, v in reqs.items() if not v]
        if missing:
            self._log(f"⚠️ Не найдено: {', '.join(missing)}")
        else:
            self._log("✅ Готов к работе")
            
    def _log(self, text: str):
        self.sidebar.append_log(text)
        print(f"[LOG] {text}")
        
    def _set_status(self, text: str):
        self.sidebar.set_status(text)


    def _on_revert_boxes(self):
        """
        Вернуть оригинал в выделенных областях.
        Координаты бокса заданы в системе working_image (то, что видит пользователь).
        Нужно пересчитать в систему original_pil, вырезать оттуда и вставить обратно.
        """
        boxes = self.viewer.get_all_manual_boxes()
        if not boxes:
            QMessageBox.warning(self, "Внимание", "Сначала выделите области для отмены!")
            return

        debug_dir = self.translator.config.debug_dir
        os.makedirs(debug_dir, exist_ok=True)

        reverted_count = 0
        affected_indices = set(b.page_index for b in boxes)

        for global_idx in affected_indices:
            if global_idx >= len(self.pages):
                continue

            page = self.pages[global_idx]
            if not page.working_image or not page.original_pil:
                self._log(f"⚠️ Стр. {global_idx+1}: нет изображений")
                continue

            page_boxes = [b for b in boxes if b.page_index == global_idx]
            if not page_boxes:
                continue

            working_copy = page.working_image.copy()
            original = page.original_pil

            w_work, h_work = working_copy.size
            w_orig, h_orig = original.size

            if w_work == 0 or h_work == 0 or w_orig == 0 or h_orig == 0:
                continue

            did_revert = False

            for i, box in enumerate(page_boxes):
                x1_w = max(0, min(box.x1, w_work))
                y1_w = max(0, min(box.y1, h_work))
                x2_w = max(0, min(box.x2, w_work))
                y2_w = max(0, min(box.y2, h_work))

                if x2_w <= x1_w or y2_w <= y1_w:
                    continue

                box_w = x2_w - x1_w
                box_h = y2_w - y1_w
                if box_w <= 0 or box_h <= 0:
                    continue

                x1_o = int(x1_w * w_orig / w_work)
                y1_o = int(y1_w * h_orig / h_work)
                x2_o = int(x2_w * w_orig / w_work)
                y2_o = int(y2_w * h_orig / h_work)

                x1_o = max(0, min(x1_o, w_orig))
                y1_o = max(0, min(y1_o, h_orig))
                x2_o = max(0, min(x2_o, w_orig))
                y2_o = max(0, min(y2_o, h_orig))

                if x2_o <= x1_o or y2_o <= y1_o:
                    continue

                try:
                    original_crop = original.crop((x1_o, y1_o, x2_o, y2_o))

                    if original_crop.size != (box_w, box_h):
                        original_crop = original_crop.resize((box_w, box_h), Image.LANCZOS)

                    ts = int(time.time())
                    debug_name = f"debug_revert_p{global_idx}_box{i}_{ts}.png"
                    debug_path = os.path.join(debug_dir, debug_name)
                    try:
                        original_crop.save(debug_path)
                        print(f"[DEBUG] Revert: {debug_path}, work=({x1_w},{y1_w})-({x2_w},{y2_w}), orig=({x1_o},{y1_o})-({x2_o},{y2_o})")
                    except Exception as e:
                        print(f"[DEBUG SAVE ERROR] {e}")

                    working_copy.paste(original_crop, (x1_w, y1_w))
                    did_revert = True
                    
                except Exception as e:
                    print(f"[REVERT ERROR] page {global_idx}, box {i}: {e}")
                    import traceback
                    traceback.print_exc()

            if did_revert:
                page.working_image = working_copy
                pixmap = pil_to_pixmap(working_copy)
                if pixmap and not pixmap.isNull():
                    page.working_pixmap = pixmap
                    self.viewer.update_page_pixmap(global_idx, pixmap)
                    self.right_sidebar.update_thumbnail(global_idx, pixmap)
                    reverted_count += 1

        self.viewer.clear_all_boxes()

        if reverted_count > 0:
            self._log(f"↩️ Отменено на {reverted_count} стр.")
        else:
            self._log("⚠️ Нечего отменять")

    def _on_open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Выберите файлы", "",
            "Все (*.png *.jpg *.jpeg *.webp *.pdf)"
        )
        if not paths:
            return
            
        pdfs = [p for p in paths if p.lower().endswith('.pdf')]
        images = [p for p in paths if not p.lower().endswith('.pdf')]
        
        if images:
            self._add_image_files(images)
            
        if pdfs:
            self._import_pdfs(pdfs)
            
    def _add_image_files(self, paths: List[str]):
        added = 0
        for path in paths:
            try:
                pil_image = Image.open(path).convert('RGB')
                pixmap = pil_to_pixmap(pil_image)
                
                if pixmap is None:
                    self._log(f"⚠️ Ошибка: {Path(path).name}")
                    continue
                
                name = Path(path).name
                page = PageState(
                    original_pil=pil_image,
                    name=name,
                    original_pixmap=pixmap,
                    working_image=pil_image.copy(),
                    working_pixmap=pixmap
                )
                self.pages.append(page)
                
                self.viewer.add_page(pixmap, name)
                self.right_sidebar.add_thumbnail(pixmap, name)
                added += 1
            except Exception as e:
                self._log(f"⚠️ Ошибка: {Path(path).name} - {e}")
            
        if added > 0:
            self._log(f"✅ Добавлено: {added}")
            self.sidebar.set_mode_buttons_state(True)
        
    def _import_pdfs(self, pdf_paths: List[str]):
        self._log(f"📄 Импорт PDF...")
        self._set_status("⏳ Импорт...")
        
        poppler = self.translator.config.poppler_path
        
        self.pdf_worker = PdfImportWorker(pdf_paths, poppler)
        self.pdf_worker.page_ready.connect(self._on_pdf_page_ready)
        self.pdf_worker.finished.connect(self._on_pdf_finished)
        self.pdf_worker.progress.connect(self._log)
        self.pdf_worker.error.connect(self._on_error)
        self.pdf_worker.start()
        
    def _on_pdf_page_ready(self, pil_image: Image.Image, name: str, current: int, total: int):
        pixmap = pil_to_pixmap(pil_image)
        
        if pixmap is None:
            self._log(f"⚠️ Ошибка страницы {current}")
            return
        
        page = PageState(
            original_pil=pil_image,
            name=name,
            original_pixmap=pixmap,
            working_image=pil_image.copy(),
            working_pixmap=pixmap
        )
        self.pages.append(page)
        
        self.viewer.add_page(pixmap, name)
        self.right_sidebar.add_thumbnail(pixmap, name)
        self.right_sidebar.show_loading(current, total)
        
        self.sidebar.set_mode_buttons_state(True)
        
        if current == 1:
            self.viewer.scroll_to_page(len(self.pages) - 1)
            
    def _on_pdf_finished(self):
        self.right_sidebar.hide_loading()
        self._set_status("✅ Готов")
        self._log(f"✅ Всего: {len(self.pages)} стр.")
            
    
    def _on_clear_all(self):
        if not self.pages:
            return
        
        if self.pdf_worker and self.pdf_worker.isRunning():
            self.pdf_worker.cancel()
            self.pdf_worker.wait()
            
        reply = QMessageBox.question(self, "Очистка", "Удалить всё?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No:
            return
            
        self.pages.clear()
        self.viewer.clear_all()
        self.right_sidebar.clear_all()
        
        self.sidebar.set_mode_buttons_state(False)
        self.sidebar.enable_download(False)
        self.sidebar.set_progress(0)
        self.sidebar.update_box_count(0)
        self.sidebar.enable_revert(False)
        self.current_mode = ViewMode.ORIGINAL
        self.sidebar.btn_original.setChecked(True)
        self.sidebar.btn_detection.setChecked(False)
        self.sidebar.btn_translation.setChecked(False)
        
        self._log("🗑️ Очищено")
        self._set_status("✅ Готов")
        
    
    def _on_drawing_toggled(self, enabled: bool):
        self.viewer.set_drawing_mode(enabled)
        if enabled:
            self._log("✏️ Режим рисования ВКЛ")
        else:
            self._log("✏️ Режим рисования ВЫКЛ")
            
    def _on_clear_boxes(self):
        self.viewer.clear_all_boxes()
        self._log("🗑️ Выделения очищены")
        
    
    def _get_target_indices(self) -> List[int]:
        selected = self.right_sidebar.get_selected_indices()
        if selected:
            valid = [i for i in selected if 0 <= i < len(self.pages)]
            if valid:
                return valid
        
        current = self.viewer.get_visible_page_index()
        if 0 <= current < len(self.pages):
            return [current]
        elif len(self.pages) > 0:
            return [0]
        return []
            
    def _prepare_files(self, indices: List[int], use_working: bool = False) -> Tuple[List[str], List[int]]:
        """Сохраняет изображения во временную папку"""
        input_dir = os.path.join(self.translator.config.temp_dir, "frontend_input")
        
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        os.makedirs(input_dir, exist_ok=True)
        
        paths = []
        valid_indices = []
        
        for idx in indices:
            if 0 <= idx < len(self.pages):
                page = self.pages[idx]
                if use_working and page.working_image:
                    img_to_save = page.working_image
                else:
                    img_to_save = page.original_pil
                    
                if img_to_save:
                    filename = f"page_{idx:04d}.png"
                    filepath = os.path.join(input_dir, filename)
                    
                    try:
                        img_to_save.save(filepath, "PNG")
                        paths.append(filepath)
                        valid_indices.append(idx)
                    except Exception as e:
                        self._log(f"⚠️ Ошибка: стр. {idx}")
                        
        return paths, valid_indices
        
        
    def _on_mode_changed(self, mode: ViewMode):
        self.current_mode = mode
        target_indices = self._get_target_indices()
        
        if not target_indices:
            self._log("⚠️ Нет страниц")
            return
        
        if mode == ViewMode.ORIGINAL:
            self._apply_original_view(target_indices)
        elif mode == ViewMode.DETECTION:
            self._handle_detection(target_indices)
        elif mode == ViewMode.TRANSLATION:
            self._handle_translation(target_indices)
            
    def _apply_original_view(self, indices: List[int]):
        for i in indices:
            if 0 <= i < len(self.pages):
                page = self.pages[i]
                if page.original_pixmap:
                    self.viewer.update_page_pixmap(i, page.original_pixmap, f"Стр. {i+1}")
                    self.viewer.clear_page_boxes(i)
                    self.right_sidebar.update_thumbnail(i, page.original_pixmap)
                    page.current_view_mode = ViewMode.ORIGINAL
        self._set_status(f"📷 Оригинал")
                
    def _handle_detection(self, indices: List[int]):
        need = [i for i in indices if 0 <= i < len(self.pages) and not self.pages[i].is_detected]
        done = [i for i in indices if 0 <= i < len(self.pages) and self.pages[i].is_detected]
        
        if done:
            self._render_detection(done)
            
        if need:
            self._start_detection(need)
            
    def _handle_translation(self, indices: List[int]):
        need = []
        done = []
        
        for i in indices:
            if 0 <= i < len(self.pages):
                page = self.pages[i]
                if page.is_translated and page.working_pixmap:
                    done.append(i)
                else:
                    need.append(i)
        
        if done:
            self._render_translation(done)
            
        if need:
            self._start_translation(need)
            
    def _render_detection(self, indices: List[int]):
        for i in indices:
            if 0 <= i < len(self.pages):
                page = self.pages[i]
                if page.original_pixmap:
                    self.viewer.update_page_pixmap(i, page.original_pixmap, f"Стр. {i+1} (детекция)")
                    self.viewer.set_page_boxes(i, page.detected_boxes)
                    page.current_view_mode = ViewMode.DETECTION
        self._set_status(f"📦 Детекция")
                    
    def _render_translation(self, indices: List[int]):
        for i in indices:
            if 0 <= i < len(self.pages):
                page = self.pages[i]
                if page.working_pixmap and not page.working_pixmap.isNull():
                    self.viewer.update_page_pixmap(i, page.working_pixmap, f"Стр. {i+1} ✓")
                    self.viewer.clear_page_boxes(i)
                    self.right_sidebar.update_thumbnail(i, page.working_pixmap)
                    page.current_view_mode = ViewMode.TRANSLATION
                    
        self._set_status(f"✨ Перевод")
        QApplication.processEvents()
                    
    
    def _start_detection(self, indices: List[int]):
        files, valid_indices = self._prepare_files(indices)
        
        if not files:
            self._log("⚠️ Нет файлов")
            return
            
        self._log(f"🔍 Детекция {len(files)} стр...")
        self._set_status("⏳ Детекция...")
        self.sidebar.setEnabled(False)
        
        self.worker = DetectionWorker(self.translator, files, valid_indices)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.finished.connect(self._on_detection_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()
        
    def _on_detection_finished(self, result: DetectionResult, indices: List[int]):
        self.sidebar.setEnabled(True)
        self._set_status("✅ Готов")
        
        if result is None:
            self._log("⚠️ Пустой результат")
            return
            
        pages_list = getattr(result, 'pages', []) or []
        boxes_list = getattr(result, 'boxes', []) or []
        
        for rel_idx, global_idx in enumerate(indices):
            if global_idx >= len(self.pages) or rel_idx >= len(pages_list):
                continue
                
            page = self.pages[global_idx]
            page_boxes = []
            for box in boxes_list:
                if hasattr(box, 'page') and box.page == rel_idx:
                    page_boxes.append(QRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1))
            
            page.detected_boxes = page_boxes
            page.is_detected = True
                
        self._log(f"✅ Детекция OK")
        self._render_detection(indices)
        
    
    def _start_translation(self, indices: List[int]):
        files, valid_indices = self._prepare_files(indices)
        
        if not files:
            self._log("⚠️ Нет файлов")
            return
            
        self._log(f"🌐 Перевод {len(files)} стр...")
        self._set_status("⏳ Перевод...")
        self.sidebar.setEnabled(False)
        
        temp_path = os.path.join(self.translator.config.temp_dir, f"result_{int(time.time())}.pdf")
        
        self.worker = TranslatorWorker(self.translator, files, valid_indices, temp_path)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.finished.connect(self._on_translation_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()
        
    def _on_translation_finished(self, images: List[Image.Image], indices: List[int]):
        self.sidebar.setEnabled(True)
        self._set_status("✅ Готов")
        
        if not images:
            self._log("⚠️ Пустой результат")
            return
        
        success_count = 0
        
        for rel_idx, global_idx in enumerate(indices):
            if global_idx >= len(self.pages):
                continue
            if rel_idx >= len(images):
                continue
                
            page = self.pages[global_idx]
            translated_img = images[rel_idx]
            
            page.translated_image = translated_img
            page.working_image = translated_img.copy()
            
            pixmap = pil_to_pixmap(translated_img)
            
            if pixmap and not pixmap.isNull():
                page.translated_pixmap = pixmap
                page.working_pixmap = pixmap
                page.is_translated = True
                success_count += 1
                
        self._log(f"✅ Перевод: {success_count}/{len(indices)} стр.")
        self.sidebar.enable_download(success_count > 0)
        self.sidebar.enable_revert(success_count > 0)
        
        if success_count > 0:
            self._render_translation(indices)
            QTimer.singleShot(100, lambda: self._force_update_ui(indices))
        
        QMessageBox.information(self, "Готово", f"Переведено {success_count} стр.!")
        
    def _force_update_ui(self, indices: List[int]):
        for i in indices:
            if 0 <= i < len(self.pages):
                page = self.pages[i]
                if page.working_pixmap and not page.working_pixmap.isNull():
                    self.viewer.update_page_pixmap(i, page.working_pixmap, f"Стр. {i+1} ✓")
                    self.right_sidebar.update_thumbnail(i, page.working_pixmap)
        
        self.viewer.update()
        QApplication.processEvents()
        
    def _on_translate_boxes(self):
        """Перевести только выделенные области"""
        boxes = self.viewer.get_all_manual_boxes()
        if not boxes:
            QMessageBox.warning(self, "Внимание", "Сначала выделите области для перевода!")
            return
            
        target_page_indices = sorted(list(set(b.page_index for b in boxes)))
        
        files, valid_indices = self._prepare_files(target_page_indices, use_working=True)
        
        if not files:
            self._log("⚠️ Нет файлов для перевода")
            return
            
        global_to_relative = {g_idx: r_idx for r_idx, g_idx in enumerate(valid_indices)}
        
        adjusted_boxes = []
        for box in boxes:
            if box.page_index in global_to_relative:
                new_box = ManualBox(
                    box.x1, box.y1, box.x2, box.y2, 
                    global_to_relative[box.page_index]
                )
                adjusted_boxes.append(new_box)
                
        self._log(f"✨ Перевод {len(adjusted_boxes)} областей...")
        self._set_status("⏳ Перевод областей...")
        self.sidebar.setEnabled(False)
        
        temp_path = os.path.join(self.translator.config.temp_dir, f"boxes_{int(time.time())}.pdf")
        
        self.worker = ProcessBoxesWorker(self.translator, files, adjusted_boxes, valid_indices, temp_path)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.finished.connect(self._on_boxes_translated)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_boxes_translated(self, images: List[Image.Image], indices: List[int]):
        """Обработчик завершения перевода выделенных областей"""
        self.sidebar.setEnabled(True)
        self._set_status("✅ Готов")
        
        if not images:
            self._log("⚠️ Пустой результат перевода")
            return
        
        debug_dir = self.translator.config.debug_dir
        os.makedirs(debug_dir, exist_ok=True)
        
        success_count = 0
        
        print(f"[DEBUG] Получено {len(images)} изображений для {len(indices)} страниц")
        
        for rel_idx, global_idx in enumerate(indices):
            if global_idx >= len(self.pages) or rel_idx >= len(images):
                print(f"[DEBUG] Пропуск: rel_idx={rel_idx}, global_idx={global_idx}")
                continue
                
            page = self.pages[global_idx]
            result_img = images[rel_idx]
            
            try:
                ts = int(time.time())
                debug_path = os.path.join(debug_dir, f"translated_result_p{global_idx}_{ts}.png")
                result_img.save(debug_path)
                print(f"[DEBUG] Результат: {debug_path}, размер: {result_img.size}")
            except Exception as e:
                print(f"[DEBUG ERROR] {e}")
            
            page.working_image = result_img.copy()
            
            pixmap = pil_to_pixmap(result_img)
            if pixmap and not pixmap.isNull():
                page.working_pixmap = pixmap
                page.is_translated = True
                success_count += 1
                
                current_name = page.name if page.name else f"Стр. {global_idx + 1}"
                
                self.viewer.update_page_pixmap(global_idx, pixmap, current_name + " ✓")
                self.right_sidebar.update_thumbnail(global_idx, pixmap)
                
                print(f"[DEBUG] Стр. {global_idx+1} обновлена")
            else:
                self._log(f"⚠️ Стр. {global_idx+1}: ошибка pixmap")
        
        self.viewer.clear_all_boxes()
        
        self._log(f"✅ Переведено {success_count}/{len(indices)} стр.")
        
        if success_count > 0:
            self.sidebar.enable_download(True)
            self.sidebar.enable_revert(True)
            
            QTimer.singleShot(50, lambda: self._force_refresh_after_translate(indices))


    def _force_refresh_after_translate(self, indices: List[int]):
        """Принудительное обновление UI после перевода"""
        for i in indices:
            if 0 <= i < len(self.pages):
                page = self.pages[i]
                if page.working_pixmap:
                    self.viewer.update_page_pixmap(i, page.working_pixmap)
        self.viewer.update()
        self.viewer.repaint()
        QApplication.processEvents()

    def _on_download(self):
        """Скачивание всех страниц"""
        if not self.pages:
            QMessageBox.warning(self, "!", "Нет страниц!")
            return
        
        has_any_work = any(p.working_image for p in self.pages)
        if not has_any_work:
            QMessageBox.warning(self, "!", "Сначала выполните перевод!")
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить", "manga_translated.pdf", "PDF (*.pdf);;PNG (*.png)"
        )
        
        if not save_path:
            return
            
        try:
            all_images = []
            for i, page in enumerate(self.pages):
                if page.working_image:
                    all_images.append(page.working_image)
                else:
                    all_images.append(page.original_pil)
            
            if not all_images:
                QMessageBox.warning(self, "!", "Нет изображений!")
                return
            
            if save_path.lower().endswith('.pdf'):
                if len(all_images) == 1:
                    all_images[0].save(save_path, "PDF", resolution=150.0)
                else:
                    all_images[0].save(
                        save_path, "PDF", 
                        resolution=150.0, 
                        save_all=True, 
                        append_images=all_images[1:]
                    )
                self._log(f"💾 Сохранено {len(all_images)} стр.")
            else:
                all_images[0].save(save_path)
                self._log(f"💾 Сохранена 1 стр.")
                        
            QMessageBox.information(self, "OK", f"Сохранено {len(all_images)} страниц!")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._on_error(f"Ошибка: {e}")
            
    
    def _on_progress(self, info: ProgressInfo):
        self.sidebar.set_progress(info.progress)
        
    def _on_error(self, error: str):
        self.sidebar.setEnabled(True)
        self._set_status("❌ Ошибка")
        self._log(f"❌ {error}")
        QMessageBox.critical(self, "Ошибка", error)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 26, 46))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(30, 30, 46))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 68))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(53, 53, 80))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.Highlight, QColor(102, 126, 234))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
