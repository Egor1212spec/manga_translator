import sys
import os
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog, QFrame,
    QMessageBox, QTextEdit, QGroupBox, QSplitter, QStatusBar
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QFont, QIcon, QDragEnterEvent, QDropEvent, QPalette, QColor

from backend import MangaTranslator, Config, ProgressInfo, TaskStatus


# ================= СТИЛИ =================
STYLESHEET = """
QMainWindow {
    background-color: #1a1a2e;
}

QWidget {
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #eaeaea;
}

QLabel {
    color: #eaeaea;
}

QLabel#title {
    font-size: 28px;
    font-weight: bold;
    color: #667eea;
    padding: 10px;
}

QLabel#subtitle {
    font-size: 14px;
    color: #a0a0a0;
    padding-bottom: 20px;
}

QPushButton {
    background-color: #667eea;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #764ba2;
}

QPushButton:pressed {
    background-color: #5a67d8;
}

QPushButton:disabled {
    background-color: #4a4a6a;
    color: #8a8a8a;
}

QPushButton#cancelBtn {
    background-color: #e53e3e;
}

QPushButton#cancelBtn:hover {
    background-color: #c53030;
}

QPushButton#openBtn {
    background-color: #38a169;
}

QPushButton#openBtn:hover {
    background-color: #2f855a;
}

QProgressBar {
    border: none;
    border-radius: 10px;
    background-color: #2d2d44;
    height: 24px;
    text-align: center;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #667eea, stop:1 #764ba2);
    border-radius: 10px;
}

QGroupBox {
    border: 2px solid #3d3d5c;
    border-radius: 10px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 5px;
    color: #667eea;
}

QFrame#dropZone {
    border: 3px dashed #4a4a6a;
    border-radius: 15px;
    background-color: #2d2d44;
    min-height: 200px;
}

QFrame#dropZone:hover {
    border-color: #667eea;
    background-color: #353550;
}

QTextEdit {
    background-color: #2d2d44;
    border: 1px solid #3d3d5c;
    border-radius: 8px;
    padding: 10px;
    color: #b0b0b0;
    font-family: 'Consolas', monospace;
    font-size: 12px;
}

QStatusBar {
    background-color: #16162a;
    color: #808080;
}

QFrame#statusIndicator {
    background-color: #38a169;
    border-radius: 5px;
    min-width: 10px;
    max-width: 10px;
    min-height: 10px;
    max-height: 10px;
}

QFrame#fileInfo {
    background-color: #2d2d44;
    border-radius: 10px;
    padding: 15px;
}
"""


# ================= WORKER THREAD =================
class TranslatorWorker(QThread):
    """Фоновый поток для обработки PDF"""
    
    progress_updated = Signal(ProgressInfo)
    finished = Signal(str)  # Путь к результату
    error = Signal(str)     # Сообщение об ошибке
    
    def __init__(self, translator: MangaTranslator, pdf_path: str, output_path: str = None):
        super().__init__()
        self.translator = translator
        self.pdf_path = pdf_path
        self.output_path = output_path
        
    def run(self):
        try:
            # Устанавливаем callback для прогресса
            self.translator.set_progress_callback(self._on_progress)
            
            result = self.translator.process(self.pdf_path, self.output_path)
            
            if result:
                self.finished.emit(result)
            else:
                self.error.emit("Обработка отменена")
                
        except Exception as e:
            self.error.emit(str(e))
    
    def _on_progress(self, info: ProgressInfo):
        self.progress_updated.emit(info)


# ================= DROP ZONE =================
class DropZone(QFrame):
    """Зона для перетаскивания файлов"""
    
    file_dropped = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # Иконка
        icon_label = QLabel("📄")
        icon_label.setFont(QFont("Segoe UI Emoji", 48))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        # Текст
        text_label = QLabel("Перетащите PDF файл сюда\nили нажмите для выбора")
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setFont(QFont("Segoe UI", 14))
        text_label.setStyleSheet("color: #808080;")
        layout.addWidget(text_label)
        
        self.setCursor(Qt.PointingHandCursor)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.toLocalFile().lower().endswith('.pdf'):
                event.acceptProposedAction()
                self.setStyleSheet("QFrame#dropZone { border-color: #667eea; background-color: #353550; }")
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet("")
    
    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("")
        url = event.mimeData().urls()[0]
        file_path = url.toLocalFile()
        if file_path.lower().endswith('.pdf'):
            self.file_dropped.emit(file_path)
    
    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите PDF файл", 
            "", 
            "PDF Files (*.pdf)"
        )
        if file_path:
            self.file_dropped.emit(file_path)


# ================= ГЛАВНОЕ ОКНО =================
class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        
        self.translator = MangaTranslator()
        self.worker: Optional[TranslatorWorker] = None
        self.current_file: Optional[str] = None
        self.result_path: Optional[str] = None
        
        self.setup_ui()
        self.check_requirements()
    
    def setup_ui(self):
        """Настройка интерфейса"""
        self.setWindowTitle("📚 Manga Translator")
        self.setMinimumSize(700, 600)
        self.resize(800, 700)
        
        # Центральный виджет
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(20)
        
        # Заголовок
        title = QLabel("📚 Manga Translator")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        subtitle = QLabel("Автоматический перевод манги с японского на русский")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)
        
        # Зона загрузки
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.on_file_selected)
        main_layout.addWidget(self.drop_zone)
        
        # Информация о файле
        self.file_info_frame = QFrame()
        self.file_info_frame.setObjectName("fileInfo")
        self.file_info_frame.setVisible(False)
        
        file_info_layout = QHBoxLayout(self.file_info_frame)
        
        self.file_icon = QLabel("📄")
        self.file_icon.setFont(QFont("Segoe UI Emoji", 24))
        file_info_layout.addWidget(self.file_icon)
        
        file_details = QVBoxLayout()
        self.file_name_label = QLabel("Файл не выбран")
        self.file_name_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        file_details.addWidget(self.file_name_label)
        
        self.file_size_label = QLabel("")
        self.file_size_label.setStyleSheet("color: #808080;")
        file_details.addWidget(self.file_size_label)
        
        file_info_layout.addLayout(file_details)
        file_info_layout.addStretch()
        
        self.clear_file_btn = QPushButton("✕")
        self.clear_file_btn.setFixedSize(30, 30)
        self.clear_file_btn.clicked.connect(self.clear_file)
        file_info_layout.addWidget(self.clear_file_btn)
        
        main_layout.addWidget(self.file_info_frame)
        
        # Прогресс
        progress_group = QGroupBox("Прогресс")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Готов к работе")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        main_layout.addWidget(progress_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(15)
        
        self.start_btn = QPushButton("🚀 Начать перевод")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_processing)
        buttons_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("❌ Отмена")
        self.cancel_btn.setObjectName("cancelBtn")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        buttons_layout.addWidget(self.cancel_btn)
        
        self.open_btn = QPushButton("📂 Открыть результат")
        self.open_btn.setObjectName("openBtn")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_result)
        buttons_layout.addWidget(self.open_btn)
        
        main_layout.addLayout(buttons_layout)
        
        # Лог
        log_group = QGroupBox("Лог")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        
        main_layout.addWidget(log_group)
        
        # Статус бар
        self.statusBar().showMessage("Готов")
    
    def check_requirements(self):
        """Проверка требований"""
        reqs = self.translator.check_requirements()
        
        missing = [k for k, v in reqs.items() if not v]
        
        if missing:
            self.log(f"⚠️ Отсутствуют компоненты: {', '.join(missing)}")
            self.statusBar().showMessage(f"Внимание: отсутствуют {len(missing)} компонент(ов)")
        else:
            self.log("✅ Все компоненты доступны")
            self.statusBar().showMessage("Все компоненты готовы")
    
    def log(self, message: str):
        """Добавление сообщения в лог"""
        self.log_text.append(message)
        # Прокрутка вниз
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_file_selected(self, file_path: str):
        """Обработка выбора файла"""
        self.current_file = file_path
        
        # Информация о файле
        file_info = Path(file_path)
        size_mb = file_info.stat().st_size / (1024 * 1024)
        
        self.file_name_label.setText(file_info.name)
        self.file_size_label.setText(f"Размер: {size_mb:.2f} MB")
        
        self.drop_zone.setVisible(False)
        self.file_info_frame.setVisible(True)
        self.start_btn.setEnabled(True)
        
        self.log(f"📁 Выбран файл: {file_info.name}")
    
    def clear_file(self):
        """Очистка выбранного файла"""
        self.current_file = None
        self.drop_zone.setVisible(True)
        self.file_info_frame.setVisible(False)
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Готов к работе")
    
    def start_processing(self):
        """Запуск обработки"""
        if not self.current_file:
            return
        
        # Выбор места сохранения
        default_name = Path(self.current_file).stem + "_translated.pdf"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            default_name,
            "PDF Files (*.pdf)"
        )
        
        if not save_path:
            return
        
        # Обновление UI
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        self.drop_zone.setEnabled(False)
        
        self.log(f"🚀 Начало обработки...")
        
        # Запуск воркера
        self.worker = TranslatorWorker(self.translator, self.current_file, save_path)
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()
    
    def cancel_processing(self):
        """Отмена обработки"""
        if self.worker and self.worker.isRunning():
            self.translator.cancel()
            self.log("⏹️ Отмена обработки...")
    
    def on_progress_updated(self, info: ProgressInfo):
        """Обновление прогресса"""
        self.progress_bar.setValue(info.progress)
        self.status_label.setText(f"[{info.current_step}/{info.total_steps}] {info.message}")
        
        # Логируем смену этапов
        status_messages = {
            TaskStatus.CONVERTING: "📄 Конвертация PDF",
            TaskStatus.DETECTING: "🔍 Детекция текста",
            TaskStatus.OCR: "👁️ Распознавание текста",
            TaskStatus.TRANSLATING: "🌐 Перевод",
            TaskStatus.RENDERING: "🎨 Рендеринг",
            TaskStatus.SAVING: "💾 Сохранение"
        }
        
        if info.status in status_messages and info.progress % 10 == 0:
            pass  # Можно добавить дополнительное логирование
    
    def on_processing_finished(self, result_path: str):
        """Завершение обработки"""
        self.result_path = result_path
        
        self.progress_bar.setValue(100)
        self.status_label.setText("✅ Готово!")
        
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.open_btn.setEnabled(True)
        self.drop_zone.setEnabled(True)
        
        self.log(f"✅ Файл сохранён: {result_path}")
        
        # Уведомление
        QMessageBox.information(
            self,
            "Готово!",
            f"Перевод завершён успешно!\n\nФайл сохранён:\n{result_path}"
        )
    
    def on_processing_error(self, error_msg: str):
        """Ошибка обработки"""
        self.progress_bar.setValue(0)
        self.status_label.setText(f"❌ Ошибка")
        
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.drop_zone.setEnabled(True)
        
        self.log(f"❌ Ошибка: {error_msg}")
        
        if "отменена" not in error_msg.lower():
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Произошла ошибка при обработке:\n\n{error_msg}"
            )
    
    def open_result(self):
        """Открытие результата"""
        if self.result_path and os.path.exists(self.result_path):
            os.startfile(self.result_path) if sys.platform == 'win32' else os.system(f'open "{self.result_path}"')
    
    def closeEvent(self, event):
        """Обработка закрытия окна"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Подтверждение",
                "Обработка ещё выполняется. Вы уверены, что хотите выйти?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.translator.cancel()
                self.worker.wait(3000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ================= ЗАПУСК =================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Применяем стили
    app.setStyleSheet(STYLESHEET)
    
    # Тёмная палитра
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 26, 46))
    palette.setColor(QPalette.WindowText, QColor(234, 234, 234))
    palette.setColor(QPalette.Base, QColor(45, 45, 68))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 80))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(234, 234, 234))
    palette.setColor(QPalette.Button, QColor(102, 126, 234))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.Highlight, QColor(102, 126, 234))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()