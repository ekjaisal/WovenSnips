# BSD 3-Clause License

# Copyright © 2024, Jaisal E. K.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
   # list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
   # this list of conditions and the following disclaimer in the documentation
   # and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
   # contributors may be used to endorse or promote products derived from
   # this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import json
import pickle
import logging
import pdfplumber
import requests
import torch
from functools import lru_cache
from pydantic import Field
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from concurrent.futures import (ThreadPoolExecutor, as_completed)
from typing import (Optional, List, Any)
from PySide6.QtCore import (Qt, QThread, Signal, QSize, QRect, QPoint, QTimer, QRectF)
from PySide6.QtGui import (QIcon, QFont, QColor, QPalette, QPainter, QPainterPath, QFontDatabase, QPen, QAction)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QTextEdit, QLabel, QFileDialog, QDialog, QComboBox, QMessageBox, QInputDialog, QScrollArea, QMenu, QDialogButtonBox, QTextBrowser)

LIGHT_THEME = {
    QPalette.Window: "#f5f3ef",
    QPalette.WindowText: "#141414",
    QPalette.Base: "#ffffff",
    QPalette.AlternateBase: "#f3f4f5",
    QPalette.ToolTipBase: "#f3f4f5",
    QPalette.ToolTipText: "#141414",
    QPalette.Text: "#141414",
    QPalette.Button: "#A580AD",
    QPalette.ButtonText: "#141414",
    QPalette.BrightText: "#ff0000",
    QPalette.Link: "#0000ff",
    QPalette.Highlight: "#B698BC",
    QPalette.HighlightedText: "#ffffff",
    QPalette.Mid: "#d0d0d0",
}

DARK_THEME = {
    QPalette.Window: "#2b2b2b",
    QPalette.WindowText: "#ffffff",
    QPalette.Base: "#323232",
    QPalette.AlternateBase: "#424242",
    QPalette.ToolTipBase: "#424242",
    QPalette.ToolTipText: "#ffffff",
    QPalette.Text: "#ffffff",
    QPalette.Button: "#A580AD",
    QPalette.ButtonText: "#141414",
    QPalette.BrightText: "#ff0000",
    QPalette.Link: "#3498db",
    QPalette.Highlight: "#B698BC",
    QPalette.HighlightedText: "#ffffff",
    QPalette.Mid: "#545454",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StraicoLLM(LLM):
    api_key: str = Field()
    model: str = Field()

    def update_credentials(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    class Config:
        arbitrary_types_allowed = True

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = "https://api.straico.com/v0/prompt/completion"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model, "message": prompt}
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
            if "data" in result and "completion" in result["data"]:
                completion = result["data"]["completion"]
                if "choices" in completion and len(completion["choices"]) > 0:
                    return completion["choices"][0]["message"]["content"]
            raise ValueError(f"Unexpected response format: {result}")
        except requests.RequestException as e:
            raise ValueError(f"API request failed: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "straico"

@lru_cache(maxsize=100)
def extract_text_from_file(file_path):
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
        elif file_extension in ['.txt', '.md', '.csv']:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return None

def process_files_in_batches(directory, batch_size=50, progress_callback=None):
    supported_extensions = ['.pdf', '.txt', '.md', '.csv']
    all_files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in supported_extensions]
    total_files = len(all_files)
    documents = []
    processed_files = 0
    
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(extract_text_from_file, os.path.join(directory, file)): file for file in all_files}
        for future in as_completed(future_to_file):
            result = future.result()
            if result:
                documents.append(result)
            processed_files += 1
            if progress_callback:
                progress_callback(int((processed_files / total_files) * 100))
    
    return documents

def create_rag_system(directory, api_key, model, batch_size=50, progress_callback=None):
    documents = process_files_in_batches(directory, batch_size, progress_callback)
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32}
    )
    
    batch_size = 1000
    all_splits = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        splits = text_splitter.create_documents(batch)
        all_splits.extend(splits)
    
    db = FAISS.from_documents(all_splits, embeddings, normalize_L2=True)
    
    retriever = db.as_retriever(search_kwargs={"k": 3, "fetch_k": 5})
    
    straico_llm = StraicoLLM(api_key=api_key, model=model)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=straico_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        verbose=False
    )
    
    return qa_chain, straico_llm, retriever

class DocumentLoader(QThread):
    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, directory, api_key, model):
        super().__init__()
        self.directory = directory
        self.api_key = api_key
        self.model = model

    def run(self):
        try:
            qa_system = create_rag_system(self.directory, self.api_key, self.model, progress_callback=self.progress.emit)
            self.finished.emit(qa_system)
        except Exception as e:
            self.error.emit(str(e))

class SpinningLoader(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(50)
        self.setFixedSize(40, 40)

    def rotate(self):
        self.angle = (self.angle + 10) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.angle)

        for i in range(8):
            painter.rotate(45)
            color = QColor(74, 144, 226, 255 - i * 32)
            painter.setPen(color)
            painter.drawLine(10, 0, 20, 0)

    def start(self):
        self.show()
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.hide()

class ChatWorker(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, qa_system, query):
        super().__init__()
        self.qa_system = qa_system
        self.query = query

    def run(self):
        try:
            response = self.qa_system.invoke({"query": self.query})
            self.finished.emit(response["result"])
        except Exception as e:
            self.error.emit(str(e))

class SaveVectorStoreWorker(QThread):
    finished = Signal(bool)
    error = Signal(str)

    def __init__(self, retriever, file_path):
        super().__init__()
        self.retriever = retriever
        self.file_path = file_path

    def run(self):
        try:
            with open(self.file_path, 'wb') as f:
                pickle.dump(self.retriever.vectorstore, f)
            self.finished.emit(True)
        except Exception as e:
            self.error.emit(str(e))

class LoadVectorStoreWorker(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            with open(self.file_path, 'rb') as f:
                vectorstore = pickle.load(f)
            self.finished.emit(vectorstore)
        except Exception as e:
            self.error.emit(str(e))

class ChatBubble(QWidget):
    def __init__(self, text, is_user, parent=None):
        super().__init__(parent)
        self.text = text
        self.is_user = is_user
        self.init_ui()
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        self.text_label = QLabel(self.text)
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        font = self.text_label.font()
        font.setPointSize(10)
        self.text_label.setFont(font)
        
        self.text_label.setAttribute(Qt.WA_TranslucentBackground)
        self.text_label.setStyleSheet("background-color: #B698BC;")

        if self.is_user:
            layout.addStretch()
            layout.addWidget(self.text_label)
        else:
            layout.addWidget(self.text_label)
            layout.addStretch()

    def show_context_menu(self, position):
        context_menu = QMenu(self)
        copy_action = context_menu.addAction("Copy")
        action = context_menu.exec_(self.mapToGlobal(position))
        if action == copy_action:
            QApplication.clipboard().setText(self.text)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bubble_color = QColor("#A580AD") if self.is_user else QColor("#71A6D2")
        painter.setPen(Qt.NoPen)
        painter.setBrush(bubble_color)

        bubble_rect = self.text_label.geometry().adjusted(-10, -5, 10, 5)
        if self.is_user:
            bubble_rect.moveRight(self.width() - 0)

        path = QPainterPath()
        path.addRoundedRect(bubble_rect, 10, 10)
        painter.drawPath(path)

        painter.setPen(QPen(QColor(0, 0, 0, 30), 1))
        painter.drawPath(path)

    def sizeHint(self):
        return self.layout().sizeHint()

    def minimumSizeHint(self):
        return self.layout().minimumSize()

class ChatWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        self.layout.addStretch()
        self.setStyleSheet("background-color: transparent;")

    def add_message(self, text, is_user):
        bubble = ChatBubble(text, is_user)
        bubble.setStyleSheet(self.styleSheet())
        self.layout.insertWidget(self.layout.count() - 1, bubble)
        QApplication.processEvents()
        if isinstance(self.parent(), QScrollArea):
            scrollbar = self.parent().verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

class WovenSnipsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WovenSnips")
        icon_path = os.path.join(os.path.dirname(__file__), "WovenSnips.ico")
        self.setWindowIcon(QIcon(icon_path))
        self.resize(400, 600)
        self.current_theme = LIGHT_THEME
        self.setup_ui()
        self.setup_menubar()
        self.load_settings()
        self.apply_theme(self.current_theme)
        self.set_roboto_font()
        self.qa_system = None
        self.straico_llm = None
        self.retriever = None
        self.update_clear_session_action()

    def setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.loader_widget = SpinningLoader(self)
        self.loader_widget.setVisible(False)
        loader_layout = QHBoxLayout()
        loader_layout.addStretch()
        loader_layout.addWidget(self.loader_widget)
        loader_layout.addStretch()
        layout.addLayout(loader_layout)

        self.chat_scroll = QScrollArea()
        self.chat_widget = ChatWidget()
        self.chat_scroll.setWidget(self.chat_widget)
        self.chat_scroll.setWidgetResizable(True)
        layout.addWidget(self.chat_scroll)

        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit(self)
        self.chat_input.setPlaceholderText("Type input here...")
        self.chat_input.setMinimumHeight(40)
        chat_input_layout.addWidget(self.chat_input)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.chat_input.returnPressed.connect(self.send_message)
        chat_input_layout.addWidget(self.send_button)

        layout.addLayout(chat_input_layout)
        self.set_roboto_font()

    def setup_menubar(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        file_menu = menubar.addMenu("File")
        load_docs_action = QAction("Load Corpus", self)
        load_docs_action.triggered.connect(self.load_documents)
        file_menu.addAction(load_docs_action)
        
        save_pkl_action = QAction("Save Vector Store", self)
        save_pkl_action.triggered.connect(self.save_vector_store)
        file_menu.addAction(save_pkl_action)

        load_pkl_action = QAction("Load Vector Store", self)
        load_pkl_action.triggered.connect(self.load_vector_store)
        file_menu.addAction(load_pkl_action)
        
        self.clear_session_action = QAction("Clear Session", self)
        self.clear_session_action.triggered.connect(self.clear_session)
        self.clear_session_action.setEnabled(False)
        file_menu.addAction(self.clear_session_action)

        settings_menu = menubar.addMenu("Settings")
        api_key_action = QAction("Set API Key", self)
        api_key_action.triggered.connect(self.set_api_key)
        settings_menu.addAction(api_key_action)

        model_action = QAction("Set Model", self)
        model_action.triggered.connect(self.set_model)
        settings_menu.addAction(model_action)

        help_menu = menubar.addMenu("Help")
        tips_action = QAction("Getting Started", self)
        tips_action.triggered.connect(self.show_tips)
        help_menu.addAction(tips_action)

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        self.theme_action = QAction("Dark Theme", self)
        self.theme_action.setCheckable(True)
        self.theme_action.triggered.connect(self.toggle_theme)
        settings_menu.addAction(self.theme_action)

    def load_documents(self):
        if not self.check_initial_setup():
            return
        
        directory = QFileDialog.getExistingDirectory(self, "Select Corpus Directory")
        if directory:
            self.loader_widget.start()
            self.loader = DocumentLoader(directory, self.api_key, self.model)
            self.loader.finished.connect(self.on_documents_loaded)
            self.loader.error.connect(self.on_loading_error)
            self.loader.start()

    def on_documents_loaded(self, result):
        self.qa_system, self.straico_llm, self.retriever = result
        self.loader_widget.stop()
        self.chat_widget.add_message("Corpus loaded successfully!", False)
        
        reply = QMessageBox.question(self, 'Save Vector Store', 
                                     "Do you want to save the vector store for future use?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.save_vector_store()
        self.update_clear_session_action()

    def on_loading_error(self, error):
        self.loader_widget.stop()
        self.chat_widget.add_message(f"Error: Failed to load corpus - {error}", False)
        
    def save_vector_store(self):
        if not self.retriever:
            QMessageBox.warning(self, "Error", "No vector store to save. Please load a corpus or vector store first!")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Vector Store", "", "Pickle Files (*.pkl)")
        if file_path:
            self.loader_widget.start()
            self.save_worker = SaveVectorStoreWorker(self.retriever, file_path)
            self.save_worker.finished.connect(self.on_vector_store_saved)
            self.save_worker.error.connect(self.on_vector_store_save_error)
            self.save_worker.start()

    def on_vector_store_saved(self):
        self.loader_widget.stop()
        self.chat_widget.add_message("Vector store saved successfully!", False)
        
    def on_vector_store_save_error(self, error):
        self.loader_widget.stop()
        self.chat_widget.add_message(f"Error: Failed to save vector store - {error}", False)

    def load_vector_store(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Vector Store", "", "Pickle Files (*.pkl)")
        if file_path:
            self.loader_widget.start()
            self.load_worker = LoadVectorStoreWorker(file_path)
            self.load_worker.finished.connect(self.on_vector_store_loaded)
            self.load_worker.error.connect(self.on_vector_store_load_error)
            self.load_worker.start()
            
    def on_vector_store_loaded(self, vectorstore):
        self.loader_widget.stop()
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "fetch_k": 5})
        self.recreate_qa_system()
        self.chat_widget.add_message("Vector store loaded successfully!", False)
        self.update_clear_session_action()

    def on_vector_store_load_error(self, error):
        self.loader_widget.stop()
        self.chat_widget.add_message(f"Error: Failed to load vector store - {error}", False)

    def clear_session(self):
        reply = QMessageBox.question(self, 'Clear Session', 
                                     "Are you sure you want to clear the current session?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.chat_widget.layout.removeWidget(self.chat_widget.layout.itemAt(0).widget())
            self.chat_widget = ChatWidget()
            self.chat_scroll.setWidget(self.chat_widget)
            self.qa_system = None
            self.straico_llm = None
            self.retriever = None
            self.vector_store_saved_message_shown = False
            self.chat_input.clear()
            self.chat_widget.add_message("Session cleared!", False)
        self.update_clear_session_action()
            
    def update_clear_session_action(self):
        has_session = bool(self.qa_system or self.retriever or self.chat_widget.layout.count() > 1)
        self.clear_session_action.setEnabled(has_session)

    def set_api_key(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Set API Key")
        layout = QVBoxLayout(dialog)
        
        current_key_label = QLabel(f"Current API Key: {'●' * len(self.api_key) if self.api_key else 'Not set'}")
        layout.addWidget(current_key_label)
        
        new_key_input = QLineEdit()
        new_key_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(new_key_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        remove_button = QPushButton("Remove Key")
        buttons.addButton(remove_button, QDialogButtonBox.ActionRole)
        
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        remove_button.clicked.connect(lambda: self.remove_api_key(dialog))
        
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.Accepted:
            new_key = new_key_input.text().strip()
            if new_key:
                self.api_key = new_key
                self.save_settings()
                QMessageBox.information(self, "API Key Updated", "API Key has been updated successfully.")
            elif not self.api_key:
                QMessageBox.warning(self, "No API Key", "Please enter an API Key.")
        if self.straico_llm:
            self.straico_llm.update_credentials(self.api_key, self.model)

    def remove_api_key(self, dialog):
        self.api_key = ''
        self.save_settings()
        dialog.close()
        QMessageBox.information(self, "API Key Removed", "API Key has been removed.")

    def set_model(self):
        models = [
            ("openai/gpt-4o-mini", "OpenAI: GPT-4o mini"),
            ("google/gemma-2-27b-it", "Google: Gemma 2 27B"),
            ("qwen/qwen-2-72b-instruct", "Qwen 2 72B Instruct"),
            ("deepseek/deepseek-coder", "DeepSeek Coder V2"),
            ("meta-llama/llama-3-70b-instruct:nitro", "Meta: Llama 3 70B Instruct (Nitro)"),
            ("anthropic/claude-3-haiku:beta", "Anthropic: Claude 3 Haiku"),
            ("meta-llama/llama-3.1-405b-instruct", "Meta: Llama 3.1 405B Instruct"),
            ("google/gemini-pro-1.5", "Google: Gemini Pro 1.5"),
            ("openai/gpt-4o", "OpenAI: GPT-4o"),
            ("anthropic/claude-3.5-sonnet", "Anthropic: Claude 3.5 Sonnet")
        ]

        current_index = 0
        for i, (model_name, _) in enumerate(models):
            if model_name == self.model:
                current_index = i
                break

        current_model = next((friendly for model_name, friendly in models if model_name == self.model), "No model set")

        dialog = QDialog(self)
        dialog.setWindowTitle("Set Model")
        layout = QVBoxLayout(dialog)

        current_label = QLabel(f"Current Model: {current_model}")
        layout.addWidget(current_label)

        combo = QComboBox()
        combo.addItems([friendly for _, friendly in models])
        combo.setCurrentIndex(current_index)
        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.Accepted:
            new_friendly = combo.currentText()
            self.model = next(model_name for model_name, friendly in models if friendly == new_friendly)
            self.save_settings()
            QMessageBox.information(self, "Model Updated", f"Model set to: {new_friendly}")

            if self.qa_system:
                self.recreate_qa_system()

            if self.straico_llm:
                self.straico_llm.update_credentials(self.api_key, self.model)

    def recreate_qa_system(self):
        if not self.check_initial_setup():
            return
        
        if self.retriever:
            self.straico_llm = StraicoLLM(api_key=self.api_key, model=self.model)
            self.qa_system = RetrievalQA.from_chain_type(
                llm=self.straico_llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=False
            )
        else:
            QMessageBox.warning(self, "Warning", "Please load a corpus or vector store.")

    def set_roboto_font(self):
        font = QFont("Roboto", 9)
        self.setFont(font)
        for child in self.findChildren(QWidget):
            child.setFont(font)

    def apply_theme(self, theme):
        palette = QPalette()
        for role, color in theme.items():
            palette.setColor(role, QColor(color))
        self.setPalette(palette)
        QApplication.setPalette(palette)

        menubar_style = f"""
        QMenuBar {{
            background-color: {theme[QPalette.Window]};
            color: {theme[QPalette.WindowText]};
        }}
        QMenuBar::item:selected {{
            background-color: {theme[QPalette.Highlight]};
        }}
        QMenu {{
            background-color: {theme[QPalette.Base]};
            color: {theme[QPalette.Text]};
        }}
        QMenu::item:selected {{
            background-color: {theme[QPalette.Highlight]};
            color: {theme[QPalette.HighlightedText]};
        }}
        QComboBox {{
            background-color: {theme[QPalette.Base]};
            color: {theme[QPalette.Text]};
        }}
        QComboBox QAbstractItemView {{
            background-color: {theme[QPalette.Base]};
            color: {theme[QPalette.Text]};
            selection-background-color: {theme[QPalette.Highlight]};
            selection-color: {theme[QPalette.HighlightedText]};
        }}
        """
        self.setStyleSheet(menubar_style)

        self.chat_input.setStyleSheet(f"""
        QLineEdit {{
            background-color: {theme[QPalette.Base]};
            color: {theme[QPalette.Text]};
            border: 1px solid {theme[QPalette.Mid]};
            padding: 5px;
        }}
        """)

        self.send_button.setStyleSheet(f"""
        QPushButton {{
            background-color: {theme[QPalette.Button]};
            color: {theme[QPalette.ButtonText]};
            border: none;
            padding: 5px 15px;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            background-color: {theme[QPalette.Highlight]};
        }}
        """)

        self.loader_widget.setStyleSheet(f"""
            background-color: {theme[QPalette.Window]};
        """)
        
        self.chat_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {theme[QPalette.Base]};
            }}
            QScrollBar:vertical {{
                border: none;
                background: {theme[QPalette.Mid]};
                width: 10px;
                margin: 0px 0px 0px 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {theme[QPalette.Button]};
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
            }}
        """)

        self.centralWidget().setStyleSheet(f"background-color: {theme[QPalette.Window]};")
        
        menu_style = f"""
        QMenu {{
            background-color: {theme[QPalette.Base]};
            color: {theme[QPalette.Text]};
            border: 1px solid {theme[QPalette.Mid]};
        }}
        QMenu::item:selected {{
            background-color: {theme[QPalette.Highlight]};
            color: {theme[QPalette.HighlightedText]};
        }}
        """
        
        self.setStyleSheet(self.styleSheet() + menu_style)

    def show_scrollable_html_dialog(self, title, html_content):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumSize(400, 300)

        layout = QVBoxLayout(dialog)

        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setHtml(html_content)

        layout.addWidget(text_browser)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.exec()

    def show_tips(self):
        tips_html = """
        <h2>Getting Started with WovenSnips</h2>
        <hr />
        <h3>Initial Setup</h3>
        <p>Select the preferred model to interact with from <strong>Settings → Set Model</strong>. This preference can be changed at any time, even while a session is in progress.</p>
        <p>Set the Straico API Key from <strong>Settings → Set API Key</strong>. Existing Straico users can find their API Key from the platform&#39;s settings page. New users may choose to create a Straico account using this <a href='https://platform.straico.com/signup?fpr=jaisal'>referral link</a>.</p>
        <hr />
        <h3>Load Corpus</h3>
        <p>Load the collection of files to be used as source material for Retrieval-Augmented Generation from <strong>File → Load Corpus → Select Corpus Directory</strong>. WovenSnips currently supports files with the following extensions to be included in the corpus directory: <code>.pdf</code>, <code>.txt</code>, <code>.md</code>, <code>.csv</code></p>
        <hr />
        <h3>Pickling and Unpickling of Vector Stores</h3>
        <p>WovenSnips allows users to save a loaded corpus as a vector store for future reuse by pickling it in <code>.pkl</code> format from <strong>File → Save Vector Store</strong>. Users can unpickle it from <strong>File → Load Vector Store → your_file.pkl</strong> for reuse. Loading a pickled vector store would significantly save initial processing time and resources compared to loading the corpora every time from scratch.</p>
        <p><em><strong>Note:</strong> WovenSnips vector stores have a minimum pickled size of 75-90 MB. The increase from the baseline is commensurate with the corpus size.</em></p>
        <hr />
        <h3>Clear Session</h3>
        <p>Start a fresh session of interaction by clearing the loaded corpus and vector files from <strong>File → Clear Session</strong>.</p>
        <p><em><strong>Note:</strong> The API Key, model, and theme preferences persist across sessions and remain unaffected on clearing sessions or application relaunch. Users can manually remove the API Key or change the model and theme preferences from Settings.</em></p>
        <hr />
        <h3>Themes</h3>
        <p>WovenSnips supports light (default) and dark themes. Toggle the dark theme on/off from <strong>Settings → Dark Theme</strong>.</p>
        """
        self.show_scrollable_html_dialog("Getting Started", tips_html)

    def show_about(self):
        about_html = """
        <h2>WovenSnips <small>v1.0.0</small></h2>        
        <p>A Lightweight, Free, and Open-source Implementation of Retrieval-Augmented Generation (RAG) using Straico API</p>
        <hr>        
        <h3>License</h3>        
        <p>WovenSnips is licensed under the <strong>BSD 3-Clause License</strong></p>       
        <p>Copyright &copy; 2024, Jaisal E. K.</p>       
        <p>Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:</p>       
        <ol>
        <li>Redistributions of source code must retain the above copyright notice, this
        list of conditions and the following disclaimer.</li>       
        <li>Redistributions in binary form must reproduce the above copyright notice,
        this list of conditions and the following disclaimer in the documentation
        and/or other materials provided with the distribution.</li>       
        <li>Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.</li>
        </ol>        
        <p>THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.</p>
        <hr>
        <h3>Third-Party Libraries and Services</h3>
        <p>WovenSnips is developed using <a href="https://www.python.org/downloads/release/python-3119">Python 3.11.9</a> and several modules from the Python Standard Library (PSF License). In addition, it uses the following third-party libraries and services:</p>
        <ul>
        <li><a href="https://straico.com">Straico</a> via API (<a href="https://documenter.getpostman.com/view/5900072/2s9YyzddrR">Documentation</a>)</li>
        <li><a href="https://pypi.org/project/PySide6">PySide6</a> (LGPL License)</li>
        <li><a href="https://github.com/pytorch/pytorch">PyTorch</a> (BSD-3-Clause License)</li>
        <li><a href="https://pypi.org/project/langchain">Langchain</a> (MIT License)</li>
        <li><a href="https://pypi.org/project/langchain-community">LangChain Community</a> (MIT License)</li>
        <li><a href="https://pypi.org/project/langchain-huggingface">Langchain Hugging Face</a> (MIT License)</li>
        <li><a href="https://github.com/kyamagu/faiss-wheels">FAISS-CPU</a> (MIT License)</li>
        <li><a href="https://github.com/pydantic/pydantic">Pydantic</a> (MIT License)</li>
        <li><a href="https://github.com/jsvine/pdfplumber">pdfplumber</a> (MIT License)</li>
        <li><a href="https://github.com/psf/requests">Requests</a> (Apache License, Version 2.0)</li>
        <li><a href="https://fonts.google.com/specimen/Roboto">Roboto Font</a> (Apache License, Version 2.0)</li>
        <li><a href="https://pyinstaller.org">PyInstaller</a> (GPL 2.0 License, with an Exception)</li>
        <li><a href="https://jrsoftware.org">Inno Setup</a> (Inno Setup License)</li>
        </ul>
        <p>Many thanks to the developers and contributors of these libraries and services. Full documentation and license details can be found at the links provided.</p>
        <hr>
        <h3>Acknowledgements</h3>
        <p>WovenSnips has benefitted significantly from the assistance of Anthropic's <a href="https://www.anthropic.com/news/claude-3-5-sonnet">Claude 3.5 Sonnet</a> with all the heavy lifting associated with coding.</p>
        """
        self.show_scrollable_html_dialog("About WovenSnips", about_html)

    def send_message(self):
        query = self.chat_input.text().strip()
        if not query:
            return
        if not self.qa_system and not self.retriever:
            QMessageBox.warning(self, "Error", "Please load a corpus or vector store to proceed!")
            return
        if not self.api_key:
            self.prompt_for_api_key()
            if not self.api_key:
                return
        if not self.qa_system:
            self.recreate_qa_system()

        self.chat_widget.add_message(query, True)
        self.update_clear_session_action()
        self.chat_input.clear()
        self.worker = ChatWorker(self.qa_system, query)
        self.worker.finished.connect(self.display_response)
        self.worker.error.connect(self.display_error)
        self.worker.start()
        
    def prompt_for_api_key(self):
        api_key, ok = QInputDialog.getText(
            self, "API Key Required", "Please enter your API Key:", 
            QLineEdit.Password
        )
        if ok and api_key:
            self.api_key = api_key
            self.save_settings()
            self.recreate_qa_system()
        else:
            QMessageBox.warning(self, "No API Key", "An API Key is required to proceed.")
            
        if ok and api_key:
            self.api_key = api_key
            self.save_settings()
            if self.straico_llm:
                self.straico_llm.update_credentials(self.api_key, self.model)

    def display_response(self, response):
        self.chat_widget.add_message(response, False)
        self.update_clear_session_action()

    def display_error(self, error):
        self.chat_widget.add_message(f"Error: {error}", False)

    def toggle_theme(self):
        if self.current_theme == LIGHT_THEME:
            self.current_theme = DARK_THEME
        else:
            self.current_theme = LIGHT_THEME
        self.apply_theme(self.current_theme)
        self.save_settings()

    def load_settings(self):
        if os.path.exists('settings.json'):
            with open('settings.json', 'r') as f:
                settings = json.load(f)
            self.api_key = settings.get('api_key', '')
            self.model = settings.get('model', '')
            self.current_theme = DARK_THEME if settings.get('dark_theme', False) else LIGHT_THEME
            self.theme_action.setChecked(self.current_theme == DARK_THEME)
        else:
            self.api_key = ''
            self.model = ''
            self.current_theme = LIGHT_THEME
            
    def check_initial_setup(self):
        if not self.api_key or not self.model:
            return self.show_initial_setup_dialog()
        return True

    def show_initial_setup_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Initial Setup")
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Welcome to WovenSnips! Please set up your preferences:"))
        layout.addWidget(QLabel("Select a model:"))

        model_combo = QComboBox()
        models = [
            ("openai/gpt-4o-mini", "OpenAI: GPT-4o mini"),
            ("google/gemma-2-27b-it", "Google: Gemma 2 27B"),
            ("qwen/qwen-2-72b-instruct", "Qwen 2 72B Instruct"),
            ("deepseek/deepseek-coder", "DeepSeek Coder V2"),
            ("meta-llama/llama-3-70b-instruct:nitro", "Meta: Llama 3 70B Instruct (Nitro)"),
            ("anthropic/claude-3-haiku:beta", "Anthropic: Claude 3 Haiku"),
            ("meta-llama/llama-3.1-405b-instruct", "Meta: Llama 3.1 405B Instruct"),
            ("google/gemini-pro-1.5", "Google: Gemini Pro 1.5"),
            ("openai/gpt-4o", "OpenAI: GPT-4o"),
            ("anthropic/claude-3.5-sonnet", "Anthropic: Claude 3.5 Sonnet")
        ]
        model_combo.addItems([friendly for _, friendly in models])
        layout.addWidget(model_combo)

        layout.addWidget(QLabel("Enter your API Key:"))
        api_key_input = QLineEdit()
        api_key_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(api_key_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setStyleSheet(self.styleSheet())

        result = dialog.exec()
        if result == QDialog.Accepted:
            selected_model = model_combo.currentText()
            api_key = api_key_input.text().strip()
            if not selected_model or not api_key:
                QMessageBox.warning(self, "Error", "Please select a model and enter an API Key.")
                return False
            self.model = next(model_name for model_name, friendly in models if friendly == selected_model)
            self.api_key = api_key
            self.save_settings()
            return True
        return False

    def save_settings(self):
        settings = {
            'api_key': self.api_key,
            'model': self.model,
            'dark_theme': self.current_theme == DARK_THEME
        }
        with open('settings.json', 'w') as f:
            json.dump(settings, f)
    
def main():
    app = QApplication(sys.argv)
    font_path = os.path.join(os.path.dirname(__file__), "fonts", "Roboto-Regular.ttf")
    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id != -1:
        font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        app.setFont(QFont(font_family, 9))
    else:
        print("Error: Failed to load Roboto font. Using system default.")
    
    window = WovenSnipsGUI()
    window.show()
    return app.exec()

if __name__ == '__main__':
    main()
