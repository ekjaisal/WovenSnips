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
import shutil
import sqlite3
import base64
import logging
import tempfile
import threading
from functools import lru_cache
from urllib.parse import (urlparse, parse_qs)
from http.server import (HTTPServer, BaseHTTPRequestHandler)
from concurrent.futures import (ThreadPoolExecutor, as_completed)
from typing import (Optional, List, Any)

import faiss
import torch
import requests
import msgpack
import pdfplumber
import numpy as np
from pydantic import Field
from langchain.llms.base import LLM
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings

from PySide6.QtCore import (Qt, Signal, QTimer, QRectF)
from PySide6.QtGui import (QIcon, QFont, QColor, QPalette, QPainter, QPainterPath, QFontDatabase, QPen, QAction)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QFileDialog, QDialog, QComboBox, QMessageBox, QInputDialog, QScrollArea, QMenu, QDialogButtonBox, QTextBrowser)

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

prompt_template = PromptTemplate(
    template="""Rely solely on the provided context for accurate answers. If the context lacks the answer, respond with "I don't know." Except for terms and phrases, quote verbatim only when essential.

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

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

class TempDatabase:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'temp_corpus.db')
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.setup_tables()

    def setup_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                metadata TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                embedding BLOB,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        self.conn.commit()

    def add_document(self, content, metadata):
        self.cursor.execute('INSERT INTO documents (content, metadata) VALUES (?, ?)',
                            (content, json.dumps(metadata)))
        return self.cursor.lastrowid

    def add_embedding(self, document_id, embedding):
        self.cursor.execute('INSERT INTO embeddings (document_id, embedding) VALUES (?, ?)',
                            (document_id, embedding.tobytes()))

    def get_document(self, doc_id):
        self.cursor.execute('SELECT content, metadata FROM documents WHERE id = ?', (doc_id,))
        content, metadata = self.cursor.fetchone()
        return content, json.loads(metadata)

    def get_all_documents(self):
        self.cursor.execute('SELECT id, content, metadata FROM documents')
        for row in self.cursor.fetchall():
            yield row[0], row[1], json.loads(row[2])

    def get_all_embeddings(self):
        self.cursor.execute('SELECT id, embedding FROM embeddings')
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in self.cursor.fetchall()]

    def iter_documents(self):
        self.cursor.execute('SELECT content FROM documents')
        for row in self.cursor.fetchall():
            yield row[0].encode('utf-8')

    def add_document_chunk(self, chunk):
        self.cursor.execute('INSERT INTO documents (content) VALUES (?)', (chunk.decode('utf-8'),))
        self.conn.commit()

    def close(self):
        self.conn.close()
        shutil.rmtree(self.temp_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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

def process_files_in_batches(directory, temp_db, batch_size=50):
    supported_extensions = ['.pdf', '.txt', '.md', '.csv']
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    skipped_files = []
    processed_files = []
    
    for file in all_files:
        file_path = os.path.join(directory, file)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension not in supported_extensions:
            skipped_files.append((file, "Unsupported file type"))
            continue
        
        if file_extension == '.pdf':
            result = process_pdf(file_path, temp_db)
            if result == "empty":
                skipped_files.append((file, "Empty or non-machine readable PDF"))
            else:
                processed_files.append(file)
        elif file_extension in ['.txt', '.md', '.csv']:
            process_text_file(file_path, temp_db, batch_size)
            processed_files.append(file)
    
    return processed_files, skipped_files

def process_pdf(file_path, temp_db):
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                return "empty"
            
            content_found = False
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    content_found = True
                    metadata = {"source": file_path, "page": page_num + 1}
                    temp_db.add_document(text, metadata)
            
            if not content_found:
                return "empty"
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}")
        return "empty"

def process_text_file(file_path, temp_db, batch_size):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = ""
        for i, line in enumerate(file):
            content += line
            if (i + 1) % batch_size == 0:
                metadata = {"source": file_path, "start_line": i - batch_size + 1, "end_line": i + 1}
                temp_db.add_document(content, metadata)
                content = ""
        if content:
            metadata = {"source": file_path, "start_line": i - len(content.splitlines()) + 1, "end_line": i + 1}
            temp_db.add_document(content, metadata)

def create_rag_system(directory, api_key, model):
    with TempDatabase() as temp_db:
        processed_files, skipped_files = process_files_in_batches(directory, temp_db)
        
        if not processed_files:
            raise ValueError("No valid files were processed. Please check your corpus.")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 32}
        )
        
        documents = []
        for doc_id, content, metadata in temp_db.get_all_documents():
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        
        db = FAISS.from_documents(splits, embeddings)
        
        retriever = db.as_retriever(search_kwargs={"k": 5, "fetch_k": 10})
        
        straico_llm = StraicoLLM(api_key=api_key, model=model)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=straico_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt_template},
            verbose=False
        )
        
        return qa_chain, straico_llm, retriever, skipped_files

class DocumentLoader(threading.Thread):
    def __init__(self, directory, api_key, model):
        super().__init__()
        self.directory = directory
        self.api_key = api_key
        self.model = model
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = create_rag_system(self.directory, self.api_key, self.model)
        except Exception as e:
            self.error = str(e)

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

class WovenSnipsSerializer:
    @classmethod
    def serialize(cls, obj):
        if isinstance(obj, FAISS):
            return msgpack.packb({
                'type': 'FAISS',
                'index': cls.serialize_faiss_index(obj.index),
                'docstore': cls.serialize_docstore(obj.docstore._dict),
                'embedding_function': cls.serialize_embedding_function(obj.embedding_function),
                'index_to_docstore_id': {str(k): v for k, v in obj.index_to_docstore_id.items()}
            })
        return msgpack.packb(obj)

    @classmethod
    def deserialize(cls, data):
        unpacked = msgpack.unpackb(data)
        if isinstance(unpacked, dict) and 'type' in unpacked and unpacked['type'] == 'FAISS':
            index = cls.deserialize_faiss_index(unpacked['index'])
            docstore = cls.deserialize_docstore(unpacked['docstore'])
            embedding_function = cls.deserialize_embedding_function(unpacked['embedding_function'])
            index_to_docstore_id = {int(k): v for k, v in unpacked['index_to_docstore_id'].items()}
            
            return FAISS(
                embedding_function=embedding_function,
                index=index,
                docstore=InMemoryDocstore(docstore),
                index_to_docstore_id=index_to_docstore_id
            )
        return unpacked

    @classmethod
    def serialize_faiss_index(cls, index):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            faiss.write_index(index, temp_file.name)
            temp_file.flush()
            with open(temp_file.name, 'rb') as f:
                serialized = base64.b64encode(f.read()).decode()
        os.unlink(temp_file.name)
        return serialized

    @classmethod
    def deserialize_faiss_index(cls, index_data):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(base64.b64decode(index_data))
            temp_file.flush()
            index = faiss.read_index(temp_file.name)
        os.unlink(temp_file.name)
        return index

    @classmethod
    def serialize_docstore(cls, docstore):
        return {k: cls.serialize_document(v) for k, v in docstore.items()}

    @classmethod
    def deserialize_docstore(cls, docstore):
        return {k: cls.deserialize_document(v) for k, v in docstore.items()}

    @classmethod
    def serialize_document(cls, doc):
        return {
            'page_content': doc.page_content,
            'metadata': doc.metadata
        }

    @classmethod
    def deserialize_document(cls, doc_dict):
        return Document(
            page_content=doc_dict['page_content'],
            metadata=doc_dict['metadata']
        )

    @classmethod
    def serialize_embedding_function(cls, embedding_function):
        if isinstance(embedding_function, HuggingFaceEmbeddings):
            return {
                'type': 'HuggingFaceEmbeddings',
                'model_name': embedding_function.model_name,
                'model_kwargs': embedding_function.model_kwargs,
                'encode_kwargs': embedding_function.encode_kwargs
            }
        else:
            raise ValueError(f"Unsupported embedding function type: {type(embedding_function)}")

    @classmethod
    def deserialize_embedding_function(cls, data):
        if data['type'] == 'HuggingFaceEmbeddings':
            return HuggingFaceEmbeddings(
                model_name=data['model_name'],
                model_kwargs=data['model_kwargs'],
                encode_kwargs=data['encode_kwargs']
            )
        else:
            raise ValueError(f"Unsupported embedding function type: {data['type']}")

class SaveVectorStoreWorker(threading.Thread):
    def __init__(self, retriever, file_path):
        super().__init__()
        self.retriever = retriever
        self.file_path = file_path
        self.success = False
        self.error = None

    def run(self):
        try:
            serialized_data = WovenSnipsSerializer.serialize(self.retriever.vectorstore)
            with open(self.file_path, 'wb') as f:
                f.write(serialized_data)
            self.success = True
        except Exception as e:
            self.error = str(e)

class LoadVectorStoreWorker(threading.Thread):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.result = None
        self.error = None

    def run(self):
        try:
            with open(self.file_path, 'rb') as f:
                serialized_data = f.read()
            self.result = WovenSnipsSerializer.deserialize(serialized_data)
        except Exception as e:
            self.error = str(e)

class ChatWorker(threading.Thread):
    def __init__(self, qa_system, query):
        super().__init__()
        self.qa_system = qa_system
        self.query = query
        self.result = None
        self.error = None

    def run(self):
        try:
            response = self.qa_system.invoke({"query": self.query})
            self.result = response["result"]
        except Exception as e:
            self.error = str(e)

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
        if parent is None:
            parent = QApplication.instance().activeWindow()
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

class WovenSnipsLocalServer:
    def __init__(self, woven_snips_instance, port=60606):
        self.woven_snips = woven_snips_instance
        self.port = port
        self.server = None
        self.server_thread = None

    def start(self):
        if self.server:
            return

        class ThreadedHTTPServer(HTTPServer):
            def __init__(self, *args, **kwargs):
                self.woven_snips = None
                super().__init__(*args, **kwargs)

        class ChatHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    query = urlparse(self.path).query
                    params = parse_qs(query)

                    if 'p' in params:
                        prompt = params['p'][0]
                        
                        if prompt.startswith("load_corpus:"):
                            directory = prompt[12:].strip()
                            response = self.server.woven_snips.load_corpus_local_server(directory)
                        elif prompt.startswith("load_vs:"):
                            vector_store_path = prompt[8:].strip()
                            response = self.server.woven_snips.load_vector_store_local_server(vector_store_path)
                        elif prompt.startswith("save_vs:"):
                            vector_store_path = prompt[8:].strip()
                            response = self.server.woven_snips.save_vector_store_local_server(vector_store_path)
                        elif prompt == "list_models":
                            response = self.server.woven_snips.list_models_local_server()
                        elif prompt.startswith("select_model:"):
                            model_name = prompt[13:].strip()
                            response = self.server.woven_snips.select_model_local_server(model_name)
                        elif prompt == "clear_session":
                            response = self.server.woven_snips.clear_session_local_server()
                        else:
                            response = self.server.woven_snips.process_chat(prompt)
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(response.encode())
                    else:
                        self.send_error(400, 'Missing "prompt" parameter')
                except Exception as e:
                    self.send_error(500, str(e))
            
            def log_message(self, format, *args):
                pass

        server_address = ('', self.port)
        self.server = ThreadedHTTPServer(server_address, ChatHandler)
        self.server.woven_snips = self.woven_snips
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"Server started on http://localhost:{self.port}/")

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            self.server_thread = None
            print("Server stopped")

    def is_running(self):
        return self.server is not None

class WovenSnipsGUI(QMainWindow):
    clear_session_signal = Signal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WovenSnips")
        icon_path = os.path.join(os.path.dirname(__file__), "WovenSnips.ico")
        self.setWindowIcon(QIcon(icon_path))
        self.resize(400, 600)
        self.current_theme = LIGHT_THEME
        self.available_models = []
        self.qa_system = None
        self.straico_llm = None
        self.retriever = None
        self.local_server = WovenSnipsLocalServer(self)
        
        self.setup_ui()
        self.setup_menubar()
        self.load_settings()
        self.apply_theme(self.current_theme)
        self.set_roboto_font()
        self.update_models()
        self.update_server_status_indicator()
        self.clear_session_signal.connect(self.clear_session_gui)

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
        
        self.server_status_indicator = QLabel()
        self.server_status_indicator.setFixedSize(10, 10)
        self.server_status_indicator.setStyleSheet("background-color: #949494; border-radius: 6px;")
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Local Server Status:"))
        status_layout.addWidget(self.server_status_indicator)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        layout.addLayout(chat_input_layout)
        self.set_roboto_font()

    def setup_menubar(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        file_menu = menubar.addMenu("File")
        load_docs_action = QAction("Load Corpus", self)
        load_docs_action.triggered.connect(self.load_documents)
        file_menu.addAction(load_docs_action)
        
        save_vs_action = QAction("Save Vector Store", self)
        save_vs_action.triggered.connect(self.save_vector_store)
        file_menu.addAction(save_vs_action)

        load_vs_action = QAction("Load Vector Store", self)
        load_vs_action.triggered.connect(self.load_vector_store)
        file_menu.addAction(load_vs_action)
        
        self.clear_session_action = QAction("Clear Session", self)
        self.clear_session_action.triggered.connect(self.clear_session_gui)
        file_menu.addAction(self.clear_session_action)

        settings_menu = menubar.addMenu("Settings")
        api_key_action = QAction("Set API Key", self)
        api_key_action.triggered.connect(self.set_api_key)
        settings_menu.addAction(api_key_action)

        model_action = QAction("Select Model", self)
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
        
        self.server_action = QAction("Local Server", self)
        self.server_action.setCheckable(True)
        self.server_action.triggered.connect(self.toggle_local_server)
        settings_menu.addAction(self.server_action)

    def load_documents(self):
        if not self.check_initial_setup():
            return
        
        directory = QFileDialog.getExistingDirectory(self, "Select Corpus Directory")
        if directory:
            self.loader_widget.start()
            self.loader = DocumentLoader(directory, self.api_key, self.model)
            self.loader.start()
            self.check_loader_thread()

    def check_loader_thread(self):
        if self.loader.is_alive():
            QTimer.singleShot(100, self.check_loader_thread)
        else:
            self.loader_widget.stop()
            if self.loader.error:
                self.on_loading_error(self.loader.error)
            else:
                self.on_documents_loaded(self.loader.result)

    def on_documents_loaded(self, result):
        self.qa_system, self.straico_llm, self.retriever, skipped_files = result
        self.loader_widget.stop()
        self.chat_widget.add_message("Corpus loaded successfully.", False)
        
        if skipped_files:
            skipped_files_message = self.format_skipped_files_message(skipped_files)
            self.chat_widget.add_message(skipped_files_message, False)
        
        reply = QMessageBox.question(self, 'Save Vector Store', 
                                     "Do you want to save the vector store for future use?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.save_vector_store()

    def format_skipped_files_message(self, skipped_files):
        message = "The following files were skipped:\n\n"
        for file, reason in skipped_files:
            message += f"{file}: {reason}\n"
        return message

    def on_loading_error(self, error):
        self.loader_widget.stop()
        self.chat_widget.add_message(f"Error: Failed to load corpus - {error}", False)
        
    def save_vector_store(self):
        if not self.retriever:
            QMessageBox.warning(self, "Error", "No vector store to save. Please load a corpus or vector store first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Vector Store", "", "WovenSnips Files (*.wov)")
        if file_path:
            if not file_path.endswith('.wov'):
                file_path += '.wov'
            self.loader_widget.start()
            self.save_worker = SaveVectorStoreWorker(self.retriever, file_path)
            self.save_worker.start()
            self.check_save_thread()
            
    def check_save_thread(self):
        if self.save_worker.is_alive():
            QTimer.singleShot(100, self.check_save_thread)
        else:
            if self.save_worker.error:
                self.on_vector_store_save_error(self.save_worker.error)
            else:
                self.on_vector_store_saved()

    def on_vector_store_saved(self):
        self.loader_widget.stop()
        self.chat_widget.add_message("Vector store saved successfully.", False)
        
    def on_vector_store_save_error(self, error):
        self.loader_widget.stop()
        self.chat_widget.add_message(f"Error: Failed to save vector store - {error}", False)

    def load_vector_store(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Vector Store", "", "WovenSnips Files (*.wov)")
        if file_path:
            self.loader_widget.start()
            self.load_worker = LoadVectorStoreWorker(file_path)
            self.load_worker.start()
            self.check_load_thread()
            
    def check_load_thread(self):
        if self.load_worker.is_alive():
            QTimer.singleShot(100, self.check_load_thread)
        else:
            if self.load_worker.error:
                self.on_vector_store_load_error(self.load_worker.error)
            else:
                self.on_vector_store_loaded(self.load_worker.result)
            
    def on_vector_store_loaded(self, vectorstore):
        self.loader_widget.stop()
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "fetch_k": 10})
        self.straico_llm = StraicoLLM(api_key=self.api_key, model=self.model)
        self.recreate_qa_system()
        self.chat_widget.add_message("Vector store loaded successfully.", False)

    def on_vector_store_load_error(self, error):
        self.loader_widget.stop()
        self.chat_widget.add_message(f"Error: Failed to load vector store - {error}", False)

    def clear_session(self):
        if self.qa_system or self.retriever or self.chat_widget.layout.count() > 1:
            self.qa_system = None
            self.straico_llm = None
            self.retriever = None
            return json.dumps({"response": "Session cleared.", "status": 200})
        else:
            return json.dumps({"response": "No active session to clear.", "status": 200})

    def clear_session_gui(self):
        if self.qa_system or self.retriever or self.chat_widget.layout.count() > 1:
            self.qa_system = None
            self.straico_llm = None
            self.retriever = None
            self.chat_widget = ChatWidget()
            self.chat_scroll.setWidget(self.chat_widget)
            self.chat_input.clear()
            self.chat_widget.add_message("Session cleared.", False)
        else:
            QMessageBox.information(self, "No Active Session", "No active session to clear.")
            
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
                QMessageBox.warning(self, "No API Key", "Please enter your Straico API Key.")
        if self.straico_llm:
            self.straico_llm.update_credentials(self.api_key, self.model)

    def remove_api_key(self, dialog):
        self.api_key = ''
        self.save_settings()
        dialog.close()
        QMessageBox.information(self, "API Key Removed", "API Key has been removed.")
        
    def toggle_local_server(self):
        try:
            if self.local_server.is_running():
                self.local_server.stop()
                QMessageBox.information(self, "Server Status", "Local server stopped.")
            else:
                self.local_server.start()
                QMessageBox.information(self, "Local Server Status", "Local server started successfully.")
            self.server_action.setChecked(self.local_server.is_running())
            self.save_settings()
            self.update_server_status_indicator()
        except Exception as e:
            QMessageBox.warning(self, "Server Error", f"An error occurred: {str(e)}")
            
    def update_server_status_indicator(self):
        if self.local_server.is_running():
            self.server_status_indicator.setStyleSheet("background-color: #A580AD; border-radius: 6px;")
            self.server_status_indicator.setToolTip("Local server is running")
        else:
            self.server_status_indicator.setStyleSheet("background-color: #949494; border-radius: 6px;")
            self.server_status_indicator.setToolTip("Local server is not running")

    def process_chat(self, prompt):
        if not self.api_key:
            return json.dumps({"response": "Error: API Key is not set. Please set the API Key in the application settings.", "status": 401})

        if not self.model:
            return json.dumps({"response": "Error: Model is not selected. Please select a model.", "status": 400})

        if prompt.startswith("load_corpus:"):
            directory = prompt[12:].strip()
            return self.load_corpus_local_server(directory)
        elif prompt.startswith("load_vs:"):
            vector_store_path = prompt[8:].strip()
            return self.load_vector_store_local_server(vector_store_path)
        elif prompt.startswith("save_vs:"):
            vector_store_path = prompt[8:].strip()
            return self.save_vector_store_local_server(vector_store_path)
        elif prompt == "list_models":
            return self.list_models_local_server()
        elif prompt.startswith("select_model:"):
            model_name = prompt[13:].strip()
            return self.select_model_local_server(model_name)
        elif prompt == "clear_session":
            return self.clear_session_local_server()

        if not self.qa_system and not self.retriever:
            return json.dumps({"response": "Error: Please load a corpus or vector store to proceed.", "status": 400})

        if not self.qa_system:
            self.recreate_qa_system()

        try:
            response = self.qa_system.invoke({"query": prompt})
            return json.dumps({"response": response["result"], "status": 200})
        except Exception as e:
            return json.dumps({"response": f"Error: {str(e)}", "status": 400})

    def load_corpus_local_server(self, directory):
        if not self.api_key:
            return json.dumps({"response": "Error: API Key is not set. Please set the API Key in the application settings.", "status": 401})

        if not os.path.exists(directory):
            return json.dumps({"response": f"Error: Directory not found - {directory}", "status": 400})

        try:
            self.loader = DocumentLoader(directory, self.api_key, self.model)
            self.loader.start()
            self.loader.join()

            if self.loader.error:
                return json.dumps({"response": f"Error loading corpus: {self.loader.error}", "status": 400})

            self.qa_system, self.straico_llm, self.retriever, skipped_files = self.loader.result
            
            response = f"Corpus loaded successfully from {directory}"
            if skipped_files:
                response += "\n\n" + self.format_skipped_files_message(skipped_files)
            
            return json.dumps({"response": response, "status": 200})
        except Exception as e:
            return json.dumps({"response": f"Error loading corpus: {str(e)}", "status": 400})

    def save_vector_store_local_server(self, file_path):
        if not self.api_key:
            return json.dumps({"response": "Error: API Key is not set. Please set the API Key in the application settings.", "status": 401})

        if not self.retriever:
            return json.dumps({"response": "Error: No vector store to save. Please load a corpus or vector store first.", "status": 400})

        try:
            if not file_path.endswith('.wov'):
                file_path += '.wov'
            save_worker = SaveVectorStoreWorker(self.retriever, file_path)
            save_worker.start()
            save_worker.join()

            if save_worker.error:
                return json.dumps({"response": f"Error saving vector store: {save_worker.error}", "status": 400})

            return json.dumps({"response": f"Vector store saved to {file_path}", "status": 200})
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return json.dumps({"response": f"Error saving vector store: {str(e)}", "status": 400})

    def load_vector_store_local_server(self, file_path):
        if not self.api_key:
            return json.dumps({"response": "Error: API Key is not set. Please set the API Key in the application settings.", "status": 401})

        if not os.path.exists(file_path):
            return json.dumps({"response": f"Error: File not found - {file_path}", "status": 400})

        try:
            load_worker = LoadVectorStoreWorker(file_path)
            load_worker.start()
            load_worker.join()

            if load_worker.error:
                return json.dumps({"response": f"Error loading vector store: {load_worker.error}", "status": 400})

            vectorstore = load_worker.result
            
            if not isinstance(vectorstore, FAISS):
                raise ValueError(f"Loaded object is not a FAISS instance. Got {type(vectorstore)}")
            
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "fetch_k": 10})
            self.recreate_qa_system()
            return json.dumps({"response": f"Vector store loaded successfully from {file_path}", "status": 200})
        except Exception as e:
            error_msg = f"Error loading vector store: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"response": error_msg, "status": 400})
                        
    def list_models_local_server(self):
        if not self.api_key:
            return json.dumps({"response": "Error: API Key is not set. Please set the API Key in the application settings.", "status": 401})

        if not self.available_models:
            self.update_models()

        if not self.available_models:
            return json.dumps({"response": "Error: Unable to fetch the list of available models. Please ensure that the API Key is correct or check the internet connection.", "status": 400})

        model_list = [{"name": model[1], "model": model[0], "coins": model[2]} for model in self.available_models]
        return json.dumps({"data": model_list, "status": 200})

    def select_model_local_server(self, model_name):
        if not self.api_key:
            return json.dumps({"response": "Error: API Key is not set. Please set the API Key in the application settings.", "status": 401})

        if not self.available_models:
            self.update_models()

        if not any(model_name == model[0] for model in self.available_models):
            return json.dumps({"response": f"Error: '{model_name}' is not a valid model. Please choose from the list of valid models.", "status": 400})

        self.model = model_name
        self.save_settings()
        if self.straico_llm:
            self.straico_llm.update_credentials(self.api_key, self.model)
        if self.qa_system:
            self.recreate_qa_system()
        return json.dumps({"response": f"Model set to: {model_name}", "status": 200})
            
    def clear_session_local_server(self):
        if not self.qa_system and not self.retriever:
            return json.dumps({"response": "No active session to clear.", "status": 400})

        try:
            self.qa_system = None
            self.straico_llm = None
            self.retriever = None
            return json.dumps({"response": "Session cleared.", "status": 200})
        except Exception as e:
            return json.dumps({"response": f"Error clearing session: {str(e)}", "status": 400})
            
    def closeEvent(self, event):
        if self.local_server.is_running():
            self.local_server.stop()
        super().closeEvent(event)

    def fetch_models_from_api(self):
        url = "https://api.straico.com/v1/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            chat_models = data.get('data', {}).get('chat', [])
            return [(model['model'], model['name'], model['pricing']['coins']) for model in chat_models]
        except requests.RequestException as e:
            logger.error(f"Error fetching models from API: {e}")
            return None
            
    def update_models(self):
        if not self.api_key:
            logger.warning("API key not set. Unable to fetch models.")
            return

        new_models = self.fetch_models_from_api()
        if new_models:
            self.available_models = new_models
            logger.info("Models updated.")
        else:
            logger.warning("Failed to update models.")
            self.available_models = []

    def set_model(self):
        if not self.available_models:
            self.update_models()
        
        if not self.available_models:
            QMessageBox.warning(self, "Error", "Unable to fetch available models. Please ensure that the API Key is in place or check the internet connection.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Model")
        layout = QVBoxLayout(dialog)

        current_model = next((f"{friendly} • {coins} Coin(s)" for model_name, friendly, coins in self.available_models if model_name == self.model), "No model set")

        current_label = QLabel(f"Current Model: {current_model}")
        layout.addWidget(current_label)

        combo = QComboBox()
        combo.addItems([f"{friendly} • {coins} Coin(s)" for _, friendly, coins in self.available_models])
        current_index = next((i for i, (model_name, _, _) in enumerate(self.available_models) if model_name == self.model), 0)
        combo.setCurrentIndex(current_index)
        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.Accepted:
            new_friendly = combo.currentText()
            self.model = next(model_name for model_name, friendly, coins in self.available_models if f"{friendly} • {coins} Coin(s)" == new_friendly)
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
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt_template}
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
        <p>Set the Straico API Key from <strong>Settings → Set API Key</strong>. Existing Straico users can find their API Key from the platform&#39;s settings page. New users may choose to create a Straico account using this <a href='https://platform.straico.com/signup?fpr=jaisal'>referral link</a>.</p>
        <p>Select the preferred model to interact with from <strong>Settings → Select Model</strong>. This preference can be changed at any time, even while a session is in progress.</p>        
        <hr />
        <h3>Load Corpus</h3>
        <p>Load the collection of files to be used as source material for Retrieval-Augmented Generation from <strong>File → Load Corpus → Select Corpus Directory</strong>. WovenSnips currently supports files with the following extensions to be included in the corpus directory: <code>.pdf</code>, <code>.txt</code>, <code>.md</code>, <code>.csv</code></p>
        <hr />
        <h3>Save and Load Vector Store</h3>
        <p>WovenSnips allows users to save the loaded corpus as vector store for future reuse by serialising it in <code>.wov</code> format from <strong>File → Save Vector Store</strong>. Users can load it from <strong>File → Load Vector Store → your_file.wov</strong> for reuse. Loading frequently used corpora from vector stores would significantly save initial processing time and resources compared to loading it every time from scratch.</p>
        <p><em><strong>Note:</strong> Load vector stores only from trusted and verified sources to ensure security and integrity.</em></p>
        <hr />
        <h3>Clear Session</h3>
        <p>Start a fresh session of interaction by clearing the loaded corpus and vector files from <strong>File → Clear Session</strong>.</p>
        <p><em><strong>Note:</strong> The API Key, model and theme preferences, and local server status persist across sessions and remain unaffected on clearing sessions or application relaunch. Users can manually remove the API Key, change the model and theme preferences, and local server status from Settings.</em></p>
        <hr />
        <h3>Themes</h3>
        <p>WovenSnips supports light (default) and dark themes. Toggle the dark theme on/off from <strong>Settings → Dark Theme</strong>.</p>
        <hr />
        <h3>Local Server</h3>
        <p>The local server allows WovenSnips to interact with other applications or scripts via HTTP requests at <code>http://localhost:60606</code>.Toggle the local server on/off from <strong>Settings → Local Server</strong>. The status indicator shows purple when the server is running and grey when inactive.</p>
        <p><em><strong>Note:</strong> Please refer to the <a href='https://github.com/ekjaisal/WovenSnips/wiki'>Local Server Documentation</a> for details on using the local server and its endpoints, with examples.</em></p>
        """
        self.show_scrollable_html_dialog("Getting Started", tips_html)

    def show_about(self):
        about_html = """
        <h2>WovenSnips <small>v1.1.0</small></h2>
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
        <li><a href="https://github.com/numpy/numpy">NumPy</a> (BSD-3-Clause License)</li>
        <li><a href="https://pypi.org/project/langchain">Langchain</a> (MIT License)</li>
        <li><a href="https://pypi.org/project/langchain-community">LangChain Community</a> (MIT License)</li>
        <li><a href="https://pypi.org/project/langchain-huggingface">Langchain Hugging Face</a> (MIT License)</li>
        <li><a href="https://github.com/kyamagu/faiss-wheels">FAISS-CPU</a> (MIT License)</li>
        <li><a href="https://github.com/pydantic/pydantic">Pydantic</a> (MIT License)</li>
        <li><a href="https://github.com/jsvine/pdfplumber">pdfplumber</a> (MIT License)</li>
        <li><a href="https://github.com/msgpack/msgpack-python">MessagePack</a> (Apache License, Version 2.0)</li>
        <li><a href="https://github.com/psf/requests">Requests</a> (Apache License, Version 2.0)</li>
        <li><a href="https://fonts.google.com/specimen/Roboto">Roboto Font</a> (Apache License, Version 2.0)</li>
        <li><a href="https://pyinstaller.org">PyInstaller</a> (GPL 2.0 License, with an Exception)</li>
        <li><a href="https://jrsoftware.org">Inno Setup</a> (Inno Setup License)</li>
        </ul>
        <p>Many thanks to the developers and contributors of these libraries and services. Full documentation and license details can be found at the links provided.</p>
        <hr>
        <h3>Acknowledgements</h3>
        <p>WovenSnips has benefitted significantly from the assistance of Anthropic's <a href="https://www.anthropic.com/news/claude-3-5-sonnet">Claude 3.5 Sonnet</a> with all the heavy lifting associated with coding, <a href="https://github.com/RoboRiley">Riley</a>'s addition of local server capability, and the overwhelming warmth and support from the Straico community.</p>
        """
        self.show_scrollable_html_dialog("About WovenSnips", about_html)

    def send_message(self):
        query = self.chat_input.text().strip()
        if not query:
            return
        if not self.qa_system and not self.retriever:
            QMessageBox.warning(self, "Error", "Please load a corpus or vector store to proceed.")
            return
        if not self.api_key:
            self.prompt_for_api_key()
            if not self.api_key:
                return
        if not self.qa_system:
            self.recreate_qa_system()

        self.chat_widget.add_message(query, True)
        self.chat_input.clear()
        self.worker = ChatWorker(self.qa_system, query)
        self.worker.start()
        self.check_chat_thread()
        
    def check_chat_thread(self):
        if self.worker.is_alive():
            QTimer.singleShot(100, self.check_chat_thread)
        else:
            if self.worker.error:
                self.display_error(self.worker.error)
            else:
                self.display_response(self.worker.result)
        
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
            if settings.get('local_server', False):
                self.local_server.start()
                self.update_server_status_indicator()
            if hasattr(self, 'server_action'):
                self.server_action.setChecked(self.local_server.is_running())
            self.update_server_status_indicator()
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

        layout.addWidget(QLabel("Please get the initial set up ready to proceed."))
        layout.addWidget(QLabel("Enter your Straico API Key:"))
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
            api_key = api_key_input.text().strip()
            if not api_key:
                QMessageBox.warning(self, "Error", "Please enter an API Key.")
                return False
            self.api_key = api_key
            self.save_settings()
            self.update_models()
            if self.available_models:
                self.model = self.available_models[0][0]
                self.save_settings()
                return True
            else:
                QMessageBox.warning(self, "Error", "Unable to fetch available models. Please ensure that the API Key is in place or check the internet connection.")
                return False
        return False

    def save_settings(self):
        settings = {
            'api_key': self.api_key,
            'model': self.model,
            'dark_theme': self.current_theme == DARK_THEME,
            'local_server': self.local_server.is_running()
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
