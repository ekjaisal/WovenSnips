# WovenSnips
[![GitHub Release](https://img.shields.io/github/v/release/ekjaisal/WovenSnips?color=141414&label=Release)](https://github.com/ekjaisal/WovenSnips/releases) [![GitHub Downloads](https://img.shields.io/github/downloads/ekjaisal/WovenSnips/total?color=141414&label=Downloads)](https://github.com/ekjaisal/WovenSnips/releases) [![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-141414.svg)](https://github.com/ekjaisal/WovenSnips/blob/main/LICENSE) [![Citation File](https://img.shields.io/badge/Citation-CFF-141414.svg)](https://github.com/ekjaisal/WovenSnips/blob/main/CITATION.cff) [![GitHub stars](https://img.shields.io/github/stars/ekjaisal/WovenSnips?color=141414)](https://github.com/ekjaisal/WovenSnips/stargazers)

WovenSnips is a lightweight, free, and open-source implementation of Retrieval-Augmented Generation (RAG) using the [Straico API](https://straico.com/). It provides a simple and clean Graphical User Interface (GUI) for users to load corpora to perform RAG-based explorations of the corpus, mediating the interactions through various language models.

## Features 🌟

- 🔍 Load and process corpora for exploration and context retrieval using Retrieval-Augmented Generation (RAG).
- 🤖 Choose from a wide selection of LLMs provided by Straico.
- 🪶 Designed lightweight to run on devices without a dedicated GPU.
- 💾 Save and load vector stores for efficient reuse of processed corpora.
- 📚 Support for various file formats (.pdf, .txt, .md, .csv).
- 💬 Minimal, user-friendly interface for clutter-free, focused engagement with the loaded corpus.
- 🔌Local server option to programmatically interact with other applications and scripts.
- 🌓 Dark and light theme options.

<div style="display: flex; justify-content: space-between;">
  <img src="assets/screenshots/interface_light.jpg" alt="WovenSnips Main Interface - Light Theme" style="width: 49%;" />
  <img src="assets/screenshots/interface_dark.jpg" alt="WovenSnips Main Interface - Dark Theme" style="width: 49%;" />
</div><br>

## Usage 💻

1. Download the latest release from the [Releases](https://github.com/ekjaisal/WovenSnips/releases) page.
2. Set up WovenSnips on the local device using the installer (currently available only for Windows).
3. Point and click to launch the application.
4. Set the Straico API Key from **Settings → Set API Key**. Existing Straico users can find their API Key from the platform's settings page. New users may choose to create a Straico account using this [referral link](https://platform.straico.com/signup?fpr=jaisal).
5. Select the preferred model to interact with from **Settings → Select Model**.
6. Load the collection of files to be used as source material for RAG from **File → Load Corpus → Select Corpus Directory**, or load a previously saved vector store from **File → Load Vector Store**.
7. Start interacting with the corpus through the main interface.
8. Save the loaded corpus as a vector store for future reuse from **File → Save Vector Store** to avoid reprocessing large corpora.
9. Enable the local server from **Settings → Local Server** to allow [programmatic interaction](https://github.com/ekjaisal/WovenSnips/wiki).

## Third-Party Libraries and Services 🛠️

WovenSnips is built using [Python 3.11.9](https://www.python.org/downloads/release/python-3119) and relies on several modules from the Python Standard Library and the following third-party libraries and services:

- [Straico API](https://straico.com) for language model integration,
- [PySide6](https://pypi.org/project/PySide6) for the graphical user interface,
- [PyTorch](https://github.com/pytorch/pytorch), [Langchain](https://pypi.org/project/langchain), [Langchain Community](https://pypi.org/project/langchain-community), [Langchain Hugging Face](https://pypi.org/project/langchain-huggingface), and [FAISS-CPU](https://github.com/kyamagu/faiss-wheels) for RAG implementation,
- [Pydantic](https://github.com/pydantic/pydantic) for data validation,
- [pdfplumber](https://github.com/jsvine/pdfplumber) for processing PDFs,
- [Requests](https://github.com/psf/requests) for API communication,
- [Roboto Font](https://fonts.google.com/specimen/Roboto) for UI typography,
- [PyInstaller](https://pyinstaller.org/) for creating standalone executables, and
- [Inno Setup](https://jrsoftware.org/) for compiling and packaging the installer.

## License 📄

This project is licensed under the BSD 3-Clause License. Please see the [LICENSE](LICENSE) file for details.

## Disclaimer 📣

This tool is provided as-is, without any warranties. Users are responsible for ensuring that their use of this implementation complies with [Straico's terms and conditions](https://straico.com/terms-and-conditions).

## Acknowledgements 🤝🏾

WovenSnips has benefitted significantly from the assistance of Anthropic's [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) with all the heavy lifting associated with coding, [Riley](https://github.com/RoboRiley)'s addition of local server capability, and the overwhelming warmth and support from the Straico community. 

<a href="https://www.buymeacoffee.com/ekjaisal" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 40px !important;width: 160px !important;" ></a>