# core/extractor.py
# ─── Extração de texto de TXT, PDF e EPUB ─────────────────────────────────────

import re
import logging
from pathlib import Path

import fitz
from ebooklib import epub
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def extract_text(filepath: str) -> str:
    """Extrai texto de ficheiros .txt, .pdf ou .epub."""
    ext = Path(filepath).suffix.lower()
    content = ""
    try:
        if ext == ".txt":
            content = _read_txt(filepath)
        elif ext == ".pdf":
            content = _read_pdf(filepath)
        elif ext == ".epub":
            content = _read_epub(filepath)
        # Corrigir hifenização no fim de linha
        content = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', content)
    except Exception as e:
        logger.error(f"Erro ao ler ficheiro: {e}")
    return content


def _read_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _read_pdf(filepath: str) -> str:
    doc = fitz.open(filepath)
    content = ""
    for page in doc:
        content += page.get_text() + "\n\n"
    doc.close()
    return content


def _read_epub(filepath: str) -> str:
    book = epub.read_epub(filepath)
    content = ""
    for item in book.get_items():
        if item.get_type() == 9:  # ITEM_DOCUMENT
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            content += soup.get_text(separator=' ') + "\n\n"
    return content
