# core/text_splitter.py
import re

def split_by_chapters(text: str) -> list[dict]:
    """Divide o texto em capítulos, preservando o título."""
    # Regex para detetar "Capítulo 1", "CAPÍTULO I", "Parte Um", etc.
    chapter_pattern = re.compile(
        r'(?i)^\s*(cap[ií]tulo\s+[ivx\d]+|parte\s+[ivx\d]+|livro\s+[ivx\d]+)\s*[:\-]?\s*(.*?)$', 
        re.MULTILINE
    )
    
    matches = list(chapter_pattern.finditer(text))
    chapters = []
    
    if not matches:
        return [{"title": "Início", "text": text}]
        
    # Texto antes do primeiro capítulo
    if matches[0].start() > 0:
        chapters.append({"title": "Prólogo", "text": text[:matches[0].start()]})
        
    for i, match in enumerate(matches):
        title = f"{match.group(1)} {match.group(2)}".strip()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        chapters.append({"title": title, "text": text[start:end].strip()})
        
    return chapters