# core/extractor.py
# ─── Extracção de texto de epub e txt ────────────────────────────────────────
# v2.2 — Correcções:
#   • inline tags (em, strong, span, a...) não quebram parágrafos
#   • parágrafos com texto partido a meio resolvidos
#   • leitura por spine OPF para ordem correcta dos capítulos

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_epub(file_path: str) -> str:
    """
    Extrai texto do epub preservando estrutura de parágrafos.
    Cada <p> ou bloco equivalente vira um parágrafo separado por \\n\\n.
    Tags inline (em, strong, span, a...) não quebram o parágrafo.
    """
    import zipfile
    from html.parser import HTMLParser

    class EpubParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paragraphs = []
            self.current    = []
            self.in_para    = False

            # Tags que delimitam parágrafos (bloco)
            self.para_tags = {
                'p', 'blockquote', 'li',
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            }

            # Tags inline — NÃO quebram o parágrafo, só continuam o texto
            self.inline_tags = {
                'em', 'strong', 'i', 'b', 'u', 'span', 'a',
                'cite', 'abbr', 'acronym', 'sup', 'sub',
                'small', 'big', 'mark', 'q', 'dfn', 'code',
                'kbd', 'var', 'samp', 'time', 'bdi', 'bdo',
            }

            # Tags a ignorar completamente (conteúdo e filhos)
            self.skip_tags  = {
                'style', 'script', 'head', 'meta',
                'link', 'nav', 'aside', 'figure', 'figcaption',
            }
            self.skip_depth = 0

        def handle_starttag(self, tag, attrs):
            tag = tag.lower()
            if tag in self.skip_tags:
                self.skip_depth += 1
            elif self.skip_depth > 0:
                pass  # dentro de tag ignorada
            elif tag in self.inline_tags:
                pass  # inline: não faz nada, texto continua no parágrafo actual
            elif tag == 'br':
                # Quebra de linha dentro de parágrafo → espaço
                if self.in_para:
                    self.current.append(' ')
            elif tag in self.para_tags:
                # Fechar parágrafo anterior se ainda estava aberto
                if self.in_para and self.current:
                    self._close_paragraph()
                self.in_para = True
                self.current = []
            elif tag == 'div':
                # div pode ser bloco ou wrapper — fechar parágrafo anterior
                if self.in_para and self.current:
                    self._close_paragraph()

        def handle_endtag(self, tag):
            tag = tag.lower()
            if tag in self.skip_tags:
                self.skip_depth = max(0, self.skip_depth - 1)
            elif self.skip_depth > 0:
                pass
            elif tag in self.inline_tags:
                pass  # inline: não faz nada
            elif tag in self.para_tags:
                if self.in_para:
                    self._close_paragraph()
            elif tag == 'div':
                if self.in_para and self.current:
                    self._close_paragraph()

        def handle_data(self, data):
            if self.skip_depth == 0 and self.in_para:
                self.current.append(data)

        def handle_entityref(self, name):
            # Entidades HTML comuns
            entities = {
                'amp': '&', 'lt': '<', 'gt': '>',
                'quot': '"', 'apos': "'",
                'nbsp': ' ', 'mdash': '—', 'ndash': '–',
                'lsquo': '\u2018', 'rsquo': '\u2019',
                'ldquo': '\u201c', 'rdquo': '\u201d',
                'laquo': '«', 'raquo': '»',
                'hellip': '…', 'bull': '•',
            }
            if self.skip_depth == 0 and self.in_para:
                self.current.append(entities.get(name, ''))

        def handle_charref(self, name):
            if self.skip_depth == 0 and self.in_para:
                try:
                    if name.startswith('x'):
                        char = chr(int(name[1:], 16))
                    else:
                        char = chr(int(name))
                    self.current.append(char)
                except (ValueError, OverflowError):
                    pass

        def _close_paragraph(self):
            text = ''.join(self.current).strip()
            # Limpar tabs e espaços múltiplos dentro da frase
            text = re.sub(r'[\t\r]+', ' ', text)
            text = re.sub(r'  +', ' ', text)
            if text and len(text) > 2:
                self.paragraphs.append(text)
            self.in_para = False
            self.current = []

        def get_text(self):
            # Fechar parágrafo pendente
            if self.in_para and self.current:
                self._close_paragraph()
            return '\n\n'.join(self.paragraphs)

    try:
        with zipfile.ZipFile(file_path) as z:
            all_names = z.namelist()

            # ── Ler ordem correcta via OPF/spine ─────────────────────────────
            opf_names   = [n for n in all_names if n.endswith('.opf')]
            spine_order = []

            if opf_names:
                opf_content = z.read(opf_names[0]).decode('utf-8', errors='ignore')

                # Extrair IDs do spine (ordem de leitura)
                spine_ids  = re.findall(r'<itemref\s+idref=["\']([^"\']+)["\']',
                                        opf_content)

                # Mapear ID → href
                id_to_href = dict(re.findall(
                    r'<item\s[^>]*\bid=["\']([^"\']+)["\'][^>]*\bhref=["\']([^"\']+)["\']',
                    opf_content
                ))
                # Também tentar href antes de id
                id_to_href.update(dict(
                    (b, a) for a, b in re.findall(
                        r'<item\s[^>]*\bhref=["\']([^"\']+)["\'][^>]*\bid=["\']([^"\']+)["\']',
                        opf_content
                    )
                ))

                base_dir = str(Path(opf_names[0]).parent)
                for sid in spine_ids:
                    href = id_to_href.get(sid, '')
                    if href:
                        if base_dir and base_dir != '.':
                            full = f"{base_dir}/{href}"
                        else:
                            full = href
                        # Normalizar separadores
                        full = re.sub(r'/{2,}', '/', full)
                        spine_order.append(full)

            # ── Fallback: ordenar por nome ────────────────────────────────────
            if not spine_order:
                spine_order = sorted([
                    n for n in all_names
                    if (n.endswith('.xhtml') or n.endswith('.html'))
                    and not any(skip in n.lower() for skip in [
                        'toc', 'nav', 'cover', 'copyright',
                        'title', 'colophon', 'dedication', 'halftitle',
                    ])
                ])

            # ── Extrair texto de cada ficheiro ────────────────────────────────
            all_parts = []
            for name in spine_order:
                # Resolver nome se não existir exactamente
                if name not in all_names:
                    candidates = [n for n in all_names
                                  if n.endswith(Path(name).name)]
                    if candidates:
                        name = candidates[0]
                    else:
                        continue

                try:
                    content = z.read(name).decode('utf-8', errors='ignore')
                    parser  = EpubParser()
                    parser.feed(content)
                    part    = parser.get_text()
                    if part.strip():
                        all_parts.append(part)
                except Exception as e:
                    logger.warning(f"Erro ao ler {name}: {e}")

            full_text = '\n\n'.join(all_parts)

            # ── Limpeza final ─────────────────────────────────────────────────
            paragraphs = full_text.split('\n\n')
            cleaned    = []
            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue

                # Ignorar títulos de capítulo/parte
                if re.match(
                    r'^(Capítulo|Livro|Parte|Chapter|Book|Part|'
                    r'I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|'
                    r'XIV|XV|XVI|XVII|XVIII|XIX|XX)'
                    r'\s*[\dIVXLCM]*\.?\s*$',
                    p, re.IGNORECASE
                ):
                    continue

                # Ignorar linhas só com número
                if re.match(r'^\d+\.?\s*$', p):
                    continue

                # Ignorar metadata Standard Ebooks / Gutenberg
                if any(x in p for x in [
                    'standardebooks.org', 'Standard Ebooks',
                    'Creative Commons', 'Projeto Gutenberg',
                    'domínio público', 'direitos de autor',
                    'epub:type', 'publicationtype',
                    'transcriber', 'proofreader',
                ]):
                    continue

                cleaned.append(p)

            result = '\n\n'.join(cleaned)
            logger.info(
                f"epub extraído: {len(cleaned)} parágrafos, "
                f"{len(result)} chars"
            )
            return result

    except Exception as e:
        logger.error(f"Erro ao ler epub {file_path}: {e}")
        raise


def _extract_txt(file_path: str) -> str:
    """
    Lê ficheiro de texto simples com detecção automática de encoding.
    Normaliza quebras de linha e garante parágrafos com \\n\\n.
    """
    for encoding in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
        try:
            text = Path(file_path).read_text(encoding=encoding)
            # Normalizar quebras de linha
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            # Limpar tabs
            text = re.sub(r'\t', ' ', text)
            # Limpar espaços múltiplos
            text = re.sub(r' {2,}', ' ', text)
            # Normalizar parágrafos: 3+ newlines → 2
            text = re.sub(r'\n{3,}', '\n\n', text)
            logger.info(
                f"txt extraído: {len(text.split(chr(10)*2))} parágrafos, "
                f"{len(text)} chars"
            )
            return text
        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Não foi possível ler {file_path} "
        f"com nenhum encoding conhecido (utf-8, latin-1, cp1252)."
    )


def extract_text(file_path: str) -> str:
    """
    Ponto de entrada principal.
    Extrai texto de epub ou txt.
    Retorna texto com parágrafos separados por \\n\\n.
    """
    path   = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == '.epub':
        return _extract_epub(file_path)
    elif suffix == '.txt':
        return _extract_txt(file_path)
    elif suffix == '.pdf':
        raise ValueError(
            "PDF não suportado directamente. "
            "Converte para epub ou txt e tenta novamente."
        )
    else:
        raise ValueError(f"Formato não suportado: {suffix}")