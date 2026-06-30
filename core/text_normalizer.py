# core/text_normalizer.py
"""
Normalização de texto baseada na lógica Alexandria, adaptada para PT-PT.
"""
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. NORMALIZAÇÃO DE TEXTO (COMPLETA)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """
    Normaliza texto para processamento consistente.
    Adaptado do Alexandria para PT-PT.
    """
    if not text:
        return ""
    
    # 1. Remover caracteres de controlo
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # 2. Normalizar espaços múltiplos
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Normalizar pontuação (espaços antes/depois)
    # Remove espaços ANTES de pontuação
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # Adiciona espaço DEPOIS de pontuação (se não houver)
    text = re.sub(r'([.,;:!?])([^\s\n])', r'\1 \2', text)
    
    # 4. Normalizar aspas tipográficas para retas
    # BUG CORRIGIDO: as chamadas anteriores tinham o caractere de origem e de
    # destino idênticos (ex: '"'.replace('"', '"')) — eram no-ops que não
    # faziam nada. Agora convertem efetivamente aspas/apóstrofos curvos
    # ("“”‘’") para as suas versões retas.
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    
    # 5. Normalizar hífenes e travessões
    # Converte travessão longo (—) para hífene curto (-)
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    
    # 6. Normalizar reticências
    text = re.sub(r'\.{2,}', '...', text)
    
    # 7. Remover espaços no início/fim de linhas
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    # 8. Remover linhas vazias múltiplas
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

# ═══════════════════════════════════════════════════════════════════════════
# 2. PALAVRAS COMUNS PARA FILTRAR (PT-PT)
# ═══════════════════════════════════════════════════════════════════════════

COMMON_WORDS_PT = {
    # Artigos
    'O', 'A', 'Os', 'As', 'Um', 'Uma', 'Uns', 'Umas',
    # Preposições
    'De', 'Da', 'Do', 'Das', 'Dos', 'Em', 'No', 'Na', 'Nos', 'Nas',
    'Por', 'Pelo', 'Pela', 'Pelos', 'Pelas', 'Para', 'Com', 'Sem',
    'Sobre', 'Entre', 'Após', 'Antes', 'Durante', 'Conforme', 'Segundo',
    # Conjunções
    'E', 'Mas', 'Porém', 'Todavia', 'Contudo', 'Entretanto', 'Ou',
    'Se', 'Que', 'Como', 'Quando', 'Onde', 'Porque', 'Pois',
    # Pronomes
    'Ele', 'Ela', 'Eles', 'Elas', 'Eu', 'Tu', 'Nós', 'Vós', 'Eles',
    'Me', 'Te', 'Se', 'Nos', 'Vos', 'Meu', 'Minha', 'Teu', 'Tua',
    # Verbos comuns
    'Ser', 'Estar', 'Ter', 'Haver', 'Fazer', 'Ir', 'Vir', 'Ver',
    'Dizer', 'Falar', 'Poder', 'Querer', 'Saber', 'Dever',
    # Palavras de estrutura
    'Capítulo', 'Parte', 'Livro', 'Secção', 'Parágrafo',
    'Nota', 'Prefácio', 'Introdução', 'Epílogo', 'Apêndice',
    # Títulos
    'Senhor', 'Senhora', 'Doutor', 'Doutora', 'Professor', 'Professora',
    'Sr', 'Sra', 'Dr', 'Dra', 'Prof'
}

def filter_common_words(names: List[str]) -> List[str]:
    """Filtra palavras comuns que não são personagens."""
    return [name for name in names if name not in COMMON_WORDS_PT]

# ═══════════════════════════════════════════════════════════════════════════
# 3. EXTRAÇÃO DE NOMES PRÓPRIOS (PT-PT)
# ═══════════════════════════════════════════════════════════════════════════

def extract_proper_names(text: str) -> List[str]:
    """
    Extrai nomes próprios do texto.
    Adaptado para PT-PT (considera acentos).
    """
    # Padrão para nomes próprios (palavras começando com maiúscula)
    # Inclui caracteres acentuados comuns em PT-PT
    pattern = r'\b[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][a-záâãàéêíóôõúç]+\b'
    
    names = re.findall(pattern, text)
    
    # Filtrar palavras comuns
    names = filter_common_words(names)
    
    return names

# ═══════════════════════════════════════════════════════════════════════════
# 4. NORMALIZAÇÃO DE NOMES DE PERSONAGENS (PT-PT)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_character_name_pt(name: str) -> str:
    """
    Normaliza um nome de personagem para PT-PT.
    Remove títulos, apelidos, e variações comuns.
    """
    if not name:
        return ""
    
    # Converter para minúsculas
    name = name.lower().strip()
    
    # Remover títulos comuns em PT-PT
    titles = [
        'sr.', 'sra.', 'dr.', 'dra.', 'prof.', 'professora',
        'senhor', 'senhora', 'doutor', 'doutora',
        'pai', 'mãe', 'tio', 'tia', 'avô', 'avó',
        'menino', 'menina', 'moço', 'moça'
    ]
    
    for title in titles:
        name = re.sub(rf'\b{re.escape(title)}\b', '', name, flags=re.IGNORECASE)
    
    # Remover caracteres especiais (manter acentos)
    name = re.sub(r'[^\w\sáâãàéêíóôõúç]', '', name)
    
    # Remover espaços extras
    name = ' '.join(name.split())
    
    return name

# ═══════════════════════════════════════════════════════════════════════════
# 5. LIMPEZA DE SEGMENTOS COM NORMALIZAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

def clean_and_normalize_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Limpa e normaliza todos os segmentos de texto.
    Combina deteção de texto corrompido com normalização completa.
    """
    cleaned = []
    removed_count = 0
    normalized_count = 0
    
    for seg in segments:
        text = seg.get("text", "").strip()
        
        # 1. Verificar se é texto corrompido
        if is_corrupted_text(text):
            removed_count += 1
            continue
        
        # 2. Normalizar o texto
        original_text = text
        text = normalize_text(text)
        
        if text != original_text:
            normalized_count += 1
        
        # 3. Verificar se o texto normalizado ainda é válido
        if len(text) < 3:
            removed_count += 1
            continue
        
        # 4. Atualizar o segmento
        seg["text"] = text
        cleaned.append(seg)
    
    if removed_count > 0:
        ratio = removed_count / len(segments) if segments else 0
        if ratio > 0.2:
            logger.warning(
                f"⚠️ clean_and_normalize_segments: {removed_count}/{len(segments)} "
                f"segmentos removidos por texto corrompido ({ratio:.0%}). "
                f"Se esta percentagem parecer alta, verificar is_corrupted_text()."
            )
        else:
            logger.info(f"🧹 Removidos {removed_count} segmentos com texto corrompido")

    if normalized_count > 0:
        logger.info(f"✨ Normalizados {normalized_count} segmentos")

    return cleaned

def is_corrupted_text(text: str) -> bool:
    """
    Deteta texto corrompido (fragmentos de JSON, texto não-latino, etc.).

    BUG CRÍTICO CORRIGIDO: a lista anterior `json_keywords` continha ":" e ","
    e procurava-os como SUBSTRING em qualquer parte do texto. Como praticamente
    toda a frase em português com mais de 5 palavras contém vírgulas ou dois
    pontos, isto marcava como "corrompida" a esmagadora maioria da prosa real
    do livro — só sobreviviam frases muito curtas sem pontuação interna.
    Esta é a mesma classe de bug já corrigida em ollama_analyzer.sanitize_segments;
    a verificação agora é estrutural (padrão de JSON "chave": valor, ou
    densidade alta de símbolos de estrutura), não substring solta.
    """
    if not text or len(text.strip()) < 3:
        return True

    # Contém caracteres não-latinos (chinês, japonês, árabe, etc.)
    if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u0600-\u06ff]', text):
        return True

    # É apenas pontuação ou espaços
    if re.match(r'^[\s\.\,\;\:\!\?\-]+$', text):
        return True

    # Fragmento de JSON que vazou para o texto (padrão real "chave": valor)
    json_leak_pattern = re.compile(
        r'"(?:character_id|pause_ms|emotion|pace|text|segments|characters)"\s*:'
    )
    if json_leak_pattern.search(text):
        return True

    # Densidade alta de símbolos de estrutura JSON ({}[]":) — não basta
    # conter um destes símbolos, tem de ser maioritariamente estrutura.
    symbol_chars = sum(1 for c in text if c in '{}[]":')
    if len(text) > 0 and symbol_chars / len(text) > 0.3:
        return True

    # Contém tags HTML/XML
    if re.search(r'<[^>]+>', text):
        return True

    # Contém URLs
    if re.search(r'https?://|www\.', text):
        return True

    return False