import json
import re

# Carregar o ficheiro
with open('Uma tragedia americana - Theodore Dreiser.analysis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Mapeamento de correções (ID antigo -> ID novo)
correcoes = {
    'clyde': 'clyde_griffiths',
    'roberta_griffiths': 'roberta',
    'griffiths': 'elvira_griffiths',
    # Se quiseres unificar outros, adiciona aqui
}

# Corrigir caracteres (remover duplicados)
chars = data.get('characters', {})
novos_chars = {}
for cid, cdata in chars.items():
    novo_id = correcoes.get(cid, cid)
    if novo_id in novos_chars:
        # Se já existe, mesclar descrições (opcional)
        continue
    novos_chars[novo_id] = cdata
data['characters'] = novos_chars

# Corrigir segmentos
segments = data.get('segments', [])
for seg in segments:
    cid = seg.get('character_id')
    if cid in correcoes:
        seg['character_id'] = correcoes[cid]

# Corrigir aliases (se existirem)
aliases = data.get('aliases', {})
novos_aliases = {}
for cid, terms in aliases.items():
    novo_id = correcoes.get(cid, cid)
    if novo_id in novos_aliases:
        # Mesclar listas de termos
        novos_aliases[novo_id].extend(terms)
    else:
        novos_aliases[novo_id] = terms
data['aliases'] = novos_aliases

# Guardar ficheiro corrigido
with open('Uma tragedia americana - Theodore Dreiser.analysis_corrigido.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Correções aplicadas. Ficheiro guardado como 'analysis_corrigido.json'.")