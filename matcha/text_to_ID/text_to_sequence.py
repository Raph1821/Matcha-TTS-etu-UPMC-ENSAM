import os
import re
from matcha.text_to_ID.symbols import symbols
from matcha.text_to_ID import cleaners 
from matcha.text_to_ID.cmudict import CMUDict

# --- 1. CONFIGURATION DU DICTIONNAIRE ---
# Chemin relatif vers le fichier de données (au même endroit que ce script)
current_dir = os.path.dirname(os.path.abspath(__file__))
CMU_DICT_PATH = os.path.join(current_dir, 'cmudict-0.7b')

# Singleton pour ne charger le gros fichier qu'une seule fois
_cmu_dict = None

def get_cmu_dict():
    global _cmu_dict
    if _cmu_dict is None:
        if not os.path.exists(CMU_DICT_PATH):
            raise FileNotFoundError(f"Fichier introuvable : {CMU_DICT_PATH}")
        print(f"Chargement de CMUDict...")
        # On utilise la classe définie dans votre fichier cmudict.py
        _cmu_dict = CMUDict(CMU_DICT_PATH, keep_ambiguous=False)
    return _cmu_dict

# --- 2. CONFIGURATION DES MAPPINGS ---
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regex pour séparer les mots de la ponctuation (garde la ponctuation)
# Sépare sur : espace, ou un des caractères de ponctuation
_split_regex = re.compile(r"([{}])".format(re.escape('!\'(),.:;? ')))

def text_to_sequence(text, cleaner_names):
    """
    Pipeline complet : Nettoyage -> Phonémisation -> Conversion en IDs
    """
    sequence = []
    
    # A. NETTOYAGE DU TEXTE (via cleaners.py)
    # cleaner_names est généralement ["english_cleaners"] venant de train.py
    for name in cleaner_names:
        if hasattr(cleaners, name):
            cleaner = getattr(cleaners, name)
            # Cette fonction va appeler numbers.py en interne si codée standardement
            text = cleaner(text)
        else:
            print(f"⚠️ Warning: Cleaner '{name}' non trouvé dans cleaners.py")

    # B. PHONÉMISATION (via cmudict.py)
    cmu = get_cmu_dict()
    
    # On sépare le texte nettoyé en mots et ponctuation
    parts = _split_regex.split(text)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # 1. Si c'est de la ponctuation connue
        if part in _symbol_to_id:
            sequence.append(_symbol_to_id[part])
            continue
            
        # 2. Si c'est un mot -> Recherche dans CMU
        pronunciation = cmu.lookup(part)
        
        if pronunciation:
            # CMU retourne une liste de possibilités (ex: ["HH AH0 L OW1", ...])
            # On prend la première
            phones_str = pronunciation[0]
            phones = phones_str.split()
            
            for phone in phones:
                if phone in _symbol_to_id:
                    sequence.append(_symbol_to_id[phone])
                else:
                    # Sécurité si un phonème du dict n'est pas dans symbols.py
                    print(f"⚠️ Phonème inconnu : {phone}")
        
        else:
            # 3. Si mot inconnu (OOV) -> On utilise les lettres
            # Le modèle apprendra à lire les lettres pour les mots rares
            for char in part:
                if char in _symbol_to_id:
                    sequence.append(_symbol_to_id[char])
        
        # Ajout d'un espace (silence) après chaque mot si le modèle le supporte
        # (Vérifier si ' ' est dans vos symbols)
        if ' ' in _symbol_to_id:
            sequence.append(_symbol_to_id[' '])

    return sequence

def sequence_to_text(sequence):
    result = ''
    for id in sequence:
        if id in _id_to_symbol:
            result += _id_to_symbol[id]
    return result
