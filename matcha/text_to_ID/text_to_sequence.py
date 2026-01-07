from matcha.text_to_ID.symbols import symbols
from phonemizer import phonemize
from phonemizer.separator import Separator

# Dictionnaire de conversion : Phonème -> Chiffre
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def text_to_sequence(text, cleaner_names):
    """
    Convertit un texte brut en une séquence d'IDs de phonèmes via phonemizer.
    """
    # 1. Phonémisation (Grapheme -> Phoneme)
    # On utilise le backend espeak par défaut pour l'anglais américain
    phones = phonemize(
        text,
        language='en-us',
        backend='espeak',
        separator=Separator(phone=None, word=' ', syllable=''),
        strip=True,
        preserve_punctuation=True,
        with_stress=True
    )
    
    # 2. Conversion en IDs (Phoneme -> ID)
    sequence = []
    
    for symbol in phones:
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
        else:
            # Optionnel : Print pour débugger les phonèmes inconnus
            # print(f"Symbole inconnu ignoré : {symbol}")
            pass
            
    return sequence
