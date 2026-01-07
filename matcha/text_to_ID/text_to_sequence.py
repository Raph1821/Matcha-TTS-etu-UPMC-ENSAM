# Imports normaux
from matcha.text_to_ID.symbols import symbols
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper # <--- Assurez-vous d'avoir cet import
import os

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def text_to_sequence(text, cleaner_names):
    """
    Convertit un texte brut en séquence d'IDs.
    """
    # --- PATCH SPECIAL WORKERS ---
    # On force le chemin ICI, car ce code s'exécute à l'intérieur du worker
    lib_path = "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"
    if os.path.exists(lib_path):
        try:
            EspeakWrapper.set_library(lib_path)
        except Exception:
            pass # Déjà configuré, on ignore
    # -----------------------------

   
    phones = phonemize(
        text,
        language='en-us',
        backend='espeak',
        # ...
    )
    
    # ... conversion en IDs ...
    sequence = []
    for symbol in phones:
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
            
    return sequence
