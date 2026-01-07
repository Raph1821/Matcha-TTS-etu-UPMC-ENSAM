from matcha.text_to_ID.symbols import symbols

# Dictionnaire de conversion : Caractère -> Chiffre
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def text_to_sequence(text, cleaner_names):
    """
    Version simplifiée pour éviter les erreurs d'installation espeak/phonemizer.
    
    Args:
        text (str): Le texte à dire.
        cleaner_names (list): Liste des noms de cleaners (ignorée ici pour simplifier).
    """
    sequence = []
    
    # Nettoyage minimal (minuscules)
    clean_text = text.lower()
    
    # Conversion en IDs
    for char in clean_text:
        if char in _symbol_to_id:
            sequence.append(_symbol_to_id[char])
        else:
            # On ignore les caractères inconnus (ex: é, à, emojis) pour éviter le crash
            # Dans un vrai système, on translittérerait (é -> e)
            print(f"Attention: caractère ignoré '{char}'")
            pass
            
    return sequence
