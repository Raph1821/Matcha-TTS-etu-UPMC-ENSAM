"""
Définit l'ensemble des symboles (vocabulaire) du modèle.
"""
# On importe les phonèmes valides directement depuis votre fichier local
from matcha.text_to_ID.cmudict import valid_symbols

_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# La liste finale : PAD + Ponctuation + Lettres + Phonèmes CMU
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + valid_symbols
