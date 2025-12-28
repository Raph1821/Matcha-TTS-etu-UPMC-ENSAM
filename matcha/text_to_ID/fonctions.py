import unittest
from matcha.text_to_ID.cleaners import english_cleaners
from matcha.text_to_ID.cmudict import CMUDict
from matcha.text_to_ID.symbols import symbols
import re 


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
CMUDICT_PATH = "D:\Master_SAR\MLA\PJT\Git\Matcha-TTS-etu-UPMC-ENSAM\matcha\\text_to_ID\cmudict-0.7b"
# CMUDICT_PATH = 'matcha/text_to_ID/cmudict_SPHINX_40'
cmu_dict = CMUDict(CMUDICT_PATH)


# Regex pour extraire les mots (avec apostrophes) et la ponctuation
_token_re = re.compile(r"([a-zA-Z']+)|([.,!?;:])")

def text_to_sequence(text):
  cleaned_text = english_cleaners(text)
  
  sequence = []
  for token in _token_re.findall(cleaned_text):
    word, punct = token  # token est un tuple (mot, ponctuation)
    
    if word:  # Si c'est un mot
        pronunciation = cmu_dict.lookup(word.upper())
        # print(pronunciation)
        
        if pronunciation:
          for ph in pronunciation[0].split(' '):
            #print(ph)
            sequence.append(_symbol_to_id['@' + ph])
            # print(sequence[-1])
        else:
          sequence.append(_symbol_to_id['<unk>'])

    elif punct:  # Si c'est un signe de ponctuation
      sequence.append(_symbol_to_id[punct])  # Ajoute l'ID du symbole
  return sequence


# Bloc d'exécution conditionnelle
if __name__ == "__main__":
    texte = "Hello world, I have 3 apples."
    sequence = text_to_sequence(texte)
    print(sequence)  # Affiche une seule fois, uniquement si le fichier est exécuté directement
