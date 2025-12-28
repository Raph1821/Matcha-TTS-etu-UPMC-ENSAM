'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''


import inflect
import re
from unidecode import unidecode
from .norm_numbers import normalize_numbers
# from matcha.text_to_ID.cmudict import CMUDict


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


# Fonction pour phonétiser le texte avec CMUDict
# def _phonetize_with_cmudict(text, cmudict):
#     words = text.split(' ')
#     phonetized_words = []
#     for word in words:
#         pronunciation = cmudict.lookup(word)
#         if pronunciation:
#             phonetized_words.append(pronunciation[0])  # Prend la première prononciation
#         else:
#             phonetized_words.append(word)  # Garde le mot original si non trouvé
#     return ' '.join(phonetized_words)

# Fonction `english_cleaners` modifiée pour inclure la phonétisation
def english_cleaners(text, cmudict_path=None):
    '''
    Pipeline for English text, including number and abbreviation expansion,
    and optional phonetization using CMUDict.
    '''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)

    # # Étape de phonétisation si un chemin vers CMUDict est fourni
    # if cmudict_path:
    #     cmudict = CMUDict(cmudict_path)
    #     text = _phonetize_with_cmudict(text, cmudict)

    return text


