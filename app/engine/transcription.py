import re

def transcribe(word):
    word = re.sub('lc̱a','lcä',word)
    word = re.sub('a̱','ä',word)
    word = re.sub('⸜','',word)
    word = word.replace("\u0331", "")
    return word

print(transcribe('pᵤḵa̱ḻ⸜'))