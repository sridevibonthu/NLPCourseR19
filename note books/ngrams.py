import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

def get_ngrams(text, n ):
    n_grams = ngrams(word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]

for i in range(1,5):
    print(i, '-grams: ', get_ngrams('This is the simplest text i could think of', i ))