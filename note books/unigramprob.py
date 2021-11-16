#Finds out the unigram probabilites

import re
import math

SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

def read_data(file):
    with open(file, 'r') as f:
        return [re.split("\s+", line.rstrip('\n')) for line in f]

class UnigramLanguageModel:
    def __init__(self, sentences, smoothing=False):
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        for sentence in sentences:
            for word in sentence:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                if word != SENTENCE_START and word != SENTENCE_END:
                    self.corpus_length += 1
        self.unique_words = len(self.unigram_frequencies.keys()) - 2 #subtract START and END
        self.smoothing = smoothing

    def calculate_unigram_probability(self, word):
        numerator = self.unigram_frequencies.get(word, 0)
        denominator = self.corpus_length
        if self.smoothing:
            numerator += 1
            denominator += self.unique_words
        return float(numerator)/float(denominator)
    
    def sorted_vocabulary(self):
        vocab = list(self.unigram_frequencies.keys())
        vocab.sort()
        return vocab

#Print unigram probabilities
def print_unigram_probs(vocab, model):
    for word in vocab:
        if word != SENTENCE_START and word != SENTENCE_END:
            print(f'{word} : {round(model.calculate_unigram_probability(word),3)}', end=" -- ")

dataset = read_data("./sampledata.txt")

unsmoothed_model = UnigramLanguageModel(dataset)
smoothed_model = UnigramLanguageModel(dataset, smoothing=True)

print("U N I G R A M  Language Model")
print("\n- - Unsmoothed - -")
print_unigram_probs(unsmoothed_model.sorted_vocabulary(), unsmoothed_model)
print("\n- - Smoothed - -")
print_unigram_probs(smoothed_model.sorted_vocabulary(),  smoothed_model)
