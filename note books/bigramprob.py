from unigramprob import UnigramLanguageModel 
import re
import math

SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

def read_data(file):
    with open(file, 'r') as f:
        return [re.split("\s+", line.rstrip('\n')) for line in f]

class BigramLanguageModel(UnigramLanguageModel):
    def __init__(self, sentences, smoothing=False):
        UnigramLanguageModel.__init__(self, sentences, smoothing)
        self.bigram_frequencies = dict()
        self.unique_bigrams = set()
        for sentence in sentences:
            prev_word = None
            for word in sentence:
                if prev_word != None:
                    self.bigram_frequencies[(prev_word, word)] = self.bigram_frequencies.get((prev_word, word), 0) + 1
                    if prev_word != SENTENCE_START and prev_word != SENTENCE_END:
                        self.unique_bigrams.add((prev_word, word))
                prev_word = word
        self.unique_bigram_words = len(self.unigram_frequencies)
    
    def calculate_bigram_probability(self, prev_word, word):
        numerator = self.bigram_frequencies.get((prev_word, word), 0)
        denominator = self.unigram_frequencies.get(prev_word, 0)
        if self.smoothing:
            numerator += 1
            denominator += self.unique_bigram_words
        return float(numerator) / float(denominator)


#Print unigram probabilities
def print_bigram_probs(vocab, model):
    for word1 in vocab:
        if word1 != SENTENCE_END:
            for word2 in vocab:
                if word2 != SENTENCE_START:
                    print(f'({word1},{word2}) : {round(model.calculate_bigram_probability(word1, word2),3)}', end=" -- ")


dataset = read_data("./sampledata.txt")

unsmoothed_model = BigramLanguageModel(dataset)
smoothed_model = BigramLanguageModel(dataset, smoothing=True)

print("B I G R A M  Language Model")
print("\n- - Unsmoothed - -")
print_bigram_probs(unsmoothed_model.sorted_vocabulary(), unsmoothed_model)
print("\n- - Smoothed - -")
print_bigram_probs(smoothed_model.sorted_vocabulary(),  smoothed_model)

