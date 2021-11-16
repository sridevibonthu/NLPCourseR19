import re

def generate_ngrams(s, n):
    s = s.lower()   # Convert to lowercases
    
    # Replace all none-alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]
    
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

s = input("Enete text : ")
n = int(input("Enter value for n : "))
print(generate_ngrams(s, n))