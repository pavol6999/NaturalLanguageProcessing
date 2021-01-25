from nltk.tokenize import word_tokenize
import nltk

sent = "I am reading a book\nIt is Python Machine Learning By Example,\n 2nd edition."
sent2 = 'I have been to U.K. and U.S.A.'

print(f"\nThe original message: \n{sent}\n")


### TOKENIZATION ###

# word_tokenize does not split the sentence  after each punctuation mark or whitespace
print(f"(TOKEN) Split message 1: \n {word_tokenize(sent)}")

# word_tokenize recognizes patterns such as U.K. and U.S.A.
print(f"(TOKEN) Split message 2: \n {word_tokenize(sent2)}")


### POS TAGGING ###

# Use nltk.help.upenn_tagset(tag) to check the meaning of a tag
tokens = word_tokenize(sent)
print(f"\n(POS) Split message 1: {nltk.pos_tag(tokens)}\n\n")


### Stemming and lemmatization ### 
# reverting a word to its root form (learning -> learn), stemming may chop letters from the stemmed word if necessary
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
print("(STEMMING) ",porter_stemmer.stem('learning'))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print("(LEMM) ", lemmatizer.lemmatize('machines'))

