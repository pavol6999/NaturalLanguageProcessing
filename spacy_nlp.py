import spacy

sent2 = 'I have been to U.K. and U.S.A.'
nlp = spacy.load('en_core_web_sm')

### TOKENIZATION ###
tokens = nlp(sent2)
print([token.text for token in tokens])

### POS TAGGING ###
print([(token.text, token.pos_) for token in tokens])

### NAMED-ENTITY RECOGNITION ###
# locate and identify words and or phrases (names of persons, cities, books, locations...)
tokens3 = nlp('The book written by Hayden Liu in 2018 was sold at $30 in America')
print([(token_ent.text, token_ent.label_) for token_ent in tokens3.ents])

