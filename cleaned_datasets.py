from nltk.corpus import names
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
    return True




all_names = set(names.words())
groups = fetch_20newsgroups()
count_vector_sw = CountVectorizer(stop_words="english", max_features=500)



lemmatizer = WordNetLemmatizer()
data_cleaned = []
for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word)
                            for word in doc.split()
                            if is_letter_only(word) and 
                            word not in all_names)
    data_cleaned.append(doc_cleaned)
data_cleaned_count = count_vector_sw.fit_transform(data_cleaned)




