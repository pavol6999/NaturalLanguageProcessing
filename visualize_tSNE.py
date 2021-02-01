from sklearn.manifold import TSNE
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names



def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
    return True



categories_3 = ['talk.religion.misc', 'comp.graphics', 'sci.space']
groups3 = fetch_20newsgroups(categories=categories_3)
all_names = set(names.words())

count_vector_sw = CountVectorizer(stop_words="english",max_features=5000)
lemmatizer = WordNetLemmatizer()


data_cleaned = []
for doc in groups3.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word)
                            for word in doc.split()
                            if is_letter_only(word) and 
                            word not in all_names)
    data_cleaned.append(doc_cleaned)

data_cleaned_count_3 = count_vector_sw.fit_transform(data_cleaned)


tsne_model = TSNE(n_components=2, perplexity=40, random_state=42, learning_rate=500)
data_tsne=tsne_model.fit_transform(data_cleaned_count_3.toarray())



import matplotlib.pyplot as plt
plt.scatter(data_tsne[:,0],data_tsne[:,1], c=groups3.target)
plt.show()