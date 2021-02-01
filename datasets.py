import seaborn as sns
import tkinter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt


groups = fetch_20newsgroups()
sns.displot(groups.target)

# distribution of classes
# plt.show()


### BoW model - bag of words ###

# initialize count vectorizer
count_vector = CountVectorizer(max_features=500)

# capture the top 500 and generate a token count matrix
data_count = count_vector.fit_transform(groups.data)

# print top 500 features uncleaned -> most popular tokens are numbers!
# print(count_vector.get_feature_names())



## dropping stop words ##
# stop words add noise to the BoW model 
# stop words are common words that provide little value in helping documents differentiate themselves

from sklearn.feature_extraction import stop_words
print(stop_words.ENGLISH_STOP_WORDS)


