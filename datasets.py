import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt


groups = fetch_20newsgroups()
sns.distplot(groups.target)

plt.show()