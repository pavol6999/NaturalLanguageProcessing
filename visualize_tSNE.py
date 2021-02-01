from sklearn.manifold import TSNE
categories_3 = ['talk.religion.misc', 'comp.graphics', 'sci.space']
groups3 = fetch_20newsgroups(categories=categories_3)

tsne_model = TSNE(n_components=2, perplexity=40, random_state=42, learning_rate=500)
data_tsne=tsne_model.fit_transform(data_cleaned_count_3.toarray())
