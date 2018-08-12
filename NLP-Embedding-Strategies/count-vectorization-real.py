from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

# Create our vectorizer
vectorizer = CountVectorizer()

# Let's fetch all the possible text data
newsgroups_data = fetch_20newsgroups()

# Why not inspect a sample of the text data?
print('Sample 0: ')
print(newsgroups_data.data[0])
print()

# Create the vectorizer
vectorizer.fit(newsgroups_data.data)

# Let's look at the vocabulary:
print('Vocabulary: ')
print(vectorizer.vocabulary_)
print()

# Converting our first sample into a vector
v0 = vectorizer.transform([newsgroups_data.data[0]]).toarray()[0]
print('Sample 0 (vectorized): ')
print(v0)
print()

# It's too big to even see...
# What's the length?
print('Sample 0 (vectorized) length: ')
print(len(v0))
print()

# How many words does it have?
print('Sample 0 (vectorized) sum: ')
print(np.sum(v0))
print()

# What if we wanted to go back to the source?
print('To the source:')
print(vectorizer.inverse_transform(v0))
print()

# So all this data has a lot of extra garbage... Why not strip it away?
newsgroups_data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

# Why not inspect a sample of the text data?
print('Sample 0: ')
print(newsgroups_data.data[0])
print()

# Create the vectorizer
vectorizer.fit(newsgroups_data.data)

# Let's look at the vocabulary:
print('Vocabulary: ')
print(vectorizer.vocabulary_)
print()

# Converting our first sample into a vector
v0 = vectorizer.transform([newsgroups_data.data[0]]).toarray()[0]
print('Sample 0 (vectorized): ')
print(v0)
print()

# It's too big to even see...
# What's the length?
print('Sample 0 (vectorized) length: ')
print(len(v0))
print()

# How many words does it have?
print('Sample 0 (vectorized) sum: ')
print(np.sum(v0))
print()

# What if we wanted to go back to the source?
print('To the source:')
print(vectorizer.inverse_transform(v0))
print()
