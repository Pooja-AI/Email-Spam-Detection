from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
labelencoder = LabelEncoder()
onehotencoding = OneHotEncoder()
abc = ['a all friend is none to']
labelencoder.fit(['a', 'all', 'friend', 'is', 'none', 'to'])
x=labelencoder.transform(["a", "friend", "to", "all", "is", "a", "friend", "to", "none"])
a = [['a'], ['friend'], ['to'], ['all'], ['is'], ['a'], ['friend'], ['to'], ['none']]
enc = onehotencoding.fit_transform(a).toarray()
vec = CountVectorizer(ngram_range=(2,2), token_pattern='[a-zA-Z]+')
count = vec.fit_transform(abc)
print(vec.vocabulary_)

print(x)
print(a)
