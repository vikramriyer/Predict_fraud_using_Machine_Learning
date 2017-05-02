from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
s1 = "hi Katie the self driving car will be late Best Sebastian"
s2 = "Hi Sebastian the machine learning class will be great great great Best Katie"
s3 = "Hi Katie, the machine learning class will be most excellent"

email_list = [s1, s2, s3]
bag_of_words = vectorizer.fit(email_list)
bag_of_words = vectorizer.transform(email_list)

# below line prints everything about the above three lines, of which each one is considered a document
print bag_of_words
# THe result of above print showed one row, (1, 6)  3 => by looking at the above doc we can guess that
# the word would be great and hence we try to see what no is 'great'
# Note: the indexes for documents are from 0, 1, ... i.e. s1 = doc0, s2 = doc1, ..
print vectorizer.vocabulary_.get("great")
# We can also find which word is associateed to a feature/number => opposite of what we did above

# Removing stopwords
from nltk.corpus import stopwords
sw = stopwords.words("english")

#nltk.download() just in case there is an error trying to get the corpus
print sw
print len(sw)

#Now to form a basic root of similar words, we use stemmer algos
# for ex: unresponsive, respond, response, etc will form the same root: respon
# Note: idea is not to minimize the data by replacing them, since respon is anyway not a word, but to 
# maintain data without loosing it
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
print stemmer.stem("response")
print stemmer.stem("responsibility")
print stemmer.stem("unresponsiveness")
print stemmer.stem("responsivity")

