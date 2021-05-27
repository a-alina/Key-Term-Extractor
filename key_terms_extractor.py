from lxml import etree
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class KeyTermsExtractor():
    """Represent a key term extractor from xml document."""

    def __init__(self):
        """initilize an extractor with file name."""
        self.document = 'news.xml'
        self.headers = []
        self.news = []
        self.key_terms = []
        self.key_terms_weighted = []
        self.text_extraxtion()

    def __init__(self):
        self.document = 'news.xml'
        self.headers = []
        self.news = []
        self.key_terms = []
        self.key_terms_weighted = []
        self.text_extraxtion()

    def text_extraxtion(self):
        """Extracting the headers and the text"""
        root = etree.parse(self.document).getroot()
        for news in root[0]:
            self.headers.append(news[0].text)
            self.news.append(news[1].text)
        self.term_extraction()
        self.tfidf()

    def most_common(self):
        """Outputs the news head and 5 most commom keywords."""
        for head, tokens in zip(self.headers, self.key_terms_weighted):
            sort = sorted(tokens, key=lambda x: (x[1], x[0]), reverse=True)
            most_common = [token for token, count in sort[:5]]
            print(f'{head}:')
            print(' '.join(most_common))
            print()

    def term_extraction(self):
        "Preprocesing news and fileting noun terms."
        lemmatizer = WordNetLemmatizer()
        for news in self.news:
            tokinized_news = word_tokenize(news.lower())
            lemmatized_news = [lemmatizer.lemmatize(i) for i in tokinized_news]
            clean_tokens = [i for i in lemmatized_news if i not in stopwords.words('english') and list(string.punctuation).count(i) != 1]
            noun_tokens = [i for i in clean_tokens if nltk.pos_tag([i])[0][1] == 'NN']
            self.key_terms.append(noun_tokens)

    def tfidf(self):
        """Filetering most common nouns using TF-IDF."""
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([' '.join(i) for i in self.key_terms])
        terms = vectorizer.get_feature_names()
        sorted_terms = pd.DataFrame(matrix.toarray(),
                      columns=vectorizer.get_feature_names())
        for i in range(10):
            top_ten = sorted_terms.loc[i].sort_values(ascending=False).head(10)
            self.key_terms_weighted.append([(term, value) for term, value in top_ten.items()])

news = KeyTermsExtractor()
news.most_common()
