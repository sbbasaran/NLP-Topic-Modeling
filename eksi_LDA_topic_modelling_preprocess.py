#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import nltk
import sys


# In[2]:


pd.set_option('display.max_rows', None)


# In[3]:


# Importing Data Set
df = pd.read_csv("/Users/ideakadikoy/Desktop/kadıkoy_df.csv")


# In[ ]:


df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])

df["Entry_Date"].dt.date

df["date"]=df["Entry_Date"].dt.date

import datetime
df["Entry_Date"] = df["date"].transform(lambda x: datetime.datetime.strftime(x, "%Y-%m-%d"))

df.groupby(["Entry_Date"])["Entry"].count()

pd.DataFrame(df.groupby(["Entry_Date"])["Entry"].count()).reset_index()


# In[5]:


df.info()


# # Text Preprocessing
# 

# In[6]:


import re


# ## Convert to lower case

# In[7]:


df['Entry'] = [token.lower() for token in df['Entry']]
df.head(5)


# ## Remove @ mentions and hyperlinks

# In[8]:


found = df[df['Entry'].str.contains('@')]
found.count()


# In[9]:


df.info()


# In[10]:


df['Entry'] = df['Entry'].replace('@[A-Za-z0-9]+', '', regex=True).replace('@[A-Za-z0-9]+', '', regex=True)
found = df[df['Entry'].str.contains('@')]
found.count()


# In[11]:


found = df[df['Entry'].str.contains('http')]
found.count()


# In[12]:


df['Entry'] = df['Entry'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
found = df[df['Entry'].str.contains('http')]
found.count()


# In[13]:


df.shape


# ## Remove Punctations & Emojies & Numbers

# In[14]:


sentences = df['Entry'].copy()
new_sent = []
i = 0
for sentence in sentences:
    new_sentence = re.sub('[0-9]+', '', sentence)
    new_sent.append(new_sentence)
    i += 1
    
df['Entry'] = new_sent
df['Entry'].head(5)


# In[15]:


import string

#table = str.maketrans('', '', string.punctuation)
sentences = df['Entry'].copy()
new_sent = []
for sentence in sentences:
    #words = sentence.split()
    
    #stripped = [letter for letter in sentence if letter not in string.punctuation]
    #new_sent.append(stripped)
    new_sent.append(sentence.translate(str.maketrans('', '', string.punctuation)))


# In[16]:


df['Entry'] = new_sent
df['Entry'].head(200)


# # Zemberek-NLP

# ##  Tokenization

# In[17]:


import time
import logging

from zemberek import (
    TurkishSpellChecker,
    TurkishSentenceNormalizer,
    TurkishSentenceExtractor,
    TurkishMorphology,
    TurkishTokenizer
)

logger = logging.getLogger(__name__)

morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)
extractor = TurkishSentenceExtractor()


# ## Sentence Normalization

# In[18]:


def normalize_long_text(text):
    normalized_sentences = [normalizer.normalize(word) for word in text]
    normalized_text = " ".join(normalized_sentences)
    return normalized_text


# In[19]:


sentences = df['Entry'].copy()
new_sent = []
start = time.time()

logger.info(f"Sentences normalized in: {time.time() - start} s")


# In[20]:


new_sent = df["Entry"]


# In[21]:


new_sent


# In[22]:


df.head()


# ## Stopwords

# In[23]:


# Turkish Stop Words
trstop = [
    'a', 'acaba', 'altı', 'altmış', 'ama', 'ancak', 'arada', 'artık', 'asla', 'aslında', 'aslında', 'ayrıca', 'az', 'bana',
    'bazen', 'bazı', 'bazıları', 'belki', 'ben', 'benden', 'beni', 'benim', 'beri', 'beş', 'bile', 'bilhassa', 'bin', 'bir',
    'biraz', 'birçoğu', 'birçok', 'biri', 'birisi', 'birkaç', 'birşey', 'biz', 'bizden', 'bize', 'bizi', 'bizim', 'böyle',
    'böylece', 'bu', 'buna', 'bunda', 'bundan', 'bunlar', 'bunları', 'bunların', 'bunu', 'bunun', 'burada', 'bütün', 'çoğu',
    'çoğunu', 'çok', 'çünkü', 'da', 'daha', 'dahi', 'dan', 'de', 'defa', 'değil', 'diğer', 'diğeri', 'diğerleri', 'diye',
    'doksan', 'dokuz', 'dolayı', 'dolayısıyla', 'dört', 'e', 'edecek', 'eden', 'ederek', 'edilecek', 'ediliyor', 'edilmesi',
    'ediyor', 'eğer', 'elbette', 'elli', 'en', 'etmesi', 'etti', 'ettiği', 'ettiğini', 'fakat', 'falan', 'filan', 'gene',
    'gereği', 'gerek', 'gibi', 'göre', 'hala', 'halde', 'halen', 'hangi', 'hangisi', 'hani', 'hatta', 'hem', 'henüz', 'hep',
    'hepsi', 'her', 'herhangi', 'herkes', 'herkese', 'herkesi', 'herkesin', 'hiç', 'hiçbir', 'hiçbiri', 'i', 'ı', 'için',
    'içinde', 'iki', 'ile', 'ilgili', 'ise', 'işte', 'itibaren', 'itibariyle', 'kaç', 'kadar', 'karşın', 'kendi', 'kendilerine',
    'kendine', 'kendini', 'kendisi', 'kendisine', 'kendisini', 'kez', 'ki', 'kim', 'kime', 'kimi', 'kimin', 'kimisi', 'kimse',
    'kırk', 'madem', 'mi', 'mı', 'milyar', 'milyon', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nedenle', 'nerde', 'nerede', 'nereye',
    'neyse', 'niçin', 'nin', 'nın', 'niye', 'nun', 'nün', 'o', 'öbür', 'olan', 'olarak', 'oldu', 'olduğu', 'olduğunu',
    'olduklarını', 'olmadı', 'olmadığı', 'olmak', 'olması', 'olmayan', 'olmaz', 'olsa', 'olsun', 'olup', 'olur', 'olur',
    'olursa', 'oluyor', 'on', 'ön', 'ona', 'önce', 'ondan', 'onlar', 'onlara', 'onlardan', 'onları', 'onların', 'onu', 'onun',
    'orada', 'öte', 'ötürü', 'otuz', 'öyle', 'oysa', 'pek', 'rağmen', 'sana', 'sanki', 'sanki', 'şayet', 'şekilde', 'sekiz',
    'seksen', 'sen', 'senden', 'seni', 'senin', 'şey', 'şeyden', 'şeye', 'şeyi', 'şeyler', 'şimdi', 'siz', 'siz', 'sizden',
    'sizden', 'size', 'sizi', 'sizi', 'sizin', 'sizin', 'sonra', 'şöyle', 'şu', 'şuna', 'şunları', 'şunu', 'ta', 'tabii',
    'tam', 'tamam', 'tamamen', 'tarafından', 'trilyon', 'tüm', 'tümü', 'u', 'ü', 'üç', 'un', 'ün', 'üzere', 'var', 'vardı',
    've', 'veya', 'ya', 'yani', 'yapacak', 'yapılan', 'yapılması', 'yapıyor', 'yapmak', 'yaptı', 'yaptığı', 'yaptığını',
    'yaptıkları', 'ye', 'yedi', 'yerine', 'yetmiş', 'yi', 'yı', 'yine', 'yirmi', 'yoksa', 'yu', 'yüz', 'zaten', 'zira'
] # https://github.com/ahmetax/trstop/blob/master/dosyalar/turkce-stop-words

nltk_trstop = [
    'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha',
    'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim',
    'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu',
    'tüm', 've', 'veya', 'ya', 'yani'
] # https://github.com/xiamx/node-nltk-stopwords/blob/master/data/stopwords/turkish

add_stop = [
    'a', 'bkz','b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş', 't',
    'u', 'ü', 'v', 'y', 'z', 'li', 'lı', 'si', 'sı', 'te', 'ta', 'ın', 'in', 'na', 'ne', 'ler', 'lar', 'de', 'da', 'nın', 'nin',
    'lık', 'ım', 'im', 'yok', 'di', 'dı'
]

stops = set(sorted(list(set(trstop).union(nltk_trstop).union(add_stop))))


# In[24]:


new_sent[:10]


# In[25]:


new_sent = new_sent.apply(lambda x: [word for word in x.split() if word not in stops])


# In[26]:


clean_sent = new_sent


# In[27]:


clean_sent[5]


# In[28]:


df["sents"] = [' '.join(item) for item in clean_sent]


# In[30]:


df["sents"]


# ##  Lemmatization

# In[32]:


all_unique_words = []

for sentence in df["sents"]:
    for word in sentence.split():
        if word not in all_unique_words:
            all_unique_words.append(word)


# In[33]:


len(all_unique_words)


# In[34]:


unique_words_to_lemmatized = {}

import zeyrek

analyzer = zeyrek.MorphAnalyzer()
#lem_sent = []
for word in all_unique_words:
    if i % 200 == 0:
        print(f"{i}/{len(all_unique_words)} | {i/len(all_unique_words)*100:.2f}%")

    lem_word = analyzer.lemmatize(word)
    unique_words_to_lemmatized[word] = lem_word[0][1][0]

    #lem_sent.append(normalized_sent)
    
print(f"{len(all_unique_words)}/{len(all_unique_words)} | {len(all_unique_words)/len(all_unique_words)*100:.2f}%")


# In[71]:


df["sents_lemma"] = df["sents"].apply(lambda x: ' '.join([unique_words_to_lemmatized[word] for word in x.split()]))


# In[72]:


df["sents_lemma"]


# In[75]:


df.head()


# In[77]:


df.to_csv("lemma_all_entries_df2.csv")


# # Data Visualization

# In[78]:


from textblob import TextBlob
import tweepy


# In[80]:


from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[81]:


common_words = get_top_n_words(df['sents'], 20)
common_df = pd.DataFrame(common_words, columns = ['sents', 'count'])
common_df.head()


# In[ ]:


common_df.groupby('sents').sum()['count'].sort_values(ascending=False).plot(
    kind='bar',
    figsize=(8, 6),
    xlabel = "Top Words",
    ylabel = "Count",
    title = "Bar Chart of Top Words Frequency")


# ## Top Bigrams

# In[83]:


def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2,2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words2 = get_top_n_bigram(df['sents'], 30)


# In[84]:


top_bigram = pd.DataFrame(common_words2, columns=['sents', "Count"])
top_bigram.head()


# ## Top Trigrams

# In[ ]:


top_bigram.groupby('sents').sum()['Count'].sort_values(ascending=False).plot(
    kind='bar',
    figsize=(8,6),
    xlabel = "Bigram Words",
    ylabel = "Count",
    title = "Bar chart of Bigrams Frequency")


# In[86]:


def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words3 = get_top_n_trigram(df['sents'], 30)
top_trigram = pd.DataFrame(common_words3, columns = ['sents' , 'Count'])
top_trigram.head(5)


# In[87]:


top_trigram.groupby('sents').sum()['Count'].sort_values(ascending=False).plot(
    kind='bar',
    figsize=(8,6),
    xlabel = "Trigram Words",
    ylabel = "Count",
    title = "Bar chart of Trigrams Frequency")


# # WordCloud

# In[89]:


import matplotlib.pyplot as plt


# In[90]:


from wordcloud import WordCloud, STOPWORDS

def creat_wordcloud(sents):
    comment_words = ''
    stopwords = set(STOPWORDS)
    
    # iterate through the csv file
    for val in sents:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 1200, height = 800,
                    background_color ='white',
                    max_words=3000,
                    stopwords = stopwords,
                    min_font_size = 10,
                    repeat = True).generate(comment_words)

    # plot the WordCloud image                       
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    plt.show()


# In[91]:


# All Entries
creat_wordcloud(df['sents'].values)


# # Remove most common words for GSDMM and LDA 
# 

# In[92]:


import dask.dataframe as dd


# In[93]:


def build_vocabulary(vocab_df, token_column, to_lower=True, chunksize=10000): #from the tsundoku module created by Eduardo Graells-Garrido :)
    if type(vocab_df) == pd.core.frame.DataFrame:
        vocab_df = dd.from_pandas(vocab_df, chunksize=chunksize)

    return (
        vocab_df[[token_column]]
        .explode(token_column)
        .assign(
            token=lambda x: x[token_column].str.lower() if to_lower else x[token_column]
        )["token"]
        .value_counts()
        .rename("frequency")
        .to_frame()
        .compute()
        .sort_values("frequency", ascending=False)
        .reset_index()
        .rename(columns={"index": "token"})
    )


# In[96]:


df["entries_tok"] = df["sents"].apply(lambda x: x.split())


# In[97]:


vocab = build_vocabulary(df, 'entries_tok')
vocab["token"][0:25]


# In[100]:


df["cat"]= df["Tweets_tok"].apply(lambda x: [item for item in x if item not in remove_list])


# In[101]:


df["cat"]


# In[102]:


df


# In[ ]:




