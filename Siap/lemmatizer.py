# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:42:40 2023

@author: User
"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
# =====================Prvi put kada se pokrece odkomentarisati========================================================
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# =============================================================================

lemmatizer = WordNetLemmatizer()

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

stop_words = set(stopwords.words('english'))
  
def stop_word_removal(sentence):
    word_tokens = word_tokenize(sentence)
    # converts the words in word_tokens to lower case and then checks whether 
    #they are present in stop_words or not
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    #with no lower case conversion
    filtered_sentence = []
      
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
            
    return " ".join(filtered_sentence)


def clean_sentence(question):
    return lemmatize_sentence(stop_word_removal(question))
