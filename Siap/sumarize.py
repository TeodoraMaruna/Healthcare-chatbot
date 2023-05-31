# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 22:28:22 2023

@author: User
"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

def summarize(text, per):
    nlp = spacy.load('en_core_web_md')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary

# 20% od ukupne recenice
# sentence = 'i constantly have bursts of anger for no reason once it is over i usually wish i would not have done it and go from being absolutely angry and freaking out to completely calm. my father is the same way i have constant anxiety and im almost never completely happy about something. my moods are all over the place but the most dominating and often reoccurring is anger i literally wake up in the morning angry for no reason. i was never like this before i hit about 14 i just turned 21 and it seems to be getting increasingly worse. it just feels like i am someone else completely when it is happening an i just do not feel like i can stop no matter who i am taking it out on then i snap out of it after it finally exhausts me and it is just almost like it was a different person acting that way. i do not have the money to go to the doctor and i do not have insurance. i feel like i can not handle this emotional roller-coaster anymore. does this classify under bipolar and if so what type?' 
# summarization = summarize(sentence, 35/len(sentence))  # len(summarization) <= 150
# print(len(sentence), len(summarization))
# print(summarization)