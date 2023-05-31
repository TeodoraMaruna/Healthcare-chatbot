import re
from lemmatizer import clean_sentence
import json


def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)    
    txt = re.sub(r"n't", " not", txt)
    txt = re.sub(r"n'", "ng", txt)
    txt = re.sub(r"'bout", "about", txt)
    txt = re.sub(r"'til", "until", txt)
    txt = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", txt)    
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt


def create_vocab():
    f1 = open('conversation_dataset1.json')
    data = json.load(f1)
    patient_questions = []
    doctor_answers = []
    
    for qa_pair in data:
        patient_questions.append(qa_pair["question"])
        doctor_answers.append(qa_pair["answer"])
    
    vocab = {}
    tidy_patient_questions = []
    for conve in patient_questions:
        text = clean_text(conve)
        text = clean_sentence(text)
        tidy_patient_questions.append(text)
    
    tidy_doctor_answers = []
    for conve in doctor_answers:    
        text = clean_text(conve)
        tidy_doctor_answers.append(text)
   
    word2count = {}
    
    for line in tidy_patient_questions:
        for word in line.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1
                
    for line in tidy_doctor_answers:
        for word in line.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1
    
    ###  remove less frequent ### ako stavimo trash 1, ne radi se remove uopste, uzima se ceo dict
    thresh = 3  # TODO: ???
    
    word_num = 0
    for word, count in word2count.items():
        if count >= thresh:
            vocab[word] = word_num
            word_num += 1    
        
    vocab_json = json.dumps(vocab,indent=4)
    with open('vocab.json', 'w') as outfile:
        outfile.write(vocab_json)
        
create_vocab()