# -*- coding: utf-8 -*-
# MAX_LEN_INPUT, MAX_LEN_TARGET 16 273
# (311, 273) (311, 273) (311, 16) 1118 1117 <PAD>
# 3.12 zapoceto treniranje

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.stem import WordNetLemmatizer
import numpy as np 
import re
import keras
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Concatenate, Dropout, Attention
from attention import AttentionLayer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sumarize import summarize
from tensorflow.keras.preprocessing.text import Tokenizer
from lemmatizer import clean_sentence
from statistics import mean
from nltk.translate.bleu_score import sentence_bleu
from json.decoder import JSONDecodeError
import matplotlib.pyplot as plt

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

# Opening JSON files
f = open('./dataset/8_icliniqQAs.json')           # 465;  MAX_LEN_INPUT, MAX_LEN_TARGET 16 158
f1 = open('./dataset/9_questionDoctorQAs.json')   # 5679; MAX_LEN_INPUT, MAX_LEN_TARGET 21 107
f2 = open('./dataset/7_ehealthforumQAs.json')     # 171;  MAX_LEN_INPUT, MAX_LEN_TARGET 21 129
f4 = open('./dataset/intents.json')             #  188 ; 6 124
f3 = open('./dataset/10_webmdQAs.json')         # 23437 (nema summary pitanja - predugacko) ; MAX_LEN_INPUT, MAX_LEN_TARGET 82 1310

# returns JSON object as a dictionary
data = json.load(f)     # new_data.json
data1 = json.load(f1)
data2 = json.load(f2)   # new_data_f2.json
data4 = json.load(f4)   # intents_final.json
data3 = json.load(f3)   # jos uvek ne radimo sa ovim izvorom

# **** FILE f - samo 1 je potrebno pokrenuti, nakon toga sve nove podatke imamo u novom json fajlu
# for pair in data:
#     del pair['question_text']
#     del pair['tags']
#     del pair['url']
#     # summarization of doctor answers 
#     if (len(pair['answer']) <= 150):
#         cleaned_answer = pair['answer']
#     else:    
#         if (150/len(pair['answer']) <= 0.3):        
#             summarization = summarize(pair['answer'], 0.3)
#             cleaned_answer = summarization
#         else:
#             summarization = summarize(pair['answer'], 150/len(pair['answer']))  # len(summarization) <= 150
#             cleaned_answer = summarization
#     del pair['answer']
#     pair.update({"answer": cleaned_answer})
# with open('./dataset/new_data.json', 'w') as f:
#     json.dump(data, f, indent=2)


# **** FILE f2 - samo 1 je potrebno pokrenuti, nakon toga sve nove podatke imamo u novom json fajlu
# for pair in data2:
#     del pair['tags']
#     del pair['url']  
#     # summarization of doctor answers 
#     if (len(pair['answer']) <= 150):
#         cleaned_answer = pair['answer']
#     else:    
#         if (150/len(pair['answer']) <= 0.3):        
#             summarization = summarize(pair['answer'], 0.3)
#             cleaned_answer = clean_text(summarization)
#         else:
#             summarization = summarize(pair['answer'], 150/len(pair['answer']))  # len(summarization) <= 150
#             cleaned_answer = clean_text(summarization)
#     del pair['answer']
#     pair.update({"answer": cleaned_answer})
# with open('./dataset/new_data_f2.json', 'w') as filename:
#     json.dump(data2, filename, indent=2)


# **** FILE f4 - samo 1 je potrebno pokrenuti,
# for pair in data4["intents"]:
#     print(pair)
#     del pair['tag']
#     del pair['context_set']
# with open('./dataset/intents_final.json', 'w') as filename:
#     json.dump(data4, filename, indent=2)

# TODO: otvaranje novih, sredjenih fajlova
f_summarized1 = open('./dataset/new_data.json')
data_summarized1 = json.load(f_summarized1)

f_summarized2 = open('./dataset/new_data_f2.json')
data_summarized2 = json.load(f_summarized2)

f_intents = open('./dataset/intents_final.json')
data_intents = json.load(f_intents)

patient_questions = []
doctor_answers = []

# for i in data1:
#     patient_questions.append(i['question'])
#     doctor_answers.append(i['answer'])

for i in data_summarized1:
    patient_questions.append(i['question'])
    doctor_answers.append(i['answer'])

for i in data_summarized2:
    patient_questions.append(i['question'])
    doctor_answers.append(i['answer'])

# for i in data_intents["intents"]:
#     for pattern in i['patterns']:    
#         patient_questions.append(pattern)
#         doctor_answers.append(i['responses'][0])

# Closing file
f.close()
f1.close()
f2.close()
f3.close()
f4.close()
f_summarized1.close()
f_summarized2.close()
f_intents.close()


tidy_patient_questions = []
for conve in patient_questions:
    text = clean_text(conve)
    text = clean_sentence(text)
    tidy_patient_questions.append(text)

tidy_doctor_answers = []
for conve in doctor_answers:    
    text = clean_text(conve)
    tidy_doctor_answers.append(text)
    
    
# samo patient-doctor data
X_train, X_test, y_train, y_test = train_test_split(tidy_patient_questions, 
                                                    tidy_doctor_answers, test_size=0.33, random_state=42)

###  count word occurences ###
word2count = {}

for line in X_train:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
for line in y_train:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

###  remove less frequent ### ako stavimo trash 1, ne radi se remove uopste, uzima se ceo dict
thresh = 1  # TODO: ???
vocab = {}
word_num = 0
for word, count in word2count.items():
    if count >= thresh:
        vocab[word] = word_num
        word_num += 1

# oznacavanje teksta koji ulazi u dekoder tokenima
for i in range(len(y_train)):
    y_train[i] = '<SOS> ' + y_train[i] + ' <EOS>'

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
x = len(vocab)
for token in tokens:
    vocab[token] = x
    x += 1

vocab['cameron'] = vocab['<PAD>']
vocab['<PAD>'] = 0

### inv answers dict ###
inv_vocab = {w:v for v, w in vocab.items()}

# tokenizacija (rucno odradjena)
encoder_inp = []
for line in X_train:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
    encoder_inp.append(lst)

decoder_inp = []
for line in y_train:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])        
    decoder_inp.append(lst)


MAX_LEN_INPUT = max(len(seq) for seq in encoder_inp)
MAX_LEN_TARGET = max(len(seq) for seq in decoder_inp)
print('MAX_LEN_INPUT, MAX_LEN_TARGET', MAX_LEN_INPUT, MAX_LEN_TARGET)
    
# padding
encoder_inp = pad_sequences(encoder_inp, MAX_LEN_INPUT, padding='post', truncating='post')
decoder_inp = pad_sequences(decoder_inp, MAX_LEN_TARGET, padding='post', truncating='post')

decoder_final_output = []
for i in decoder_inp:
    decoder_final_output.append(i[1:]) 

decoder_final_output = pad_sequences(decoder_final_output, MAX_LEN_TARGET, padding='post', truncating='post')

VOCAB_SIZE = len(vocab)

# print(decoder_final_output.shape, decoder_inp.shape, encoder_inp.shape, len(vocab), len(inv_vocab), inv_vocab[0])
decoder_final_output = to_categorical(decoder_final_output, len(vocab))

# GLOVE
embeddings_index = {}
with open('glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
print("Glove Loded!")


embedding_dimention = 50
def embedding_matrix_creater(embedding_dimention, word_index):
    embedding_matrix = np.zeros((len(word_index)+1, embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
embedding_matrix = embedding_matrix_creater(50, word_index=vocab)    


# embedding
embed = Embedding(VOCAB_SIZE+1, 50, input_length=MAX_LEN_INPUT, trainable=True)
embed.build((None,))
embed.set_weights([embedding_matrix])

enc_inp = Input(shape=(MAX_LEN_INPUT, ))
enc_embed = embed(enc_inp)
enc_lstm = Bidirectional(LSTM(400, return_state=True, dropout=0.05, return_sequences = True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = enc_lstm(enc_embed)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
enc_states = [state_h, state_c] # hidden_state, cell_state

dec_inp = Input(shape=(MAX_LEN_TARGET, ))
dec_embed = embed(dec_inp)
dec_lstm = LSTM(400*2, return_state=True, return_sequences=True, dropout=0.05)
output, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

# attention
attn_layer = AttentionLayer()
attn_op, attn_state = attn_layer([encoder_outputs, output]) # attn_op - Output context vector sequence for the decoder. 
                                                            # This is to be concat with the output of decoder.
decoder_concat_input = Concatenate(axis=-1)([output, attn_op])

# a dense layer is used as the output for the network
dec_dense = Dense(VOCAB_SIZE, activation='softmax')
final_output = dec_dense(decoder_concat_input)

# sklapanje modela
model = Model([enc_inp, dec_inp], final_output)
model.summary()

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# # treniranje modela
# history = model.fit([encoder_inp, decoder_inp], decoder_final_output, epochs=100, batch_size=24, validation_split=0.15)
# model.save('chatbot_doctor_glove_sources_f_f2.h5')
# model.save_weights('chatbot_weights_doctor_glove_sources_f_f2.h5')
model.load_weights('chatbot_weights_doctor_glove_sources_f_f2.h5')

# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.savefig('acc_f_f2.svg')
# plt.show()

# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.savefig('loss_f_f2.svg')
# plt.show()

enc_model = tf.keras.models.Model(enc_inp, [encoder_outputs, enc_states])

decoder_state_input_h = tf.keras.layers.Input(shape=( 400 * 2,))
decoder_state_input_c = tf.keras.layers.Input(shape=( 400 * 2,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = dec_lstm(dec_embed , initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

dec_model = tf.keras.models.Model([dec_inp, decoder_states_inputs], [decoder_outputs] + decoder_states)


print("##########################################")
print("#       start chatting ver. 1.0          #")
print("##########################################")


def evaluate():
    chatbot_answers = []
    for question in X_test:
        chatbot_answers.append(chatbot(question))
    bleu_score_list = []
    for real_answer, chatbot_answer in zip(y_test, chatbot_answers):
        ans = clean_text(real_answer)
        a = ans.split()
        b = chatbot_answer.split()
        bleu_score = sentence_bleu(a, b, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_score_list.append(bleu_score)
        print('Cumulative 1-gram: %f' % sentence_bleu(a, b, weights=(1, 0, 0, 0)))
        # print('Cumulative 2-gram: %f' % sentence_bleu(a, b, weights=(0.5, 0.5, 0, 0)), sentence_bleu(a, b, weights=(0.5, 0.5, 0, 0)))
        # print('Cumulative 3-gram: %f' % sentence_bleu(a, b, weights=(0.33, 0.33, 0.33, 0)), sentence_bleu(a, b, weights=(0.33, 0.33, 0.33, 0)))
        # print('Cumulative 4-gram: %f' % sentence_bleu(a, b, weights=(0.25, 0.25, 0.25, 0.25)), sentence_bleu(a, b, weights=(0.25, 0.25, 0.25, 0.25)))
        print("------------------------------------------------------------------------------")
    print('Bleu score: %f' % mean(bleu_score_list), mean(bleu_score_list))   
    
    
def chatbot(text):
    try:
        text = clean_text(text)
        text = clean_sentence(text)
        prepro = [text]
        
        txt = []
        for x in prepro:
            lst = []
            for y in x.split():
                try:
                    lst.append(vocab[y])
                except:
                    lst.append(vocab['<OUT>'])
            txt.append(lst)
        txt = pad_sequences(txt, MAX_LEN_INPUT, padding='post')

        ###
        enc_op, stat = enc_model.predict( txt )

        empty_target_seq = np.zeros( ( 1 , 1) )
        empty_target_seq[0, 0] = vocab['<SOS>']
        stop_condition = False
        decoded_translation = ''

        while not stop_condition :
            dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + stat )
            
            ###########################
            attn_op, attn_state = attn_layer([enc_op, dec_outputs])
            decoder_concat_input = Concatenate(axis=-1)([dec_outputs, attn_op])
            decoder_concat_input = dec_dense(decoder_concat_input)
            ###########################

            sampled_word_index = np.argmax( decoder_concat_input[0, -1, :] )
            sampled_word = inv_vocab[sampled_word_index] + ' '

            if sampled_word != '<EOS> ':
                decoded_translation += sampled_word           

            if sampled_word == '<EOS> ' or len(decoded_translation.split()) > MAX_LEN_TARGET:
                stop_condition = True

            empty_target_seq = np.zeros( ( 1 , 1 ) )  
            empty_target_seq[ 0 , 0 ] = sampled_word_index
            stat = [ h , c ] 

        print("CHATBOT: ")
        print("============================================================================================")
        print(decoded_translation)
        return decoded_translation
    except:
        # print("sorry I didn't understand that, please type again :( ")
        return "sorry I didn't understand that, please type again :("


option = input("Please choose testing type - M (Manual) or A (Automatic):")

if option == 'A':
    evaluate()
else:   
    prepro1 = ""
    while prepro1 != 'q':
        prepro1 = input("YOU : ")
        print(chatbot(prepro1))
