# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 23:19:34 2023

@author: User
"""

import json
from sklearn.model_selection import train_test_split

patient_questions = []
doctor_answers = []

f = open('./dataset/intents_processed.json')
data = json.load(f)

for pair in data:
      patient_questions.append(pair["question"])
      doctor_answers.append(pair["answer"])
      
X_train, X_test, y_train, y_test = train_test_split(patient_questions, 
                                                    doctor_answers, test_size=0.2, random_state=42)

data_ouput = []
for i, j in zip(X_train, y_train):
    qa_pairs = {
    "question": i,
    "answer": j
    }
    data_ouput.append(qa_pairs)
    
data_ouput1 = []
for i, j in zip(X_test, y_test):
    qa_pairs = {
    "question": i,
    "answer": j
    }
    data_ouput1.append(qa_pairs)
    
full_data_json = json.dumps(data_ouput,indent=4)
    
with open('conversation_dataset_only_intents_train.json', 'w') as outfile:
    outfile.write(full_data_json)
    
full_data_json = json.dumps(data_ouput1,indent=4)
    
with open('conversation_dataset_only_intents_test.json', 'w') as outfile:
    outfile.write(full_data_json)