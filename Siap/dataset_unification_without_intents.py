import json

f_summarized1 = open('./dataset/new_data.json')
data_summarized1 = json.load(f_summarized1)

f_summarized2 = open('./dataset/new_data_f2.json')
data_summarized2 = json.load(f_summarized2)

f1 = open('./dataset/9_questionDoctorQAs.json')
data1 = json.load(f1)

# f_intents = open('./dataset/intents_final.json')
# data_intents = json.load(f_intents)


def fill_data1(data_input, data_ouput):
    for i in data_input:
        qa_pairs = {
        "question": i["question"],
        "answer": i["answer"]
        }
        data_ouput.append(qa_pairs)

def fill_data_intents(data_input, data_ouput):
    for i in data_input["intents"]:
        for pattern in i['patterns']:
             qa_pairs = {
                "question": pattern,
                "answer": i['responses'][0]
                }
             data_ouput.append(qa_pairs)



full_data = []
fill_data1(data1, full_data)
fill_data1(data_summarized1, full_data)
fill_data1(data_summarized2, full_data)
# fill_data_intents(data_intents, full_data)
f1.close()
f_summarized1.close()
f_summarized2.close()
# f_intents.close()
full_data_json = json.dumps(full_data,indent=4)

with open('conversation_dataset_without_intents.json', 'w') as outfile:
    outfile.write(full_data_json)