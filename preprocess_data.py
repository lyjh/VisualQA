import json
import pickle
import operator

annFile = '../data/v2_mscoco_val2014_annotations.json'
quesFile = '../data/v2_OpenEnded_mscoco_val2014_questions.json'

ques = questions = json.load(open(quesFile, 'r'))
ques = ques['questions']
qa = json.load(open(annFile, 'r'))
qa = qa['annotations']

quesId2Question = {}
for q in ques:
	quesId2Question[q['question_id']] = q['question']

dict = []
for q in ques:
	for w in q['question'][:-1].split():
		dict.append(w)

dict.append('<Unk>')
unique = set(dict)
with open("unique.p", "wb") as encoded_pickle:
	pickle.dump(unique, encoded_pickle)

answer_dict = {}
for ann in qa:
	for answer in ann['answers']:
		w = answer['answer']
		if w in answer_dict:
			answer_dict[w] += 1
		else:
			answer_dict[w] = 1

sorted_answer = sorted(answer_dict.items(), key=operator.itemgetter(1), reverse=True)
answer2index = {v[0]:i for i,v in enumerate(sorted_answer[:1000])}
answer2index['<Unk>'] = 1000
index2answer = {v:i for i,v in answer2index.items()}
with open("answer2index.p", "wb") as encoded_pickle:
	pickle.dump(answer2index, encoded_pickle)
with open("index2answer.p", "wb") as encoded_pickle:
	pickle.dump(index2answer, encoded_pickle)

f = open('vqa_dataset_v2.txt', 'w')
f.write("image_id\tquestion\tanswer\n")
for ann in qa:
	question = quesId2Question[ann['question_id']]
	imgFileName = 'COCO_val2014_'+str(ann['image_id']).zfill(12)+'.jpg'
	for a in ann['answers']:
		answer = a['answer']
		answerIndex = answer2index[answer] if answer in answer2index else 1000
		f.write(imgFileName + "\t" + question + "\t" + str(answerIndex) + "\n")