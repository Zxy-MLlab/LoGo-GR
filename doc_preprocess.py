import numpy as np
import os
import json
from nltk.tokenize import WordPunctTokenizer
import argparse
import copy
parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type = str, default =  "../data")
parser.add_argument('--out_path', type = str, default = "prepro_data")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16
train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

word2id = json.load(open(os.path.join(out_path, 'word2id.json'), "r"))
rel2id = json.load(open(os.path.join(out_path, 'rel2id.json'), "r"))
id2rel = {v:u for u,v in rel2id.items()}
json.dump(id2rel, open(os.path.join(out_path, 'id2rel.json'), "w"))
fact_in_train = set([])
fact_in_dev_train = set([])

def init(data_file_name, rel2id, max_length = 512, is_training = True, suffix=''):

	ori_data = json.load(open(data_file_name))

	data = []

	for i in range(len(ori_data)):
		doc = copy.deepcopy(ori_data[i])
		sen_len_list = np.cumsum(np.array([0] + [len(sen) for sen in doc["sents"]]))
		doc["sents"] = [x for sen in doc["sents"] for x in sen]
		mention_list = []
		for d_i, entity in enumerate(doc["vertexSet"]):
			for mention in entity:
				mention["pos"][0] = int(mention["pos"][0]) + int(sen_len_list[int(mention["sent_id"])])
				mention["pos"][1] = int(mention["pos"][1]) + int(sen_len_list[int(mention["sent_id"])])
				mention["id"] = len(mention_list)
				mention_list.append((d_i, len(mention_list), mention["pos"][0], mention["pos"][1], mention["sent_id"]))

		dt = np.dtype([('id', int), ('ent_id', int), ('pos_s', int), ('pos_e', int), ('sent_id', int)])
		mention_list = np.array(mention_list, dtype=dt)
		mention_list = np.sort(mention_list, order='pos_s')
		mentions = [{"id": int(m), "ent_id": int(id), "pos": [int(s), int(e)], "sent_id": int(sen)} for id, m, s, e, sen
					in mention_list]
		doc["mentions"] = mentions


		Ls = [0]  #存储每个句子的长度
		L = 0
		#计算每个实例中所有句子的长度
		for x in ori_data[i]['sents']:
			L += len(x)
			Ls.append(L)

		vertexSet =  ori_data[i]['vertexSet']
		# point position added with sent start position
		for j in range(len(vertexSet)):
			for k in range(len(vertexSet[j])):
				vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

				sent_id = vertexSet[j][k]['sent_id']
				dl = Ls[sent_id]
				pos1 = vertexSet[j][k]['pos'][0]
				pos2 = vertexSet[j][k]['pos'][1]
				vertexSet[j][k]['pos'] = (pos1+dl, pos2+dl)

		words = []
		word_ids = []
		for sent in ori_data[i]['sents']:
			words += sent

		global word2id

		for t, word in enumerate(words):
			word = word.lower()
			if t < max_length:
				if word in word2id:
					word_ids.append(word2id[word])
				else:
					word_ids.append(word2id['UNK'])

		for t in range(t + 1, max_length):
			word_ids.append(word2id['BLANK'])

		ori_data[i]['vertexSet'] = vertexSet  #加入位置偏移后的实体提及

		item = {}

		item['word_ids'] = word_ids
		item['vertexSet'] = vertexSet
		labels = ori_data[i].get('labels', [])  #获取一个实例中的所有关系

		train_triple = set([])
		new_labels = []
		for label in labels:
			rel = label['r']
			assert(rel in rel2id)

			label['r'] = rel2id[label['r']]  #将关系字符串转换成关系id

			train_triple.add((label['h'], label['t']))  #添加关系的首尾实体索引


			if suffix=='_train':
				for n1 in vertexSet[label['h']]:
					for n2 in vertexSet[label['t']]:
						fact_in_dev_train.add((n1['name'], n2['name'], rel))


			if is_training:
				for n1 in vertexSet[label['h']]:
					for n2 in vertexSet[label['t']]:
						fact_in_train.add((n1['name'], n2['name'], rel))

			else:
				# fix a bug here
				label['intrain'] = False
				label['indev_train'] = False

				for n1 in vertexSet[label['h']]:
					for n2 in vertexSet[label['t']]:
						if (n1['name'], n2['name'], rel) in fact_in_train:
							label['intrain'] = True

						if suffix == '_dev' or suffix == '_test':
							if (n1['name'], n2['name'], rel) in fact_in_dev_train:
								label['indev_train'] = True


			new_labels.append(label)

		item['labels'] = new_labels
		item['title'] = ori_data[i]['title']

		na_triple = []
		for j in range(len(vertexSet)):
			for k in range(len(vertexSet)):
				if (j != k):
					if (j, k) not in train_triple:
						na_triple.append((j, k))

		item['na_triple'] = na_triple
		item['Ls'] = Ls
		item['sents'] = ori_data[i]['sents']

		all_triples = []
		for label in labels:
			all_triples.append(label)

		for triple in na_triple:
			label = {}
			label['h'] = triple[0]
			label['t'] = triple[1]
			label['r'] = 0
			all_triples.append(label)

		item['all_triples'] = all_triples

		data.append(item)

	print ('data_len:', len(ori_data))
	# print ('Ma_V', Ma)
	# print ('Ma_e', Ma_e)
	# print (suffix)
	# print ('fact_in_train', len(fact_in_train))
	# print (intrain, notintrain)
	# print ('fact_in_devtrain', len(fact_in_dev_train))
	# print (indevtrain, notindevtrain)


	# saving
	print("Saving files")
	if is_training:
		name_prefix = "train"
	else:
		name_prefix = "dev"

	json.dump(data , open(os.path.join(out_path, name_prefix + suffix + '.json'), "w"))

	return



# init(train_distant_file_name, rel2id, max_length = 512, is_training = True, suffix='')
init(train_annotated_file_name, rel2id, max_length = 512, is_training = False, suffix='_train')
init(dev_file_name, rel2id, max_length = 512, is_training = False, suffix='_dev')
init(test_file_name, rel2id, max_length = 512, is_training = False, suffix='_test')


