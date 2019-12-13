import csv
import spacy
import pandas as pd
from gensim.models import word2vec
import os
import csv
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle

TEST_X_PATH = sys.argv[1]
test_data = pd.read_csv(TEST_X_PATH)
test_x = test_data['comment'].values
nlp = spacy.load("en_core_web_sm")
useless_word = ['.@user', '@user', '..', '...', '@', '*', '#', '&', 'URL', ' ', '  ', '   ']
def loadpkl():
    with open('word_to_ix.pkl', 'rb') as f:
        return pickle.load(f)
word_to_ix =loadpkl()


test_data = []  # should be a list of lists!!!
for row in test_x:
    doc = nlp(row)
    inner_list = []
    for token in doc:
        if token.text in useless_word:
            continue
        inner_list.append(token.text)
    test_data.append(inner_list)
    
with open('corpus.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for row in test_data :
        writer.writerow(row)
sentences = word2vec.LineSentence('corpus.csv')
test_data = []
for i, data in enumerate(sentences):
    test_data.append(data)

cuda = True if torch.cuda.is_available() else False

# sentences = word2vec.LineSentence(seg_test_x)

# test_data = []
# for i, data in enumerate(sentences):
#         test_data.append(data)




def text2index(corpus, to_ix): # corpus: 語料庫 should be a list of lists
    new_corpus = []
    for seq in corpus:
        vec = np.zeros(len(to_ix))
        for word in seq:
            vec[word_to_ix[word]] += 1
        new_corpus.append(vec)
        
    return np.array(new_corpus, dtype=np.float32)
test_data = text2index(test_data, word_to_ix)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False, num_workers=4)
    
class BoWClassifier(nn.Module):
    
    def __init__(self, num_labels, vocab_size):
        
        super(BoWClassifier, self).__init__()

        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        
        return F.log_softmax(self.linear(bow_vec), dim=1)




model =torch.load("bow_dnn_best.pkl",map_location = "cpu")


ans = []
model.eval()
with torch.no_grad():
    for instance in test_loader:
        if cuda:
            instance = instance.cuda()
        log_probs = model(instance)
        _, predicted = torch.max(log_probs.data, 1)
    
        predicted = predicted.cpu().numpy() 

        ans.extend(predicted)

### Write file
with open(sys.argv[2],"w") as f:  # 'results/predict.csv'
    w = csv.writer(f)
    title = ['id','label']
    w.writerow(title)
    for i in range(len(ans)):
        content = [i,ans[i]]
        w.writerow(content)