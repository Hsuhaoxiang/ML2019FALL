import csv
import spacy
import pandas as pd
from gensim.models import word2vec
import csv
import spacy
import pandas as pd
from gensim.models import word2vec
import os
import csv
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

TRAIN_X_PATH = sys.argv[1]
TRAIN_Y_PATH = sys.argv[2]
TEST_X_PATH = sys.argv[3]

train_data = pd.read_csv(TRAIN_X_PATH)
train_x = train_data['comment'].values
train_y = pd.read_csv(TRAIN_Y_PATH)['label'].values
test_data = pd.read_csv(TEST_X_PATH)
test_x = test_data['comment'].values


nlp = spacy.load("en_core_web_sm")
useless_word = ['.@user', '@user', '..', '...', '@', '*', '#', '&', 'URL', ' ', '  ', '   ']

    
seg_train_x = []  # should be a list of lists!!!
for row in train_x:
    doc = nlp(row)
    inner_list = []
    for token in doc:
        if token.text in useless_word:
            continue
        inner_list.append(token.text)

    seg_train_x.append(inner_list)

print(len(seg_train_x))  # 13240

seg_test_x = []  # should be a list of lists!!!
for row in test_x:
    doc = nlp(row)
    inner_list = []
    for token in doc:
        if token.text in useless_word:
            continue
        inner_list.append(token.text)

    seg_test_x.append(inner_list)

print(len(seg_test_x))  # 13240

data = seg_train_x + seg_test_x

# generate a corpus.csv

with open('corpus.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for row in data:
        writer.writerow(row)



torch.manual_seed(3344)
TRAIN_Y_PATH = "./train_y.csv"  #'data/train_y.csv'
train_y = pd.read_csv(TRAIN_Y_PATH)['label'].values


# corpus.csv is formed by tokenzied train_x & test_x
sentences = word2vec.LineSentence('corpus.csv')
train_data, test_data = [], []
for i, data in enumerate(sentences):
    if len(train_data) < 13240:
        train_data.append(data)
    else:
        test_data.append(data)


# word_to_ix maps each word in the vocab to a unique integer, which will be its
# `index` into the Bag of words vector
word_to_ix = {}
for sent in train_data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2
BATCH_SIZE= 32
NUM_EPOCH = 13

# word_to_ix should've already built
def text2index(corpus, to_ix): # corpus: 語料庫 should be a list of lists
    new_corpus = []
    for seq in corpus:
        vec = np.zeros(len(to_ix))
        for word in seq:
            vec[word_to_ix[word]] += 1
        new_corpus.append(vec)
        
    return np.array(new_corpus, dtype=np.float32)

train_data = text2index(train_data, word_to_ix)
test_data = text2index(test_data, word_to_ix)


train_data = [(train_data[i], train_y[i]) for i in range(len(train_y))]

# train valid split
train_set = train_data[len(train_data)//20:]
valid_set = train_data[:len(train_data)//20]
print('len(train_set):', len(train_set), 'len(valid_set):', len(valid_set))

# grab a list & show what's look like
# for sentence, label in train_set:
#     print(sentence, label)
#     break
    

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# for seq, label in train_loader:
#     print(seq, label.size())
#     break
    
class BoWClassifier(nn.Module):
    
    def __init__(self, num_labels, vocab_size):
        
        super(BoWClassifier, self).__init__()

        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        
        return F.log_softmax(self.linear(bow_vec), dim=1)



model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)
print(model)
if cuda:
    model.cuda()

with torch.no_grad():
    for instance, label in valid_loader:
        if cuda:
            instance = instance.cuda()
        log_probs = model(instance)
#         print(log_probs)
        break

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_acc = 0.0
train_acc_history, train_loss_history = [], []
val_acc_history, val_loss_history = [], []

for epoch in range(NUM_EPOCH):
    epoch_start_time = time.time()
    train_acc, train_loss = 0.0, 0.0
    val_acc, val_loss = 0.0, 0.0
    
    model.train()

    for i, (instance, label) in enumerate(train_loader):

        model.zero_grad()
        
        if cuda:
            instance = instance.cuda()
            label = torch.LongTensor(label).cuda()
            
        log_probs = model(instance)

        loss = loss_function(log_probs, label)
        loss.backward()
        optimizer.step()
        
        train_acc += np.sum(np.argmax(log_probs.cpu().data.numpy(), axis=1) == label.cpu().numpy())
        train_loss += loss.item()
        
        progress = ('#' * int(float(i)/len(train_loader)*40)).ljust(40)
        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, NUM_EPOCH, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)
        
    train_acc = train_acc/len(train_set)
    train_acc_history.append(train_acc)
    train_loss_history.append(train_loss)
    
    model.eval()
    
    for i, (instance, label) in enumerate(valid_loader):

        model.zero_grad()
        
        if cuda:
            instance = instance.cuda()
            label = torch.LongTensor(label).cuda()
            
        log_probs = model(instance)
        
        val_acc += np.sum(np.argmax(log_probs.cpu().data.numpy(), axis=1) == label.cpu().numpy())
        val_loss += loss.item()
        
        progress = ('#' * int(float(i)/len(valid_loader)*40)).ljust(40)
        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, NUM_EPOCH, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)
        
    val_acc = val_acc/len(valid_set)
    val_acc_history.append(val_acc)
    val_loss_history.append(val_loss)

    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % \
            (epoch + 1, NUM_EPOCH, time.time()-epoch_start_time, \
             train_acc, train_loss, val_acc, val_loss))
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
with open("result.csv","w") as f:  # 'results/predict.csv'
    w = csv.writer(f)
    title = ['id','label']
    w.writerow(title)
    for i in range(len(ans)):
        content = [i,ans[i]]
        w.writerow(content)

all_end_time = time.time()

print('time elapsed: {}'.format(time.strftime("%H:%M:%S", time.gmtime(all_end_time-all_start_time))))