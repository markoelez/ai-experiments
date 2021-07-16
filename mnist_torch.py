#!/usr/bin/env python3
import os
import gzip
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import matplotlib.pyplot as plt
import numpy as np
from torch import tensor, optim
from tqdm import tqdm
from torchvision import transforms


# data starts at offset 16
x_train_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
x_test_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'

# data starts at index 8
y_train_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
y_test_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

def process(url):
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    fp = md5 = 'tmp/' + hashlib.md5(url.encode()).hexdigest()
    if os.path.isfile(fp):
        with open(fp, 'rb') as f:
            dat = f.read()
    else:
        with open(fp, 'wb') as f:
            res = requests.get(url)
            res.raise_for_status()
            dat = res.content
            f.write(dat)
    assert b'503' not in dat, 'request failed'
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

x_train = process(x_train_url)[16:].reshape(-1, 28*28)
y_train = process(y_train_url)[8:]

x_test = process(x_test_url)[16:].reshape(-1, 28*28)
y_test = process(y_test_url)[8:]

'''
i = 10
img = x_train[i]
print(y_train[i])
plt.imshow(img)
plt.show()
'''

# setup
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x

model = Net()

# train
epochs = 500
batch_size = 128
loss_func = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
losses, accs = [], []
for i in (t := tqdm(range(epochs))):
    idx = np.random.randint(0, x_train.shape[0], size=(batch_size))
    x = tensor(x_train[idx]).float()
    y = tensor(y_train[idx]).long()
    optimizer.zero_grad()
    out = model(x)
    loss = loss_func(out, y)
    loss.backward()
    optimizer.step()

    acc = torch.div(torch.sum(torch.argmax(out, dim=1) == y), batch_size)

    losses.append(loss.detach().numpy())
    accs.append(acc.detach().numpy())
    t.set_description(f'Loss: {loss:.2f} Accuracy: {acc:.2f}')

print('finished training')

plt.ylim(-0.1, 1.1)
plt.plot(losses, label='loss')
plt.plot(accs, label='accuracy')
plt.legend(loc='upper left')
plt.show()

# eval
correct = 0
for i in range(x_test.shape[0]):
    x = tensor(x_test[i]).float()
    y = tensor(y_test[i]).long()
    out = model(x)
    correct += (torch.argmax(out) == y)

print(f'finished testing, got accuracy: {correct/x_test.shape[0]:.2f}')
