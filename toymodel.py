import torch
import numpy as np
from os import listdir
from os.path import isfile, join
import os
from mydataloader import KNDataset
from sklearn.model_selection import train_test_split
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import matplotlib.pyplot as plt

'''
#This sets up the dataloader..ignore if you've done it already
preprocess_data = False
if preprocess_data:
    mypath = './pt_files/'
    num_files = len(os.listdir(mypath))
    params = np.zeros((num_files,7))
    for i,f in enumerate(os.listdir(mypath)):
        if not f.endswith('.pt'):
            continue
        params[i,:] = np.asarray(f.split('kn')[1].split('pt')[0].split('-'),dtype=float)
        if i%5000==0:
           print(i,num_files)
    np.savez('params.npz',params=params,nfiles=num_files)
'''

# Parameters
params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 1}
max_epochs = 1
do_training = True

# Datasets
data = np.load('params.npz')
all_param = data['params'][0:1000]
N = data['nfiles']
N = len(all_param)
#ind_train, ind_test = train_test_split(np.arange(N,dtype=int), test_size=0.01, random_state=42)
ind_train = np.arange(N,dtype=int)

# Generators
training_data = KNDataset(ind_train, all_param[ind_train])

training_generator = torch.utils.data.DataLoader(training_data, batch_size=256, shuffle=True)
'''
validation_set = KNDataset(partition['validation'], labels)
validation_generator = torch.utils.data.DataLoader(validation_set)
'''


#Model
class MLP(torch.nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.act = torch.nn.ReLU()
        self.hidden1 = torch.nn.Linear(n_inputs, 100)
        self.hidden2 = torch.nn.Linear(100, 500)
        self.hidden3 = torch.nn.Linear(500, 1024)
        self.hidden4 = torch.nn.Linear(1024, 1024)
        self.hidden5 = torch.nn.Linear(1024, 1025)

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act(X)
        X = self.hidden2(X)
        X = self.act(X)
        X = self.hidden3(X)
        X = self.act(X)
        X = self.hidden4(X)
        X = self.act(X)
        X = self.hidden5(X)
        X = self.act(X)
        return X
 

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        print(loss)


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc
 
# make a class prediction for one row of data
def predict(indx, model):
    # make prediction
    yhat = model(torch.tensor(all_param[indx]).float())
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# define the network
model = MLP(7)
# train the model
if do_training:
    train_model(training_generator, model)
    torch.save(model.state_dict(), './mymodel.pt')
else:
    model.load_state_dict(torch.load('./mymodel.pt'))

# evaluate the model
yhat = predict(500, model)
print(np.mean(yhat),np.min(yhat),np.max(yhat))
plt.plot(yhat,color='black')
param = all_param[500]
mypath = '/gpfs/group/vav5084/default/ashley/knemulator/pt_files/'
y = torch.load(mypath + 'kn' + str(param[0])+'-'+str(param[1])+'-'+str(param[2])+'-'+str(param[3])+'-'+str(param[4])+'-'+str(int(param[5]))+'-'+str(int(param[6]))+'.pt').float()
plt.plot(y/torch.mean(y),color='red')
plt.savefig('test.png')
plt.clf()
plt.plot(yhat[:-1]*10.**yhat[-1],color='black')
print(yhat[-1],torch.mean(y))
param = all_param[500]
mypath = '/gpfs/group/vav5084/default/ashley/knemulator/pt_files/'
y = torch.load(mypath + 'kn' + str(param[0])+'-'+str(param[1])+'-'+str(param[2])+'-'+str(param[3])+'-'+str(param[4])+'-'+str(int(param[5]))+'-'+str(int(param[6]))+'.pt').float()
plt.plot(y,color='red')
plt.savefig('test2.png')
print(yhat)
