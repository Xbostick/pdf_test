import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import joblib

#models
class Linear_model(nn.Module):
    def __init__(self, labels):
        super(Linear_model, self).__init__()

        self.in_lin = nn.Linear(49255, 10000)
        self.in_lin2 = nn.Linear(10000,2000)
        self.in_lin3 = nn.Linear(2000,400)
        self.out_lin = nn.Linear(400, labels)
        self.sig = nn.ReLU()
        self.drop = nn.Dropout(0.02)
        self.sm = nn.Softmax()
        self.norm = nn.BatchNorm1d(2000)
    
    def forward(self,x):
        x = self.sig(self.in_lin(x))
        # x = self.drop(x)
        x = self.norm(self.sig(self.in_lin2(x)))
        # x = self.drop(x)
        x = self.sig(self.in_lin3(x))
        # x = self.drop(x)
        x = self.sig(self.out_lin(x))
        return self.sm(x)
    
class CNN_model(nn.Module):
    def __init__(self, labels):
        super(CNN_model, self).__init__()

        self.in_lin = nn.Linear(49255, 5000)
        
        self.conv1 = nn.Conv1d(1,32,4, padding = 'same')
        self.conv2 = nn.Conv1d(32,32,4, padding = 'same')
        self.pool1 = nn.MaxPool1d(3)
        self.dense = nn.Linear(32 * 1666 ,5000)
        self.in_lin2 = nn.Linear(5000,1000)
        self.out_lin = nn.Linear(1000, labels)
        self.sig = nn.ReLU()
        self.drop = nn.Dropout(0.02)
        self.sm = nn.Softmax()
        self.norm = nn.BatchNorm1d(32 * 1666)
    
    def forward(self,x):
        x = self.sig(self.in_lin(x))
        #TODO dropout?
        # x = self.drop(x)
        x = self.sig(self.conv1(x))
        # x = self.drop(x)
        x = self.sig(self.conv2(x))
        x = self.pool1(x)
        x = torch.flatten(x,1)
        x = self.norm(x)
        x = self.dense(x)
        x = self.sig(self.in_lin2(x))
        x = self.sig(self.out_lin(x))
        return self.sm(x)
    

def dl_data_preprocess( x_train, x_test,y_train,y_test):
    x_train = torch.Tensor(x_train.toarray())
    x_test = torch.Tensor(x_test.toarray())
    
    enc = OneHotEncoder()
    y_train = enc.fit_transform(y_train.reshape(-1, 1))
    y_train = enc.fit(y_test.reshape(-1, 1))

    return x_train, x_test,y_train,y_test

def dl_model_training(model, x_train, x_test,y_train,y_test, 
                      loss_func = nn.CrossEntropyLoss(), 
                      num_epoch = 20
                      ):
    
    optimizer = torch.optim.Adam(params= model.parameters(),lr = 1e-5)
    loss_storage = []
    model = model.cuda()
    model.train()

    for epoch in range(num_epoch):
        epoch_loss = 0
        for x,y in zip(DataLoader(x_train,batch_size=500),DataLoader(torch.Tensor(y_train), batch_size=500)):
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            loss = loss_func(pred,y)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()

            epoch_loss+=loss.cpu().detach().numpy()
        loss_storage.append(epoch_loss)
        if epoch % 5 == 0:
            clear_output()
            plt.plot(loss_storage)
            plt.show() 
            with torch.no_grad():
                model_pred = model(x_test.cuda())
                test_score = accuracy_score(y_test, torch.argmax(model_pred, dim = 1).cpu().detach().numpy()) * 100
                print(f"Train accuracy score: {test_score:.2f}%")
    return model
    

def train( x_train,x_test,y_train,y_test, model_type = "LogReg", save = "./models/"):
    
    if model_type =='LogReg':
        model = LogisticRegression()
        model.fit(x_train.toarray(), y_train)

        lr_pred = model.predict(x_test.toarray())
        test_score = accuracy_score(y_test, lr_pred) * 100
        print(f"Test accuracy score: {test_score:.2f}%")
        if save:
            joblib.dump(model, save + "model_LogReg.pkl")

    elif model_type == 'Linear':
        model = Linear_model(len(np.unique_values(y_test)))
        x_train,x_test,y_train,y_test = dl_data_preprocess()
        model = dl_model_training(model, x_train,x_test,y_train,y_test)
        if save:
            torch.save(model.state_dict(), save + "model_Linear.pkl")

    elif model_type == 'CNN':
        model = CNN_model(len(np.unique_values(y_test)))
        x_train,x_test,y_train,y_test = dl_data_preprocess()
        model = dl_model_training(model, x_train,x_test,y_train,y_test)
        if save:
            torch.save(model.state_dict(), save + "model_CNN.pkl")

    print("\nTrain and saving done\n")
    return model

#TODO main?