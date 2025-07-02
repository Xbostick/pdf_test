"""
Module for training and predicting with ML models (LogReg, Linear, CNN)
on the 20 Newsgroups dataset.
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

import numpy as np

from IPython.display import clear_output
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from labels_newsground import labels_newsground

#models
class Linear_model(nn.Module):
    """A simple feedforward neural network for text classification."""
    def __init__(self, num_labels: int):
        super(Linear_model, self).__init__()
        # Assert to ensure valid number of labels
        assert num_labels > 0, "Number of labels must be positive"
        self.in_lin = nn.Linear(500, 100)
        self.in_lin2 = nn.Linear(100,50)
        self.out_lin = nn.Linear(50, num_labels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.sm = nn.Softmax()
        self.norm = nn.BatchNorm1d(100)
        
    
    def forward(self,x):
        x = self.norm(self.act(self.in_lin(x)))
        x = self.drop(x)
        x = self.act(self.in_lin2(x))
        x = self.drop(x)
        x = self.act(self.out_lin(x))
        return self.sm(x)
    
class CNN_model(nn.Module):
    """A convolutional neural network for text classification."""
    def __init__(self, num_labels):
        super(CNN_model, self).__init__()
        assert num_labels > 0, "Number of labels must be positive"
        self.in_lin = nn.Linear(500, 300)
        self.conv1 = nn.Conv1d(1,32,3, padding = 'same')
        self.conv2 = nn.Conv1d(1,32,4, padding = 'same')
        self.conv3 = nn.Conv1d(1,32,2, padding = 'same')
        self.pool = nn.MaxPool1d(3)
        self.dense = nn.Linear(9600, 1000)
        self.in_lin2 = nn.Linear(2000,500)
        self.out_lin = nn.Linear(1000, num_labels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.sm = nn.Softmax()
        self.norm = nn.BatchNorm1d(9600)
    
    def forward(self,x):
        x = self.act(self.in_lin(x))
        x1 = self.pool(self.act(self.conv1(x)))
        x2 = self.pool(self.act(self.conv2(x)))
        x3 = self.pool(self.act(self.conv3(x)))
        x = torch.flatten(torch.cat([x1,x2,x3], dim = 1),1)
        x = self.drop(self.norm(x))
        x = self.act(self.dense(x))
        x = self.act(self.out_lin(x))
        return self.sm(x)
    

def dl_data_preprocess( x_train, x_test,y_train,y_test):
    """
    Preprocess data for deep learning models by converting to tensors and one-hot encoding labels.
    
    Args:
        x_train: Training features.
        x_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
    
    Returns:
        Tuple of preprocessed (x_train, x_test, y_train, y_test).
    """
    # Assert input data is not empty
    assert x_train.shape[0] > 0 and x_test.shape[0] > 0, "Input data cannot be empty"
    assert y_train.shape[0] > 0 and y_test.shape[0] > 0, "Labels cannot be empty"
    
    x_train = torch.Tensor(x_train)
    x_test = torch.Tensor(x_test)
    
    enc = OneHotEncoder()
    y_train = enc.fit_transform(y_train.reshape(-1, 1)).toarray()
    return x_train, x_test,y_train,y_test

def dl_model_training(model, x_train, x_test,y_train,y_test, 
                      loss_func = nn.CrossEntropyLoss(), 
                      num_epoch = 10,
                      CNN = False
                      ):
    """
    Train a deep learning model and plot training loss.
    
    Args:
        model: PyTorch model to train.
        x_train: Training features.
        x_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        loss_func: Loss function (default: CrossEntropyLoss).
        num_epoch: Number of training epochs (default: 10).
        cnn: Whether the model is a CNN (default: False).
    
    Returns:
        Trained model
    """
    assert x_train.shape[0] == y_train.shape[0], "Training data and labels must have the same length"

    optimizer = torch.optim.Adam(params= model.parameters(),lr = 1e-4)
    loss_storage = []
    model = model.cuda()
    model.train()

    for epoch in tqdm(range(num_epoch), total = num_epoch, desc="Training Epochs"):
        epoch_loss = 0
        for x,y in zip(DataLoader(x_train,batch_size=50),DataLoader(torch.Tensor(y_train), batch_size=50)):
            x, y = x.cuda(), y.cuda()
            pred = model(x.unsqueeze(1) if CNN else x)
            loss = loss_func(pred,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.cpu().detach().numpy()

        loss_storage.append(epoch_loss)
        if epoch % 5 == 0 or epoch == num_epoch - 1:
            clear_output(wait=True)
            plt.plot(loss_storage)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.show()
            with torch.no_grad():
                model_pred = model(x_test.cuda().unsqueeze(1) if CNN else x_test.cuda())
                test_score = accuracy_score(y_test, torch.argmax(model_pred, dim=1).cpu().numpy()) * 100
                print(f"Test accuracy score: {test_score:.2f}%")
    return model
    

def train( x_train,x_test,y_train,y_test, model_type = "LogReg", save = "./models/", epoch = 10):
    """
    Train a specified model type and save it.
    
    Args:
        x_train: Training features.
        x_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        model_type: Model type ('LogReg', 'Linear', 'CNN') (default: 'LogReg').
        save: Directory to save the model (default: './models/').
        epoch: Number of epochs for deep learning models (default: 10).
    
    Returns:
        Trained model
    """

    # Assert valid model type
    valid_models = ["LogReg", "Linear", "CNN"]
    assert model_type in valid_models, f"Model type must be one of {valid_models}"

    if model_type =='LogReg':
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)
        lr_pred = model.predict(x_test)
        test_score = accuracy_score(y_test, lr_pred) * 100
        print(f"Test accuracy score: {test_score:.2f}%")
        if save:
            joblib.dump(model, save + "model_LogReg.pkl")

    elif model_type == 'Linear':
        model = Linear_model(len(np.unique_values(y_test)))
        x_train,x_test,y_train,y_test = dl_data_preprocess(x_train,x_test,y_train,y_test)
        model = dl_model_training(model, x_train,x_test,y_train,y_test, num_epoch=epoch)
        if save:
            torch.save(model.state_dict(), save + "model_Linear.pkl")

    elif model_type == 'CNN':
        model = CNN_model(len(np.unique_values(y_test)))
        x_train,x_test,y_train,y_test = dl_data_preprocess(x_train,x_test,y_train,y_test)
        model = dl_model_training(model, x_train,x_test,y_train,y_test, num_epoch=epoch, CNN=True)
        if save:
            torch.save(model.state_dict(), save + "model_CNN.pkl")

    print("\nTrain and saving done\n")
    return model

def predict(tokens, model_type = "Linear", save = "./models/"):
    """
    Predict the class of input tokens using the specified model.
    
    Args:
        tokens: Input tokens for prediction.
        model_type: Model type ('LogReg', 'Linear', 'CNN') (default: 'Linear').
        save: Directory where the model is saved (default: './models/').
    
    Returns:
        Predicted class index or indices.
    """
    valid_models = ["LogReg", "Linear", "CNN"]
    assert model_type in valid_models, f"Model type must be one of {valid_models}"
    assert len(tokens) > 0, "Input tokens cannot be empty"
    
    if model_type =='LogReg':
        model = joblib.load(save + "model_LogReg.pkl")
        pred = model.predict(tokens)
    else:
        tokens = torch.tensor(tokens, dtype=torch.float32)
        num_labels = len(labels_newsground)
        model = CNN_model(num_labels) if model_type == "CNN" else Linear_model(num_labels)
        model.load_state_dict(torch.load(f"{save}model_{model_type}.pkl", weights_only=True))
        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(tokens.unsqueeze(1) if model_type == "CNN" else tokens), dim=1).cpu().numpy()
        
    return pred
