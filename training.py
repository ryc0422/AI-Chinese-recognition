import os, sys, pickle, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import VGG16
from preprocess_data import MyDataset, make_dict, load_file



def train(train_loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_correct = 0

    for i, data in enumerate(tqdm(train_loader)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item() * data.size(0)
        preds = torch.max(output.data)
        running_correct += (preds == target).sum().item() 
        loss.backward()
        optimizer.step()
    
    train_loss = running_loss/len(train_loader.dataset)
    train_accuracy = 100 * running_correct / len(train_loader.dataset)
    
    return train_loss, train_accuracy


def eval(test_loader, model):
    model.eval()
    running_loss = 0.0
    running_correct = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data, target = data[0].to(device), data[1].to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size()
            preds = torch.max(output.data)
            running_correct += (preds == target).sum().item() 
    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = 100 * running_correct / len(test_loader.dataset)
    
    return eval_loss, eval_accuracy


if __name__ == "__main__":
    
    """ Hyper Parameters """
    N_CLASS = 801
    TRAIN_BATCH_SIZE = 512
    VALID_BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 2    
    
    """ GPU or CPU """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Load model """
    model = VGG16(N_CLASS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    )

    """ Generate Data """
    word_dict = make_dict()
    dataset_path = './train'
    data_x, data_y = load_file(dataset_path, word_dict) # x:pic; y: label   

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42, shuffle=True)
    trainset = MyDataset(x_train, y_train, transform)
    testset = MyDataset(x_test, y_test, transform)
    train_loader = DataLoader(dataset = trainset, batch_size = TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset = testset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    """ Start training """
    Best_acc = 0
    recorder = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        Train_loss, Train_acc = train(train_loader, model, criterion, optimizer)
        Test_loss, Test_acc = validate(test_loader, model)
        epoch_time = time.time()-start_time
        recoder.append([epoch, Train_loss, Train_acc, Test_loss, Test_acc])
        print(f'Epoch{epoch}: Train_loss={Test_loss:.4f}, Train_acc={Train_acc:.4f}; Test_loss={Test_loss:.4f}, Test_acc={Train_acc:.4f}')

        """ Save model """
        if Train_acc > Best_acc:
            Best_acc = Train_acc
            torch.save(model.state_dict(), 'Model/vgg16.pt')

        with open('Model/vgg16.record', 'wb') as f:
            pickle.dump(np.array(recoder), f)
