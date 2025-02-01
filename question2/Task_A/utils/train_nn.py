import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import os

# load dataset from csv file
def load_data(data_path: str):
    data_csv = pd.read_csv(data_path)
    target = data_csv.pop('label').astype('category').cat.codes
    # converting the string to floats in data_csv
    data = data_csv.astype(np.float32)
    
    # converting target to one hot encoding
    target = pd.get_dummies(target)

    return data_csv, target


# Neural network model - 4 fully connected layers - ReLU activation for hidden layers and sigmoid for output layer
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
    

# fucntion to train model for an epoch
def train_nn_model(data_loader, model, criterion, optimizer, device):
    model.train()
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model


# fucntion to test model after an epoch
def test_nn_model(data_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            running_acc += (outputs.argmax(1) == labels.argmax(1)).float().mean().item()

    return running_loss / len(data_loader), 100 * running_acc / len(data_loader)


def main(window_type, epochs):
    logging.info(f'Training neural network on embeddings for window: {window_type}')

    logging.info(f'Loading data from train_data/embeddings/{window_type}_embeddings.csv')
    data, target = load_data(f'train_data/embeddings/{window_type}_embeddings.csv')
    data = torch.tensor(data.values).float()
    target = torch.tensor(target.values).float()
    dataset = TensorDataset(data, target)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    logging.info('Data loaded successfully')

    model = NeuralNetwork(data.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.0001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    logging.info('Training neural network')
    # training the model
    for epoch in range(epochs):
        model = train_nn_model(train_loader, model, criterion, optimizer, device)
        train_loss, train_acc = test_nn_model(train_loader, model, criterion, device)
        test_loss, test_acc = test_nn_model(test_loader, model, criterion, device)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')
        logging.info(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        logging.info(f'Train Acc: {train_acc}, Test Acc: {test_acc}\n')

    logging.info('Finished training neural network')

    logging.info('Saving model')
    if not os.path.exists('models/nn'):
        os.makedirs('models/nn')
    torch.save(model.state_dict(), f'models/nn/{window_type}.pth')
    logging.info('Model saved successfully\n')



# if __name__ == '__main__':
#     main()