import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from PIL import Image
import os
from tqdm import tqdm
import logging 

BATCH_SIZE=128

def get_dataloaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    
    # split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        
        resnet_backbone = resnet18(weights=None)
        self.cnn_backbone = nn.Sequential(*list(resnet_backbone.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.cnn_backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for i, data in tqdm(enumerate(train_loader, 0)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += (outputs.argmax(dim=1) == labels).float().mean().item()

    return running_loss / len(train_loader), running_corrects / len(train_loader)


def test_epoch(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_corrects += (outputs.argmax(dim=1) == labels).float().mean().item()

    return running_loss / len(test_loader), running_corrects / len(test_loader)


def train(model, train_loader, test_loader, criterion, optimizer, device, epochs):
    logging.info('Starting training')
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        print(f'Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')
        logging.info(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        logging.info(f'Train Accuracy: {train_acc}, Test Accuracy: {test_acc}\n')
    logging.info('Training finished')

def main(window_type, epochs):
    logging.info(f'Training CNN on spectrograms for window: {window_type}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    logging.info(f'Using device: {device}')

    logging.info(f'Loading data from train_data/spectrograms/{window_type}')
    train_loader, test_loader = get_dataloaders(f'train_data/spectrograms/{window_type}', BATCH_SIZE)
    logging.info('Data loaded successfully')

    model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, test_loader, criterion, optimizer, device, epochs)

    logging.info('Saving model')
    if not os.path.exists('models/cnn'):
        os.makedirs('models/cnn')
    torch.save(model.state_dict(), f'models/cnn/{window_type}.pth')
    logging.info('Model saved')


# if __name__ == '__main__':
#     data_dir = 'train_data/spectrograms/hann'
#     batch_size = 128
#     epochs = 10
#     main(data_dir, batch_size, epochs)