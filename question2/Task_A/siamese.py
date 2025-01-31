import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import os
import random
import pandas as pd
from tqdm import tqdm
import argparse
import logging

BATCH_SIZE = 32
MARGIN = 1.0

class SiameseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.image_paths = self.get_image_paths()

        self.pairs = self._create_pairs()

    def get_image_paths(self):
        image_paths = {}

        for cls in self.classes:
            cls_dir = os.path.join(self.data_dir, cls)
            image_paths[cls] = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir)]

        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def _create_pairs(self):
        pairs = []
        
        for img_cls in self.classes:
            images_class = self.image_paths[img_cls]
            for _ in range(len(images_class)//2):
                img1, img2 = random.sample(images_class, 2)
                pairs.append((img1, img2, 1))

            other_classes = [cls for cls in self.classes if cls != img_cls]
            for j in range(len(images_class)//2):
                img1 = random.choice(images_class)
                img2 = random.choice(self.image_paths[random.choice(other_classes)])
                pairs.append((img1, img2, 0))

        return pairs
    
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # print(img1.shape)
        return img1, img2, label


class SiameseNet(nn.Module):
    def __init__(self, cnn_backbone):
        super(SiameseNet, self).__init__()
        self.cnn_backbone = cnn_backbone

    def forward(self, x1, x2):
        # print(x1.shape)
        x1 = self.cnn_backbone(x1)
        x2 = self.cnn_backbone(x2)
        return x1, x2

    def get_embedding(self, x):
        return self.cnn_backbone(x)
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 0.5))

        return loss_contrastive
    

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        input1, input2, labels = data
        input1 = input1.to(device)
        input2 = input2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def test_epoch(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            input1, input2, labels = data
            input1 = input1.to(device)
            input2 = input2.to(device)
            labels = labels.to(device)

            output1, output2 = model(input1, input2)
            loss = criterion(output1, output2, labels)

            running_loss += loss.item()

    return running_loss / len(test_loader)


def train(model, train_loader, test_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = test_epoch(model, test_loader, criterion, device)

        # print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}')
        logging.info(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}')

    # print('Finished Training')


def get_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    dataset = SiameseDataset(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def save_embeddings(model, device):
    df_dict = {'embedding': [], 'label': []}

    all_images = SiameseDataset(f'train_data/spectrograms/{window_name}', transform=None).image_paths

    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    for cls in all_images:
        # print(cls)
        logging.info(f'Getting embeddings for class: {cls}')
        for img_pth in all_images[cls]:
            # print(img_pth)
            img = Image.open(img_pth).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            embedding = model.get_embedding(img)
            df_dict['embedding'].append(embedding.cpu().detach().numpy().flatten())
            df_dict['label'].append(cls)

    # distribute embedding list elements to columns
    for i in range(len(df_dict['embedding'][0])):
        df_dict[f'embedding_{i}'] = [emb[i] for emb in df_dict['embedding']]

    for i in range(len(df_dict['embedding'][0])):
        df_dict[f'embedding_{i}'] = (df_dict[f'embedding_{i}'] - min(df_dict[f'embedding_{i}'])) / (max(df_dict[f'embedding_{i}']) - min(df_dict[f'embedding_{i}']))

    del df_dict['embedding']

    df = pd.DataFrame(df_dict)

    if not os.path.exists('train_data/embeddings/'):
        os.mkdir('train_data/embeddings')
    df.to_csv(f'train_data/embeddings/{window_name}_embeddings.csv', index=False)


def main(window_name, epochs):
    logging.info(f"Initalizing training for window: {window_name}")

    data_dir = 'train_data/spectrograms/rectangular'
    batch_size = BATCH_SIZE
    margin = MARGIN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    logging.info('Getting data loaders')
    train_loader, test_loader = get_data_loaders(data_dir, batch_size)
    logging.info('Data loaders ready')

    resnet_backbone = resnet18(weights=None)
    cnn_backbone = nn.Sequential(*list(resnet_backbone.children())[:-1])
    model = SiameseNet(cnn_backbone).to(device)
    criterion = ContrastiveLoss(margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logging.info('Starting training')
    train(model, train_loader, test_loader, criterion, optimizer, device, epochs)
    logging.info('Training complete')

    logging.info('Saving embeddings')
    save_embeddings(model, device)
    logging.info('Embeddings saved')

    logging.info('Saving model')
    if not os.path.exists('models/siamese/'):
        os.makedirs('models/siamese')
    torch.save(model.state_dict(), f'models/siamese/model_{window_name}.pth')
    logging.info('Done')
    logging.info('-----------------------------------\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Siamese network')
    parser.add_argument('--window', type=str, default='hann', help='Window type')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    args = parser.parse_args()


    window_name = args.window
    epochs = args.epochs

    if not os.path.exists('logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename=f'logs/siamese_{window_name}.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filemode='w')

    if window_name not in ['hann', 'hamming', 'rectangular', 'all']:
        logging.error('Invalid window type')
        raise ValueError('Invalid window type')
    
    if window_name == 'all':
        for window_name in ['hann', 'hamming', 'rectangular']:
            main(window_name, epochs)
    else:
        main(window_name, epochs)