

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import warnings
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning


LATENT_DIM = 2 

# Assuming the MR images are grayscale and we need to resize them for a consistent input size
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Load the dataset
data_path = 'C:\\Users\\25060\\Desktop\\keras_png_slices_data'
dataset = datasets.ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(64*64, 400)
        self.fc21 = nn.Linear(400, LATENT_DIM)  # mean
        self.fc22 = nn.Linear(400, LATENT_DIM)  # variance

        # Decoder
        self.fc3 = nn.Linear(LATENT_DIM, 400)
        self.fc4 = nn.Linear(400, 64*64)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 64*64))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

device = torch.device("cpu")

model = VAE().to(device)

import umap
import matplotlib.pyplot as plt

model.eval()
all_latent = []

with torch.no_grad():
    for data, _ in dataloader:
        data = data.to(device)
        mu, _ = model.encode(data.view(-1, 64*64))
        all_latent.append(mu.cpu().numpy())

all_latent = np.vstack(all_latent)
embedding = umap.UMAP().fit_transform(all_latent)

plt.scatter(embedding[:, 0], embedding[:, 1], cmap='viridis')
plt.colorbar()
plt.title("VAE Latent Space Visualization with UMAP")
plt.show()
