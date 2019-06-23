%matplotlib inline
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Initialize Batch Size
batch_size = 64

# number of subprocesses to use for data loading
num_workers = 0

transform = transforms.ToTensor()

# Load Dataset
train_dataset = datasets.MNIST(root='data', train=True, download =True, transform=transform)

# Prepare the dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers )

import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

  def __init__(self, input_size, hidden_dim, output_size):
    super(Discriminator, self).__init__()

    # Layers
    self.conv1 = nn.Linear(input_size, hidden_dim*4)    # Input Layer
    self.conv2 = nn.Linear(hidden_dim*4, hidden_dim*2)
    self.conv3 = nn.Linear(hidden_dim*2, hidden_dim)
    self.conv4 = nn.Linear(hidden_dim, output_size)  # Final Layer

    # Dropout layer
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    # Flatten image
    x = x.view(-1, 28*28)

    x = F.leaky_relu(self.conv1(x), 0.2)   # (input, negative_slope)
    x = self.dropout(x)
    x = F.leaky_relu(self.conv2(x), 0.2)
    x = self.dropout(x)
    x = F.leaky_relu(self.conv3(x), 0.2)
    x = self.dropout(x)

    # Final Layer
    out = self.conv4(x)
    return out


class Generator(nn.Module):

  def __init__(self, input_size, hidden_dim, output_size):
    super(Generator, self).__init__()

    # Layers
    self.conv1 = nn.Linear(input_size, hidden_dim)    # Input Layer
    self.conv2 = nn.Linear(hidden_dim, hidden_dim*2)
    self.conv3 = nn.Linear(hidden_dim*2, hidden_dim*4)
    self.conv4 = nn.Linear(hidden_dim*4, output_size)  # Final Layer

    # Dropout layer
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x), 0.2)   # (input, negative_slope)
    x = self.dropout(x)
    x = F.leaky_relu(self.conv2(x), 0.2)
    x = self.dropout(x)
    x = F.leaky_relu(self.conv3(x), 0.2)
    x = self.dropout(x)

    # Final Layer
    output = F.tanh(self.conv4(x))

    return output


# Discriminator Hyperarameters

# Size of input image
input_size = 784
# Size of discriminator output
d_output_size = 1
# Size of last hidden layer in discriminator (layer 3)
d_hidden_size = 32


# Generator Hyperparamters

# Size of latent vector to give to Generator (reffered as noise)
z_size = 100
# Size of discriminator output (generated image)
g_output_size = 784
# Size of first hidden layer
g_hidden_size = 32


D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)

print(D)
print()
print(G)


# Calculating losses
def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1

    # numerically stable loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


import torch.optim as optim

# Learning Rate
lr = 0.002

# Optimizers
d_optimizer = optim.Adam(D.parameters(), lr)
g_optimizer = optim.Adam(G.parameters(), lr)



import pickle as pkl

# training hyperparams
num_epochs = 100

# keep track of loss and generated, "fake" samples
samples = []
losses = []

print_every = 400

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

# train the network
D.train()
G.train()
for epoch in range(num_epochs):
    
    for batch_i, (real_images, _) in enumerate(train_loader):
        
        batch_size = real_images.size(0)
        
        ## Important rescaling step ##
        real_images = real_images*2 - 1  # rescale input images from [0,1) to [-1, 1)
        
        #  TRAIN THE DISCRIMINATOR
        d_optimizer.zero_grad()
        
        # 1. Train with real images
        
        # Compute the discriminator losses on real images
        # smooth the real labels
        D_real = D(real_images)
        d_real_loss = real_loss(D_real, smooth=True)
        
        # 2. Train with fake images
        
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)
        
        # Compute the discriminator losses on fake images
        D_fake = D(fake_images)
        d_fake_loss = fake_loss(D_fake)
        
        # add up loss and perform backprop
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        
        
        #  TRAIN THE GENERATOR
        g_optimizer.zero_grad()
        
        # 1. Train with fake images and flipped labels
        
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)
        
        # Compute the discriminator losses on fake images
        # using flipped labels!
        D_fake = D(fake_images)
        g_loss = real_loss(D_fake) # use real loss to flip labels
        
        # perform backprop
        g_loss.backward()
        g_optimizer.step()
        
        # Print some loss stats
        if batch_i % print_every == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                                                                                   epoch+1, num_epochs, d_loss.item(), g_loss.item()))


# AFTER EACH EPOCH
# append discriminator loss and generator loss
losses.append((d_loss.item(), g_loss.item()))

    # generate and save sample, fake images
    G.eval() # eval mode for generating samples
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train() # back to train mode




# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
  pkl.dump(samples, f)



fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()


# Helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')

# Load samples from generator, taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)
