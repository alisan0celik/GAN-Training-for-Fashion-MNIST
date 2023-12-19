# This code was created by Alişan Çelik

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Loading the Fashion MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Generator Model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Model Size
z_dim = 100
hidden_dim = 128
image_dim = 28 * 28

# Create Models
generator = Generator(z_dim, hidden_dim, image_dim)
discriminator = Discriminator(image_dim, hidden_dim, 1)

# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)

# Loss function and optimizer
criterion = nn.BCELoss()
lr = 0.0002
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

# Learning rate scheduler
scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.99)
scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.99)

# Training
num_epochs = 500
save_interval = 10

for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    total_loss_g = 0.0
    total_loss_d = 0.0

    for batch_idx, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1)

        # Label for real images
        real_labels = torch.rand(batch_size, 1) * 0.1 + 0.9
        # Label for generated images
        fake_labels = torch.rand(batch_size, 1) * 0.1

        # Noise generation and production
        noise = torch.randn(batch_size, z_dim)
        fake_images = generator(noise)

        # Train Discriminator
        discriminator.zero_grad()
        outputs_real = discriminator(real_images)
        outputs_fake = discriminator(fake_images.detach())

        loss_real = criterion(outputs_real, real_labels)
        loss_fake = criterion(outputs_fake, fake_labels)
        loss_d = loss_real + loss_fake

        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        generator.zero_grad()
        outputs_fake = discriminator(fake_images)
        loss_g = criterion(outputs_fake, real_labels)

        loss_g.backward()
        optimizer_g.step()

        total_loss_g += loss_g.item()
        total_loss_d += loss_d.item()

    avg_loss_g = total_loss_g / len(train_loader)
    avg_loss_d = total_loss_d / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {avg_loss_g:.4f}, Discriminator Loss: {avg_loss_d:.4f}")

    if (epoch + 1) % save_interval == 0:
        # Visualize images produced every 10 epochs
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(16, z_dim)
            generated_images = generator(noise).view(-1, 28, 28).detach().numpy()

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(4):
            for j in range(4):
                axes[i, j].imshow(generated_images[i * 4 + j], cmap='gray')
                axes[i, j].axis('off')

        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.show()

        generator.train()

    # Learning rate scheduling
    scheduler_g.step()
    scheduler_d.step()
# This code was created by Alişan Çelik
