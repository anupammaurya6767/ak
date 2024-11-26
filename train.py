import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            # First layer
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            # Second layer
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            # Third layer
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            # Output layer
            nn.Linear(1024, 784),  # 28x28 = 784
            nn.Tanh()
        )
        
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # First layer
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Second layer
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Third layer
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# Trainer class
class HandwritingTrainer:
    def __init__(self, generator, discriminator, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Create directory for samples
        os.makedirs('samples', exist_ok=True)
    
    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            running_d_loss = 0
            running_g_loss = 0
            
            with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for i, (real_images, _) in enumerate(pbar):
                    batch_size = real_images.size(0)
                    real_images = real_images.to(self.device)
                    
                    # Train Discriminator
                    self.d_optimizer.zero_grad()
                    label_real = torch.ones(batch_size, 1).to(self.device)
                    label_fake = torch.zeros(batch_size, 1).to(self.device)
                    
                    output_real = self.discriminator(real_images)
                    d_loss_real = self.criterion(output_real, label_real)
                    
                    noise = torch.randn(batch_size, 100).to(self.device)
                    fake_images = self.generator(noise)
                    output_fake = self.discriminator(fake_images.detach())
                    d_loss_fake = self.criterion(output_fake, label_fake)
                    
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    self.d_optimizer.step()
                    
                    # Train Generator
                    self.g_optimizer.zero_grad()
                    output_fake = self.discriminator(fake_images)
                    g_loss = self.criterion(output_fake, label_real)
                    g_loss.backward()
                    self.g_optimizer.step()
                    
                    running_d_loss += d_loss.item()
                    running_g_loss += g_loss.item()
                    
                    if i % 100 == 0:
                        # Save sample images
                        with torch.no_grad():
                            fake_images = self.generator(torch.randn(16, 100).to(self.device))
                            self.save_images(fake_images, f'samples/fake_epoch_{epoch+1}_batch_{i}.png')
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'D_loss': f'{d_loss.item():.4f}',
                        'G_loss': f'{g_loss.item():.4f}'
                    })
            
            # Save models after each epoch
            torch.save(self.generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
            torch.save(self.discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')
            
            # Print epoch statistics
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'D_loss: {running_d_loss/len(dataloader):.4f}')
            print(f'G_loss: {running_g_loss/len(dataloader):.4f}')
    
    @staticmethod
    def save_images(images, path):
        plt.figure(figsize=(4, 4))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(images[i].cpu().squeeze().numpy(), cmap='gray')
            plt.axis('off')
        plt.savefig(path)
        plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    # Initialize models
    generator = Generator()
    discriminator = Discriminator()
    
    # Initialize trainer
    trainer = HandwritingTrainer(generator, discriminator, device)
    
    # Train the model
    trainer.train(train_loader, num_epochs=50)
    
    return generator, discriminator

if __name__ == '__main__':
    generator, discriminator = main()