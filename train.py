import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class RNNGenerator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=256, num_layers=2):
        super(RNNGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 784),
            nn.Tanh()
        )
        
    def forward(self, z, seq_length=1):
        batch_size = z.size(0)
        z = z.unsqueeze(1).repeat(1, seq_length, 1)
        
        lstm_out, _ = self.rnn(z)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        output = self.decoder(lstm_out)
        output = output.view(batch_size, seq_length, 1, 28, 28)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Calculate the size after convolutions
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

class HandwritingTrainer:
    def __init__(self, generator, discriminator, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        
        os.makedirs('samples', exist_ok=True)
    
    def generate_handwritten_page(self, num_chars=50, chars_per_row=10):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(1, 100).to(self.device)
            fake_chars = self.generator(noise, seq_length=num_chars)
            fake_chars = fake_chars.squeeze(0)
            
            # Create a blank page
            rows = (num_chars + chars_per_row - 1) // chars_per_row
            page = Image.new('L', (28 * chars_per_row, 28 * rows), 255)
            
            for i in range(num_chars):
                row = i // chars_per_row
                col = i % chars_per_row
                char_img = fake_chars[i].cpu().squeeze().numpy()
                char_img = ((char_img + 1) * 127.5).astype(np.uint8)
                char_pil = Image.fromarray(char_img)
                page.paste(char_pil, (col * 28, row * 28))
            
            return page
    
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
                    fake_images = self.generator(noise).squeeze(1)
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
                        page = self.generate_handwritten_page()
                        page.save(f'samples/handwritten_page_epoch_{epoch+1}_batch_{i}.png')
                    
                    pbar.set_postfix({
                        'D_loss': f'{d_loss.item():.4f}',
                        'G_loss': f'{g_loss.item():.4f}'
                    })
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'D_loss: {running_d_loss/len(dataloader):.4f}')
            print(f'G_loss: {running_g_loss/len(dataloader):.4f}')

def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load EMNIST dataset (contains letters)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.EMNIST(
        root='./data',
        split='letters',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    generator = RNNGenerator()
    discriminator = Discriminator()
    
    trainer = HandwritingTrainer(generator, discriminator, device)
    trainer.train(train_loader, num_epochs=50)
    
    # Generate final sample page
    final_page = trainer.generate_handwritten_page(num_chars=100, chars_per_row=10)
    final_page.save('final_handwritten_page.png')
    
    return generator, discriminator

if __name__ == '__main__':
    generator, discriminator = main()
