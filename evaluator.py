import torch
import numpy as np
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Import the model architectures
from train import Generator, Discriminator  # Make sure this matches your training file name

class HandwritingEvaluator:
    def __init__(self, generator, discriminator, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.generator.eval()
        self.discriminator.eval()
    
    def generate_samples(self, num_samples=100):
        with torch.no_grad():
            noise = torch.randn(num_samples, 100).to(self.device)
            samples = self.generator(noise)
        return samples
    
    def evaluate_quality(self, test_loader):
        total_real_accuracy = 0
        total_fake_accuracy = 0
        total_batches = 0
        
        with torch.no_grad():
            for real_images, _ in tqdm(test_loader, desc="Evaluating"):
                batch_size = real_images.size(0)
                real_images = real_images.to(self.device)
                
                # Generate fake images
                noise = torch.randn(batch_size, 100).to(self.device)
                fake_images = self.generator(noise)
                
                # Get discriminator predictions
                real_preds = self.discriminator(real_images).cpu().numpy()
                fake_preds = self.discriminator(fake_images).cpu().numpy()
                
                # Calculate accuracy
                real_accuracy = np.mean((real_preds > 0.5).astype(np.float32))
                fake_accuracy = np.mean((fake_preds < 0.5).astype(np.float32))
                
                total_real_accuracy += real_accuracy
                total_fake_accuracy += fake_accuracy
                total_batches += 1
        
        avg_real_accuracy = total_real_accuracy / total_batches
        avg_fake_accuracy = total_fake_accuracy / total_batches
        
        return {
            'real_accuracy': avg_real_accuracy,
            'fake_accuracy': avg_fake_accuracy,
            'discriminator_accuracy': (avg_real_accuracy + avg_fake_accuracy) / 2
        }
    
    def save_sample_grid(self, num_samples=25, filename='sample_grid.png'):
        samples = self.generate_samples(num_samples)
        grid = make_grid(samples, nrow=5, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.cpu().numpy().transpose(1, 2, 0), cmap='gray')
        plt.axis('off')
        plt.savefig(filename)
        plt.close()

def test_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load test dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        print("Loading MNIST test dataset...")
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Initialize models
        print("Initializing models...")
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)
        
        # Load the trained weights
        print("Loading trained weights...")
        generator.load_state_dict(torch.load('generator_epoch_100.pth', map_location=device))
        discriminator.load_state_dict(torch.load('discriminator_epoch_100.pth', map_location=device))
        
        # Initialize evaluator
        print("Initializing evaluator...")
        evaluator = HandwritingEvaluator(generator, discriminator, device)
        
        # Generate and save sample grid
        print("Generating sample grid...")
        evaluator.save_sample_grid(filename='generated_samples.png')
        
        # Evaluate model
        print("Evaluating model...")
        metrics = evaluator.evaluate_quality(test_loader)
        
        # Print metrics
        print("\nEvaluation Results:")
        print("-" * 50)
        print(f"Real Image Accuracy: {metrics['real_accuracy']:.4f}")
        print(f"Fake Image Accuracy: {metrics['fake_accuracy']:.4f}")
        print(f"Overall Discriminator Accuracy: {metrics['discriminator_accuracy']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    # First make sure you have all required packages
    import subprocess
    import sys
    
    def install_requirements():
        requirements = [
            'torch',
            'torchvision',
            'numpy',
            'matplotlib',
            'tqdm',
            'scikit-learn'
        ]
        for package in requirements:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("Checking and installing required packages...")
    install_requirements()
    
    # Run the test
    metrics = test_model()
