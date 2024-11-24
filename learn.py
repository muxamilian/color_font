import numpy as np
from tqdm import tqdm
from base import *
import os
import datetime

current_time = datetime.datetime.now().isoformat().replace(':', '-').replace('.', '-')
out_mixed_dir = 'out_mixed_' + current_time
out_images_dir = 'out_images_' + current_time

os.makedirs(out_mixed_dir, exist_ok=True)
os.makedirs(out_images_dir, exist_ok=True)

# Generate character images
raw_images, sizes, text_sizes, positions, actual_ascii, _ = generate_char_images(FONT_PATH)
# images = [np.random.rand(3, *IMG_SIZE) * 0.2 - 0.1 for _ in raw_images]
# assert np.max(images) <= 0.1
# assert np.min(images) >= -0.1
images = [np.zeros((1, *IMG_SIZE)) for _ in raw_images]
assert np.max(images) <= 0.0
assert np.min(images) >= -0.0

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
mps_device = torch.device("mps")
# torch.set_default_device('cuda')

batch_size = 64

# Sample images as torch tensors (batch_size, 3, 224, 224)
num_images = len(images)

# Convert images to torch tensors and make them require gradients
images = torch.stack([torch.tensor(image.astype(np.float32)).float() for image in images])
raw_images = torch.stack([torch.tensor(image.astype(np.float32)).float() for image in raw_images])
images = nn.Parameter(images, requires_grad=True)

# Create a pre-trained ResNet50 model
classifier = models.resnet18(pretrained=True)
classifier.fc = nn.Linear(classifier.fc.in_features, num_images)  # Adjust output layer

def add_noise(img):
    return img + torch.rand(tuple(), dtype=torch.float32)*0.5 * torch.randn_like(img)

def get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Creates a 2D Gaussian kernel."""
    # Create a 1D Gaussian kernel
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # Create a 2D Gaussian kernel by outer product
    gauss_kernel = gauss[:, None] @ gauss[None, :]
    return gauss_kernel

def get_divisor(n, max_wanted):
    # Step 1: Find all divisors
    divisors = [1]
    for i in range(2, n+1):
        if i > max_wanted:
            break
        if n % i == 0 and i <= max_wanted:
            divisors.append(i)
    return divisors[-1]

def apply_gaussian_blur(image: torch.Tensor) -> torch.Tensor:
    """Applies Gaussian blur with backpropagation support."""
    # Create a Gaussian kernel
    sigma = torch.rand(tuple(), dtype=torch.float32) * 224/4
    max_divisor = get_divisor(224, sigma)
    noise_mat = (torch.rand(tuple(), dtype=torch.float32)*0.5 * torch.rand((1, 1, int(224/max_divisor), int(224/max_divisor)))).repeat_interleave(repeats=int(max_divisor), dim=2).repeat_interleave(repeats=int(max_divisor), dim=3)
    image = image + noise_mat

    kernel_size = (max(int(sigma), 2) // 2) * 4 - 1
    kernel = get_gaussian_kernel(kernel_size, sigma)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)  # Shape it for convolution
    kernel = kernel.expand(image.size(1), 1, kernel_size, kernel_size)  # Expand to match channel size

    # Apply reflection padding to avoid black edges
    padding = kernel_size // 2
    padded_image = F.pad(image, (padding, padding, padding, padding), mode='reflect')

    # Apply Gaussian blur using convolution
    blurred_image = F.conv2d(padded_image, kernel, padding=0, groups=image.size(1))

    return blurred_image

# Optimizer
optimizer = optim.SGD([{'params': classifier.parameters(), 'weight_decay': 0.0001, 'lr': 1e-2},
                       {'params': images, 'lr': 30}], momentum=0.9)

def normalize_img(img):
    with torch.no_grad():
        min = torch.minimum(img.min(), torch.tensor(0.0, dtype=torch.float32))
        max = torch.maximum(img.max(), torch.tensor(1.0, dtype=torch.float32))
    return (img - min) / (max - min)

def total_variation_loss(batch):
    # Calculate the difference between neighboring pixels in the horizontal direction (along the width)
    diff_x = batch[:, :, :, :-1] - batch[:, :, :, 1:]
    
    # Calculate the difference between neighboring pixels in the vertical direction (along the height)
    diff_y = batch[:, :, :-1, :] - batch[:, :, 1:, :]
    
    # Compute the total variation loss by summing the absolute differences
    loss = torch.mean(diff_x**2) + torch.mean(diff_y**2)
    assert len(loss.shape) == 0

    return loss

def deviation_loss(batch):
    loss = torch.mean(batch**2)
    assert len(loss.shape) == 0

    return loss

# Cross-entropy loss
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(1000000000):
    for step in tqdm(range(1000)):
        optimizer.zero_grad()

        # Randomly sample a batch of images with replacement
        idx = torch.randint(0, num_images, (batch_size,))
        batch = torch.sigmoid(images[idx])
        
        mixed_batch = (1 - raw_images[idx][:,None,:,:]) * batch + raw_images[idx][:,None,:,:]
        augmented_single = [
            normalize_img(
                add_noise(
                    apply_gaussian_blur(
                        # add_noise(img)[None,:,:,:], 
                        img[None,:,:,:]
                    ).squeeze(0)
                )
            )
        for img in mixed_batch]
        augmented_batch = torch.stack(augmented_single)
        
        assert torch.min(augmented_batch) >= 0, f'{torch.min(augmented_batch)}'
        assert torch.max(augmented_batch) <= 1, f'{torch.max(augmented_batch)}'

        if step == 0:
            save_img(augmented_batch, f'{out_mixed_dir}/mixed_{epoch}')
            save_img((1 - raw_images[:,None,:,:]) * torch.sigmoid(images) + raw_images[:,None,:,:], f'{out_images_dir}/images_{epoch}')
        
        normalized_batch = augmented_batch * 2 - 1
        normalized_batch = normalized_batch.expand(-1, 3, -1, -1)

        # Forward pass through the classifier
        outputs = classifier(normalized_batch)

        # Create targets (the correct image indices)
        targets = idx

        smoothness_loss = deviation_loss(images[idx])
        # Compute loss
        classification_loss = criterion(outputs, targets)
        
        loss = classification_loss + 3. * smoothness_loss
        # Backpropagation
        loss.backward()

        # Update the classifier and the images
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item()}, Class: {classification_loss.item()}, Smooth: {smoothness_loss.item()}')
