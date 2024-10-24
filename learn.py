import numpy as np
from PIL import Image, ImageFont, ImageDraw
import string
import os
from tqdm import tqdm
import datetime

# Constants
FONT_PATH = "RobotoMono-VariableFont_wght.ttf"
IMG_SIZE = (224, 224)
ASCII_PRINTABLE = string.printable  # All printable ASCII characters

current_time = datetime.datetime.now().isoformat().replace(':', '-').replace('.', '-')

out_dir = 'out_' + current_time
out_mixed_dir = 'out_mixed_' + current_time
out_images_dir = 'out_images_' + current_time
os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_mixed_dir, exist_ok=True)
os.makedirs(out_images_dir, exist_ok=True)

def generate_char_images(font_path, img_size=(224, 224)):
    """Generate 64x64 matrices for each printable ASCII character."""
    # Specify font size in pixels and the image's DPI
    font_size = 160
    font = ImageFont.truetype(font_path, size=font_size)  # Adjust size to fit in 64x64
    char_images = []

    for char in ASCII_PRINTABLE:
        if len(char.strip()) == 0:
            continue
        # Create a blank image and a drawing context
        image = Image.new('L', img_size, color=255)  # 'L' mode for grayscale
        draw = ImageDraw.Draw(image)
        
        # Get character size and calculate positioning
        text_left, text_top, text_right, text_bottom = draw.textbbox((0,0), char, font=font, font_size=font_size, spacing=0) 
        text_size = (text_right - text_left, text_bottom - text_top)
        position = ((img_size[0] - text_size[0]) // 2, 0)
        
        # Draw character onto the image
        draw.text(position, char, fill=0, font=font)
        char_for_filename = char.replace('.', 'dot').replace('/', 'slash').replace(':', 'colon')
        image.save(f'{out_dir}/{char_for_filename}.png')

        # Convert image to numpy array and normalize
        char_images.append(np.array(image) / 255.0)
    
    return char_images

# Generate character images
raw_images = generate_char_images(FONT_PATH)
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
import torchvision.transforms as T
import torch.nn.functional as F
# mps_device = torch.device("mps")
torch.set_default_device('cuda')

# Define a transform to convert the tensor to a PIL image
to_pil = T.ToPILImage()

def save_img(batch, name):
    
    n_columns = 10
    n_images = batch.size(0)
    n_rows = (n_images + n_columns - 1) // n_columns  # Calculate required rows

    # Define a transform to convert the tensor to a PIL image
    to_pil = T.ToPILImage()

    # Image size (assuming all images are the same size)
    img_width, img_height = 224, 224

    # Create a blank canvas for the tiled image
    tiled_image = Image.new('RGB', (n_columns * img_width, n_rows * img_height))

    # Loop over each image and paste it on the canvas
    for i in range(n_images):
        img_tensor = batch[i]
        img_pil = to_pil(img_tensor)  # Convert to PIL Image
        
        # Calculate position in the grid
        row = i // n_columns
        col = i % n_columns
        
        # Paste the image in the correct position
        tiled_image.paste(img_pil, (col * img_width, row * img_height))
    tiled_image.save(f'{name}.png')

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
    return img + torch.rand(tuple(), dtype=torch.float32)*0.25 * torch.randn_like(img)

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
    noise_mat = (torch.rand(tuple(), dtype=torch.float32)*0.25 * torch.rand((1, 1, int(224/max_divisor), int(224/max_divisor)))).repeat_interleave(repeats=int(max_divisor), dim=2).repeat_interleave(repeats=int(max_divisor), dim=3)
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
                       {'params': images, 'lr': 10}], momentum=0.9)

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
    loss = torch.mean(torch.abs(diff_x)) + torch.mean(torch.abs(diff_y))
    
    return loss

# Cross-entropy loss
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(1000000000):
    for step in tqdm(range(100)):
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
                    ).squeeze()
                )
            )
        for img in mixed_batch]
        augmented_batch = torch.stack(augmented_single)
        
        assert torch.min(augmented_batch) >= 0
        assert torch.max(augmented_batch) <= 1

        if step == 0:
            save_img(augmented_batch, f'{out_mixed_dir}/mixed_{epoch}')
            save_img((1 - raw_images[:,None,:,:]) * torch.sigmoid(images) + raw_images[:,None,:,:], f'{out_images_dir}/images_{epoch}')
        
        normalized_batch = augmented_batch * 2 - 1

        # Forward pass through the classifier
        outputs = classifier(normalized_batch)

        # Create targets (the correct image indices)
        targets = idx

        # smoothness_loss = total_variation_loss(images[idx])
        # Compute loss
        classification_loss = criterion(outputs, targets)
        
        loss = classification_loss# + 0.01 * smoothness_loss
        # Backpropagation
        loss.backward()

        # Update the classifier and the images
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item()}, Class: {classification_loss.item()}')#, Smooth: {smoothness_loss.item()}')
