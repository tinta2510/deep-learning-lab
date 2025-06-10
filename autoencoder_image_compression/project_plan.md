📌 Project Overview: Image Compression with Autoencoders
🎯 Goal
Train an autoencoder to:

Compress an image into a low-dimensional latent vector.

Reconstruct the original image from this vector.

Compare performance with JPEG compression.

🧱 Step-by-Step Plan
1. Dataset Preparation
Dataset Options:

MNIST (grayscale, 28x28) – start here

CIFAR-10 (RGB, 32x32) – for more challenge

Normalize image values to [0, 1].

2. Autoencoder Architecture (Example for CIFAR-10)
python
Copy
Edit
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [B, 16, 16, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [B, 32, 8, 8]
            nn.ReLU(),
            nn.Flatten(),                              # [B, 32*8*8]
            nn.Linear(32*8*8, 128),                    # Latent vector
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 32*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, 8)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0,1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
3. Training Loop
Loss: MSELoss() or BCEWithLogitsLoss() (if binary)

Optimizer: Adam

Epochs: ~20–50

4. Latent Space Insights
Visualize a few 2D or 3D embeddings using PCA/t-SNE.

See how similar images cluster together.

5. Compression Comparison with JPEG
Method:
For each test image:

Compress it using JPEG (e.g., using PIL or OpenCV)

Reconstruct it using your autoencoder

Compare reconstruction quality using:

MSE

PSNR (Peak Signal-to-Noise Ratio)

SSIM (Structural Similarity Index)

Sample JPEG Comparison (Python):
python
Copy
Edit
from PIL import Image
from io import BytesIO
import numpy as np

def jpeg_compress_decompress(image, quality=25):
    buffer = BytesIO()
    pil_img = Image.fromarray((image * 255).astype(np.uint8))
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer)) / 255.0
📈 Evaluation Metrics
MSE (Mean Squared Error)

PSNR: Higher = better quality

SSIM: Closer to 1 = better structure preservation

You can plot original vs JPEG vs autoencoder-reconstructed images side by side for visual insight.

🔍 Extensions & Variants
Add a bottleneck size slider and visualize how image quality degrades as compression increases.

Try convolutional autoencoders vs. fully connected.

Add quantization to simulate actual compression storage.

Would you like a full runnable notebook for this project or a visual diagram of the architecture?




