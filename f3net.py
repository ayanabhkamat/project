import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np

# ==========================================
# 1. Frequency Aware Decomposition (FAD) Head
# ==========================================
class FAD_Head(nn.Module):
    def __init__(self, width=256, height=256):
        super(FAD_Head, self).__init__()
        self.width = width
        self.height = height
        # Pre-compute DCT filters for Low, Mid, High bands
        self.register_buffer('dct_filters', self.get_dct_filter(width, height))

    def get_dct_filter(self, tile_size_x, tile_size_y):
        filters = torch.zeros(3, tile_size_x, tile_size_y)
        c_x = torch.arange(tile_size_x).repeat(tile_size_y, 1)
        c_y = torch.arange(tile_size_y).repeat(tile_size_x, 1).t()
        freq = c_x + c_y

        # Thresholds for 256x256 image
        low_thresh = tile_size_x // 3
        high_thresh = (tile_size_x // 3) * 2

        filters[0] = torch.where(freq < low_thresh, 1.0, 0.0) # Low
        filters[1] = torch.where((freq >= low_thresh) & (freq < high_thresh), 1.0, 0.0) # Mid
        filters[2] = torch.where(freq >= high_thresh, 1.0, 0.0) # High
        return filters

    def forward(self, x):
        # x: [Batch, 3, H, W]
        B, C, H, W = x.shape
        # FFT transform
        x_freq = torch.fft.rfft2(x, norm='ortho')

        # Resize filters to match current input if necessary
        # rfft2 returns width//2 + 1
        filters = F.interpolate(self.dct_filters.unsqueeze(1), size=(H, W//2 + 1), mode='nearest').squeeze(1)

        out_list = []
        for i in range(3): # Apply Low, Mid, High
            mask = filters[i]
            filtered_freq = x_freq * mask.unsqueeze(0).unsqueeze(0)
            filtered_spatial = torch.fft.irfft2(filtered_freq, s=(H, W), norm='ortho')
            out_list.append(filtered_spatial)

        # Concatenate: Result is [Batch, 9, H, W] (3 bands * 3 channels)
        return torch.cat(out_list, dim=1)

# ==========================================
# 2. Local Frequency Statistics (LFS) Head
# ==========================================
class LFS_Head(nn.Module):
    def __init__(self, in_channels=3, window_size=10, stride=2):
        super(LFS_Head, self).__init__()
        self.window_size = window_size
        self.stride = stride

        # Map flattened frequency spectrum to feature channels
        # rfft2 returns width//2 + 1 columns
        freq_flat_dim = in_channels * window_size * (window_size // 2 + 1)
        self.stat_conv = nn.Sequential(
            nn.Conv2d(freq_flat_dim, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Extract patches: [B, C*Window*Window, N_Patches]
        patches = F.unfold(x, kernel_size=self.window_size, stride=self.stride)

        # Reshape for FFT: [B * N_Patches, C, Window, Window]
        L = patches.size(2)
        patches = patches.view(B, C, self.window_size, self.window_size, L)
        patches = patches.permute(0, 4, 1, 2, 3).contiguous().view(B * L, C, self.window_size, self.window_size)

        # FFT on small patches
        freq_patches = torch.fft.rfft2(patches, norm='ortho')
        power_spectrum = torch.abs(freq_patches)
        power_spectrum = torch.log(power_spectrum + 1e-8)

        # Flatten spectrum: [B*L, Features]
        flat_spectrum = power_spectrum.reshape(B*L, -1)

        # Reshape back to spatial map: [B, Features, H_out, W_out]
        H_out = (H - self.window_size) // self.stride + 1
        W_out = (W - self.window_size) // self.stride + 1
        spectral_map = flat_spectrum.view(B, L, -1).permute(0, 2, 1).view(B, -1, H_out, W_out)

        # Reduce dimension
        out = self.stat_conv(spectral_map)

        # Upsample to match original image size
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

# ==========================================
# 3. F3-Net Model Assembly
# ==========================================
class F3Net(nn.Module):
    def __init__(self, num_classes=2, img_size=256):
        super(F3Net, self).__init__()
        self.fad_head = FAD_Head(width=img_size, height=img_size)
        self.lfs_head = LFS_Head(window_size=10, stride=2)

        # Backbone: ResNet50 (Original paper uses Xception, but ResNet is standard in PyTorch)
        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

        # FAD outputs 9 channels. LFS outputs 64 channels.
        # Total input to backbone = 73 channels.
        # We must modify the first convolution layer of ResNet to accept 73 channels.
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=73,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

        # Initialize the new conv1 weights
        # We average the original RGB weights to start reasonable initialization
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3] = original_conv1.weight
            self.backbone.conv1.weight[:, 3:] = original_conv1.weight.mean(dim=1, keepdim=True).repeat(1, 70, 1, 1)

        # Modify Fully Connected layer for binary classification
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # 1. Frequency Decomposition
        fad_features = self.fad_head(x)   # [B, 9, H, W]
        lfs_features = self.lfs_head(x)   # [B, 64, H, W]

        # 2. Concatenate
        combined = torch.cat([fad_features, lfs_features], dim=1) # [B, 73, H, W]

        # 3. Classification
        out = self.backbone(combined)
        return out

# ==========================================
# 4. Training Loop
# ==========================================
def run_training():
    # --- Configuration ---
    DATA_DIR = "./data" # Structure: data/train/real, data/train/fake
    BATCH_SIZE = 8      # Keep small; LFS module consumes memory
    LR = 0.001
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # No normalization here, or use standard ImageNet mean/std
    ])

    # --- Data Loading (Mock logic if folder doesn't exist) ---
    if os.path.exists(DATA_DIR):
        train_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    else:
        print("Dataset not found. Generating dummy data for demonstration...")
        # Create dummy tensors
        dummy_x = torch.randn(16, 3, 256, 256)
        dummy_y = torch.randint(0, 2, (16,))
        train_data = torch.utils.data.TensorDataset(dummy_x, dummy_y)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # --- Model Setup ---
    model = F3Net(num_classes=2, img_size=256).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- Training Loop ---
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}], Loss: {loss.item():.4f}")

        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}] Complete. Avg Loss: {running_loss/len(train_loader):.4f}, Acc: {epoch_acc:.2f}%")

    print("Training Complete.")

    # Save Model
    torch.save(model.state_dict(), "f3net_model.pth")
    print("Model saved to f3net_model.pth")

if __name__ == "__main__":
    # Windows needs this guard for multiprocessing in DataLoader
    run_training()