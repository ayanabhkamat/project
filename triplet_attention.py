import torch
import torch.nn as nn

class BasicConv(nn.Module):
    """
    Standard Convolutional Block from Triplet Attention.
    Consists of Conv2d -> BatchNorm -> ReLU.
    """
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    """
    ZPool: Concatenates MaxPool and AvgPool across the channel dimension.
    Reduces dimension C to 2 (1 max + 1 avg).
    """
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )

class AttentionGate(nn.Module):
    """
    Computes attention weights using ZPool and a Convolution.
    """
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7 # Kernel size of 7 provides the "Local" receptive field
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletGlobalLocalAttention(nn.Module):
    """
    Global-Local Attention Module using the Triplet Attention Method.

    - 'Local' aspect is handled by the spatial branch (H, W).
    - 'Global' aspect is handled by the rotation branches (C, H) and (C, W).

    Accepts sequence inputs (L, B, E) typical of Transformers, reshapes to (B, C, H, W),
    applies attention, and reshapes back.
    """
    def __init__(self, no_spatial=False):
        super(TripletGlobalLocalAttention, self).__init__()
        # Branch 1: Channel-Width Interaction (Global C, Local W)
        self.cw = AttentionGate()
        # Branch 2: Height-Channel Interaction (Global C, Local H)
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            # Branch 3: Spatial Interaction (Local H, Local W)
            self.hw = AttentionGate()

    def forward(self, x, H, W):
        """
        Args:
            x: Input tensor of shape (L, B, C) or (B, C, H, W)
            H, W: Spatial dimensions of the original image (required if input is sequence)
        """
        is_sequence = x.dim() == 3

        if is_sequence:
            # Reshape Sequence (L, B, C) -> Image (B, C, H, W)
            L, B, C = x.shape
            assert L == H * W, "Sequence length L must equal H * W"
            # Permute to (B, C, L) then view as (B, C, H, W)
            x = x.permute(1, 2, 0).view(B, C, H, W).contiguous()

        # --- Triplet Attention Logic Starts Here ---

        # Branch 1: Cross-Dimension (C, W) - Rotation
        # Permute (B, C, H, W) -> (B, H, C, W)
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        # Branch 2: Cross-Dimension (C, H) - Rotation
        # Permute (B, C, H, W) -> (B, W, H, C)
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        # Branch 3: Spatial (H, W) - Standard
        if not self.no_spatial:
            x_out = self.hw(x)
            # Average aggregation of Global (Rotated) and Local (Spatial) branches
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)

        # --- Triplet Attention Logic Ends Here ---

        if is_sequence:
            # Reshape Image (B, C, H, W) -> Sequence (L, B, C)
            # View as (B, C, L) then permute to (L, B, C)
            x_out = x_out.flatten(2).permute(2, 0, 1).contiguous()

        return x_out

# Example Usage
if __name__ == "__main__":
    # Simulate an input from SFGLA (Sequence Length, Batch, Channels)
    B, C, H, W = 2, 64, 32, 32
    L = H * W
    input_tensor = torch.randn(L, B, C)

    # Initialize the module
    triplet_gl_att = TripletGlobalLocalAttention()

    # Forward pass
    output = triplet_gl_att(input_tensor, H, W)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}") # Should be (1024, 2, 64)