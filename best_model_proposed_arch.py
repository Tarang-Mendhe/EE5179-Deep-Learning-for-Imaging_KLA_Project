import torch
import torch.nn as nn

#Proposed_U-net

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv_block(x)
        residual = self.residual_conv(x) if self.residual_conv else x
        return out + residual

class SCA(nn.Module):
    def __init__(self, in_channels):
        super(SCA, self).__init__()
        self.spatial_attention = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # Pointwise Convolution for Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Spatial Attention
        spatial_out = self.spatial_attention(x)
        spatial_out = torch.sigmoid(spatial_out)

        # Channel Attention
        channel_out = self.channel_attention(x)

        # Element-wise multiplication for attention fusion
        fused_out = x * spatial_out * channel_out

        return fused_out

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.theta = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.g = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.W = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Compute theta, phi, g
        theta_x = self.theta(x).view(batch_size, C // 2, -1)
        phi_x = self.phi(x).view(batch_size, C // 2, -1)
        g_x = self.g(x).view(batch_size, C // 2, -1)

        # Compute attention map
        f = torch.matmul(theta_x.permute(0, 2, 1), phi_x)  # (B, H*W, H*W)
        f_div_C = f / (H * W)  # Normalize
        y = torch.matmul(f_div_C, g_x.permute(0, 2, 1))  # (B, H*W, C//2)
        y = y.view(batch_size, C // 2, H, W)  # Reshape

        # Combine with original features
        y = self.W(y)
        return x + y  # Residual connection

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 16)
        self.att1 = SCA(16)

        self.dconv_down2 = double_conv(16, 32)
        self.att2 = SCA(32)

        self.dconv_down3 = double_conv(32, 64)
        self.att3 = SCA(64)

        self.dconv_down4 = double_conv(64, 128)
        self.att4 = SCA(128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_bottleneck = double_conv(128, 128)  # Bottleneck convolution
        self.nl_block = NonLocalBlock(128)  # Non-Local Block

        self.dconv_up3 = double_conv(64 + 128, 64)
        self.dconv_up2 = double_conv(32 + 64, 32)
        self.dconv_up1 = double_conv(16 + 32, 16)

        self.conv_last = nn.Conv2d(16, n_class, 1)

    def forward(self, x):
        x_input = x
        conv1 = self.dconv_down1(x)
        conv1s = self.att1(conv1)  # Apply attention to skip connection
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        conv2s = self.att2(conv2)  # Apply attention to skip connection
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        conv3s = self.att3(conv3)  # Apply attention to skip connection
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        x = self.att4(x)  # Apply attention to the bottleneck features

        # Bottleneck with Non-Local Block
        x = self.dconv_bottleneck(x)
        x = self.nl_block(x)  # Apply Non-Local Block

        x = self.upsample(x)
        x = torch.cat([x, conv3s], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2s], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1s], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        out = out + x_input

        return out

# Example usage
n_classes = 3
model = UNet(n_class=n_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move the model to GPU if available

# Example input (batch size of 1, 3 channels, 256x256 image)
x = torch.randn(1, 3, 256, 256).to(device)
output = model(x)
print(output.shape)  # Should output: torch.Size([1, 3, 256, 256])

# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")