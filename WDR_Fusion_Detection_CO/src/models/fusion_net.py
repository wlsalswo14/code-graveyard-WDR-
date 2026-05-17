import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if not self.is_last:
            out = F.relu(out, inplace=True)
        return out

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, 3, 1)
        self.conv2 = ConvLayer(in_channels + out_channels, out_channels, 3, 1)
        self.conv3 = ConvLayer(in_channels + 2 * out_channels, out_channels, 3, 1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(torch.cat([x, out1], 1))
        out3 = self.conv3(torch.cat([x, out1, out2], 1))
        return out3

class GRDB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GRDB, self).__init__()
        self.dense = DenseBlock(in_channels, out_channels)
        self.conv = ConvLayer(out_channels, in_channels, 1, 1)

    def forward(self, x):
        out = self.dense(x)
        out = self.conv(out)
        return x + out

class FusionNet(nn.Module):
    def __init__(self, out_channels=3):
        """
        WDR Fusion Network
        Feature Extraction -> Concatenation -> Feature Reconstruction
        """
        super(FusionNet, self).__init__()
        
        # Feature Extraction
        self.conv1_orig = ConvLayer(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.conv1_bin = ConvLayer(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        
        # Feature Reconstruction (64 channels -> 32 channels blocks)
        self.grdb1 = GRDB(64, 32)
        self.grdb2 = GRDB(64, 32)
        self.grdb3 = GRDB(64, 32)
        
        self.conv2 = ConvLayer(64, 64, 3, 1)
        
        self.grdb4 = GRDB(64, 32)
        self.grdb5 = GRDB(64, 32)
        self.grdb6 = GRDB(64, 32)
        
        self.conv3 = ConvLayer(64, out_channels, 3, 1, is_last=True)
            
    def forward(self, original, binary):
        # Independent feature extraction
        feat_orig = self.conv1_orig(original)  # (B, 3, H, W) -> (B, 32, H, W)
        feat_bin = self.conv1_bin(binary)      # (B, 1, H, W) -> (B, 32, H, W)
        
        # Concatenation -> (B, 64, H, W)
        out1 = torch.cat([feat_orig, feat_bin], dim=1)
        
        # Feature Reconstruction
        g1 = self.grdb1(out1)
        g2 = self.grdb2(g1)
        g3 = self.grdb3(g2)
        
        out2 = self.conv2(g3) + out1 # Skip connection
        
        g4 = self.grdb4(out2)
        g5 = self.grdb5(g4)
        g6 = self.grdb6(g5)
        
        out3 = self.conv3(g6)
        
        # Sigmoid to normalize output to [0, 1]
        out3 = torch.sigmoid(out3)
        
        return out3
