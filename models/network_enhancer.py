import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureLearningNetwork(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(FeatureLearningNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1)
        self.norm1 = norm_layer(ngf)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1)
        self.norm2 = norm_layer(ngf)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1)
        self.norm3 = norm_layer(ngf)
        self.relu3 = nn.ReLU(True)
        self.conv4 = nn.Conv2d(ngf, input_nc, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.conv4(x)
        return x

class FeatureLearningNetwork1by1(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(FeatureLearningNetwork1by1, self).__init__()
        
        # 1x1 Convolutions to learn channel-wise interactions
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=1, stride=1, padding=0)
        self.norm1 = norm_layer(ngf)
        self.relu1 = nn.ReLU(True)
        
        self.conv2 = nn.Conv2d(ngf, ngf, kernel_size=1, stride=1, padding=0)
        self.norm2 = norm_layer(ngf)
        self.relu2 = nn.ReLU(True)
        
        self.conv3 = nn.Conv2d(ngf, ngf, kernel_size=1, stride=1, padding=0)
        self.norm3 = norm_layer(ngf)
        self.relu3 = nn.ReLU(True)
        
        self.conv4 = nn.Conv2d(ngf, input_nc, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.conv4(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class AttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        b, c, h, w = x.size()
        query = x.view(b, c, -1)  # B x C x HW
        key = x.view(b, c, -1).permute(0, 2, 1)  # B x HW x C
        energy = torch.bmm(query, key)  # B x C x C
        attention = self.softmax(energy)
        out = torch.bmm(attention, query).view(b, c, h, w)
        out = self.conv(out)
        return out

class EnhancerNetwork(nn.Module):
    def __init__(self, in_channels=3, num_residual_blocks=5):
        super(EnhancerNetwork, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual Blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        
        # Attention Layer
        self.attention = AttentionLayer(64)
        
        # Output Convolution
        self.output_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = self.initial_conv(x)
        out = self.relu(out)
        out = self.residual_blocks(out)
        out = self.attention(out)
        out = self.output_conv(out)
        return out

# class FeatureExtractor(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(FeatureExtractor, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
    
#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))

# class FeatureFusion(nn.Module):
#     def __init__(self, in_channels):
#         super(FeatureFusion, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
    
#     def forward(self, sr_features, ref_features):
#         combined = torch.cat((sr_features, ref_features), dim=1)
#         out = self.relu(self.conv1(combined))
#         return self.relu(self.conv2(out))

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(in_channels)
        
#     def forward(self, x):
#         residual = x
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += residual
#         return out

# class EnhancerNetwork(nn.Module):
#     def __init__(self, in_channels=3, num_residual_blocks=5):
#         super(EnhancerNetwork, self).__init__()
#         self.feature_extractor_sr = FeatureExtractor(in_channels, 64)
#         self.feature_extractor_ref = FeatureExtractor(in_channels, 64)
        
#         # Feature fusion layer
#         self.feature_fusion = FeatureFusion(64)
        
#         # Residual blocks
#         self.residual_blocks = nn.Sequential(
#             *[ResidualBlock(64) for _ in range(num_residual_blocks)]
#         )
        
#         # Output layer
#         self.output_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
    
#     def forward(self, data_sr_patch, data_ref_patches):
#         # Feature extraction from both sr and ref patches
#         sr_features = self.feature_extractor_sr(data_sr_patch)
#         ref_features = self.feature_extractor_ref(data_ref_patches)
        
#         # 배치 크기를 원래대로 되돌리기
#         ref_features = ref_features.view(data_sr_patch.size(0), -1, ref_features.size(1), ref_features.size(2), ref_features.size(3))
#         ref_features = ref_features.mean(dim=1)  # ref_features를 평균내서 sr_features와 동일한 크기로 맞춤

#         # Fusion of the features
#         fused_features = self.feature_fusion(sr_features, ref_features)
        
#         # Passing through residual blocks
#         enhanced_features = self.residual_blocks(fused_features)
        
#         # Output enhanced data_sr_patch
#         enhanced_data_sr_patch = self.output_conv(enhanced_features)
#         return enhanced_data_sr_patch
