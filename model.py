import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, num_boxes=2, num_classes=20, dropout_rate=0.5):
        super(YOLOv1, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # NOTE: Simplified architecture following the original YOLO v1 paper
        # Key differences:
        # 1. OG had 24 conv layers, this has fewer
        # 2. OG used skip connections, this uses sequential blocks
        # 3. OG had more complex backbone with additional conv blocks
        self.feature_extractor = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(512, 1024),
            ConvBlock(1024, 1024),
            ConvBlock(1024, 1024),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(1024, 1024),
            ConvBlock(1024, 1024),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 7 * 7 * (5 * num_boxes + num_classes)),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def use_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        self.feature_extractor.use_checkpointing()

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output.view(-1, 7, 7, (5 * self.num_boxes + self.num_classes))
