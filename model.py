import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
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
            ConvBlock(in_channels, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 192, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(192, 128, 1),
            ConvBlock(128, 256, 3, padding=1),
            ConvBlock(256, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            ConvBlock(1024, 1024, 3, stride=2, padding=1),
            ConvBlock(1024, 1024, 3, padding=1),
            ConvBlock(1024, 1024, 3, padding=1),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),  # Dropout with configurable rate
            nn.Linear(4096, 7 * 7 * (5 * num_boxes + num_classes)),
            nn.Sigmoid(),  # For normalizing the output predictions
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.fc_layers(features)

        # Reshape to match the YOLO output format: S x S x (5*B + C)
        # where S=7 (grid size), B=2 (boxes per cell), C=20 (num classes)
        batch_size = x.shape[0]
        return output.reshape(batch_size, 7, 7, (5 * self.num_boxes + self.num_classes))
