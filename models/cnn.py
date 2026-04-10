import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn(nn.Module):
    """
    1D CNN for multivariate time-series classification
    Input: [B, T, C] -> [B, C, T] in conv1d
    Output: [B, num_classes]
    """

    def __init__(self, in_channels, num_classes, hidden_channels=[64, 128, 256], kernel_size=3, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        layers = []
        prev_ch = in_channels
        for ch in hidden_channels:
            layers.append(nn.Conv1d(prev_ch, ch, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(dropout))
            prev_ch = ch
        self.feature_extractor = nn.Sequential(*layers)

        # Global average pooling + linear classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [B, C, 1]
        self.classifier = nn.Sequential(
            nn.Linear(prev_ch, num_classes)
        )

    def extract_feature(self, x):
        """
        x: [B, T, C]
        return: [B, d]
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        z = self.feature_extractor(x)  # CNN
        z = self.global_pool(z).squeeze(-1)
        return z

    def forward(self, x):
        x = self.extract_feature(x)  # [B, C]
        out = self.classifier(x)
        return out


# Example usage
if __name__ == "__main__":
    B, T, C = 32, 200, 3  # batch, time steps, channels
    num_classes = 18
    x = torch.randn(B, T, C)
    model = cnn(in_channels=C, num_classes=num_classes)
    logits = model(x)
    print(logits.shape)  # [32, 18]
