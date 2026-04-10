import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, C]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, C]

    def forward(self, x):
        # x: [B, T, C]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class ClassHead(nn.Module):
    """Standalone classification head."""
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x_cls):
        # x_cls: [B, C]
        return self.fc(x_cls)


class transformer(nn.Module):
    """
    Transformer-based time-series classification model.
    Input: [B, W, C]
    """
    def __init__(self, input_dim, embed_dim, depth, num_heads, num_classes, dropout=0.1):
        super().__init__()

        # Project multivariate input to embedding dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True   # **VERY IMPORTANT**, so we keep [B, T, C]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # [CLS] token (optional but recommended for classification)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Classification head (独立写)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [B, W, C]
        """
        B, W, C = x.shape

        # project input
        x = self.input_proj(x)  # [B, W, embed_dim]

        # prepend CLS token
        cls_token = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, W+1, embed_dim]

        # add positional encoding
        x = self.pos_encoding(x)

        # transformer encoder
        x = self.encoder(x)  # [B, W+1, embed_dim]

        # retrieve CLS token
        x_cls = x[:, 0, :]  # [B, embed_dim]

        # classification head
        out = self.classifier(x_cls)  # [B, num_classes]

        return out
