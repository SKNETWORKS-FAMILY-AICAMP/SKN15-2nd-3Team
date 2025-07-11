import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import streamlit as st

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.proj = (
            nn.Conv1d(in_channels, out_channels, 1, stride=stride)
            if downsample or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.proj(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ImprovedCNN1DClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.initial_bn = nn.BatchNorm1d(input_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResidualBlock1D(64, 128, downsample=True),
            ResidualBlock1D(128, 256, downsample=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, feature_dim]
        x = self.conv(x)
        return self.mlp(x)

    def predict(self, input_df, threshold=0.5):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(input_df.values, dtype=torch.float32)
            logits = self.forward(x_tensor)
            probs = torch.sigmoid(logits).view(-1).numpy()  # 안전하게 reshape
            preds = (probs >= threshold).astype(int)
        return preds

    def predict_proba(self, input_df):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(input_df.values, dtype=torch.float32)
            logits = self.forward(x_tensor)
            probs = torch.sigmoid(logits).view(-1).numpy()
        return probs
