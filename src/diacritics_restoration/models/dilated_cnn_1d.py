import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = F.relu(out)

        return out


class DiacriticsCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_channels=256, num_blocks=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.input_conv = nn.Conv1d(embedding_dim, num_channels, kernel_size=3, padding=1)

        self.residual_blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2**i
            self.residual_blocks.append(ResidualDilatedBlock(num_channels, kernel_size=3, dilation=dilation))

        self.output_conv = nn.Conv1d(num_channels, vocab_size, kernel_size=1)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        x = F.relu(self.input_conv(x))  # (batch_size, num_channels, seq_len)

        for block in self.residual_blocks:
            x = block(x)  # (batch_size, num_channels, seq_len)

        logits = self.output_conv(x)  # (batch_size, vocab_size, seq_len)

        return logits
