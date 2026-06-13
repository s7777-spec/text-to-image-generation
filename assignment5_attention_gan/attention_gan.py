import torch
import torch.nn as nn

# Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()

        query = self.query_conv(x).view(batch_size, -1, width * height)
        key = self.key_conv(x).view(batch_size, -1, width * height)

        attention = self.softmax(torch.bmm(query.permute(0, 2, 1), key))

        value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x

        return out


# Example Generator with Attention
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc = nn.Linear(100, 256)
        self.attention = SelfAttention(16)

    def forward(self, z):
        x = self.fc(z)
        return x


# Initialize Model
generator = Generator()

print("Attention GAN initialized successfully.")
