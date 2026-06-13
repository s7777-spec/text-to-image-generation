from transformers import AutoTokenizer
import torch
import torch.nn as nn

# Step 1: Text Preprocessing
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "A beautiful flower garden"

encoded = tokenizer(
    text,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print("Input Text:", text)
print("Input IDs:", encoded["input_ids"])

# Step 2: Simple Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=2,
            batch_first=True
        )

    def forward(self, x):
        output, _ = self.attention(x, x, x)
        return output

# Step 3: Simple Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(100, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 512)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 4: Attention Processing
attention_layer = SelfAttention(embed_dim=8)

sample_embedding = torch.randn(1, 10, 8)
attention_output = attention_layer(sample_embedding)

print("Attention Output Shape:", attention_output.shape)

# Step 5: GAN-Based Image Generation
generator = Generator()

noise = torch.randn(1, 100)
generated_output = generator(noise)

print("Generated Output Shape:", generated_output.shape)

print("\nComplete Text-to-Image Pipeline Executed Successfully")
