import torch
import torch.nn as nn

# ============================================
# N-HiTS BLOCK
# ============================================
class NHiTSBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ============================================
# N-HiTS MAIN MODEL
# ============================================
class NHiTS(nn.Module):
    def __init__(self, seq_len=168, num_features=14, hidden_size=128, num_blocks=3):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features

        input_size = seq_len * num_features  # 168 × 14 = 2352

        # Model consists of several forecasting blocks
        self.blocks = nn.ModuleList([
            NHiTSBlock(input_size, hidden_size, 1)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x shape: (batch, seq_len, num_features)
        b, s, f = x.shape
        x = x.reshape(b, s * f)  # flatten → (batch, 2352)

        out = 0
        for block in self.blocks:
            out = out + block(x)

        return out.squeeze()  # final scalar output
