import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# =============== 1. 读取数据 ===============
data = pd.read_csv("temperature.csv")
series = data.iloc[:, 1].values.astype(np.float32)

# 简单标准化
mean, std = series.mean(), series.std()
series = (series - mean) / std

# =============== 2. 构造窗口数据 ===============
def make_dataset(series, window=3):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    X = torch.tensor(X).unsqueeze(-1)  # [样本数, window, 1]
    y = torch.tensor(y).unsqueeze(-1)  # [样本数, 1]
    return X, y

# 划分训练 / 测试（例如 80% / 20%）
split = int(len(series) * 0.8)
train_series = series[:split]
test_series  = series[split - 3:]   # 保留3个历史点作为预测起点

X_train, y_train = make_dataset(train_series)
X_test, y_test = make_dataset(test_series)

# =============== 3. 定义模型 ===============
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=8, batch_first=True)
        self.fc = nn.Linear(8, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = SimpleRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# =============== 4. 训练 ===============
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            model.eval()
            val_loss = criterion(model(X_test), y_test).item()
        print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Test Loss: {val_loss:.4f}")

# =============== 5. 测试预测（反标准化） ===============
with torch.no_grad():
    last_seq = torch.tensor(series[-4:-1]).unsqueeze(0).unsqueeze(-1)  # [1,3,1]
    pred = model(last_seq).item()
    pred_real = pred * std + mean
    print(f"\n预测下一个: {pred_real:.3f}")
