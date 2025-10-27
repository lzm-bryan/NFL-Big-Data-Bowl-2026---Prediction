import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# =============== 1. 读取并预处理 ===============
data = pd.read_csv("temperature.csv")
series = data.iloc[:, 1].values.astype(np.float32)
mean, std = series.mean(), series.std()
series = (series - mean) / std

# =============== 2. 构造窗口7 -> 窗口7 的数据 ===============
def make_dataset(series, in_len=7, out_len=7):
    X, y = [], []
    total_len = in_len + out_len
    for i in range(len(series) - total_len):
        X.append(series[i:i+in_len])
        y.append(series[i+in_len:i+total_len])
    X = torch.tensor(X).unsqueeze(-1)  # [样本, in_len, 1]
    y = torch.tensor(y).unsqueeze(-1)  # [样本, out_len, 1]
    return X, y

# 训练/测试划分
split = int(len(series) * 0.8)
train_series = series[:split]
test_series  = series[split - 7:]   # 保留7个历史点做上下文

X_train, y_train = make_dataset(train_series, 7, 7)
X_test,  y_test  = make_dataset(test_series, 7, 7)

# =============== 3. 定义模型（RNN序列到序列） ===============
class Seq2SeqRNN(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # x: [B, in_len, 1]
        out, _ = self.rnn(x)          # [B, in_len, hidden]
        # 我们只用最后隐藏状态作为上下文，重复预测 out_len 次
        h_last = out[:, -1:, :]       # [B, 1, hidden]
        preds = []
        h = h_last
        for _ in range(7):            # 输出长度
            y = self.fc(h)            # [B,1,1]
            preds.append(y)
            # 下一步的输入可以选择y或者保持h
            _, h = self.rnn(y, h.permute(1,0,2).contiguous())
            h = h.permute(1,0,2)
        return torch.cat(preds, dim=1)  # [B, out_len, 1]

model = Seq2SeqRNN(hidden_size=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# =============== 4. 训练 ===============
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        with torch.no_grad():
            val_loss = criterion(model(X_test), y_test).item()
        print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Test Loss: {val_loss:.4f}")

# =============== 5. 测试预测 + 反标准化 ===============
with torch.no_grad():
    last_seq = torch.tensor(series[-14:-7]).unsqueeze(0).unsqueeze(-1)  # [1,7,1]
    pred_seq = model(last_seq).squeeze(0).squeeze(-1).numpy()        # [7]
    pred_real = pred_seq * std + mean
    print("\n预测接下来的7个:")
    print(np.round(pred_real, 2))



# self.fc = nn.Linear(hidden_size, out_len)输出不是一个维度的话

# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
#
# # =============== 1. 读取并预处理 ===============
# data = pd.read_csv("temperature.csv")
# series = data.iloc[:, 1].values.astype(np.float32)
# mean, std = series.mean(), series.std()
# series = (series - mean) / std
#
# # =============== 2. 构造窗口7 -> 窗口7 的数据 ===============
# def make_dataset(series, in_len=7, out_len=7):
#     X, y = [], []
#     total_len = in_len + out_len
#     for i in range(len(series) - total_len):
#         X.append(series[i:i+in_len])
#         y.append(series[i+in_len:i+total_len])
#     X = torch.tensor(X).unsqueeze(-1)  # [样本, in_len, 1]
#     y = torch.tensor(y).unsqueeze(-1)  # [样本, out_len, 1]
#     return X, y
#
# # 训练/测试划分
# split = int(len(series) * 0.8)
# train_series = series[:split]
# test_series  = series[split - 7:]   # 保留7个历史点做上下文
#
# X_train, y_train = make_dataset(train_series, 7, 7)
# X_test,  y_test  = make_dataset(test_series, 7, 7)
#
# # =============== 3. 模型定义：一步输出七个 ===============
# class Seq2SeqRNN(nn.Module):
#     def __init__(self, hidden_size=16, out_len=7):
#         super().__init__()
#         self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
#         # 输出层：一次性输出 out_len*1 个数
#         self.fc = nn.Linear(hidden_size, out_len)
#
#     def forward(self, x):
#         # x: [B, in_len, 1]
#         out, _ = self.rnn(x)            # [B, in_len, hidden]
#         h_last = out[:, -1, :]          # 取最后时刻隐藏状态 [B, hidden]
#         y = self.fc(h_last)             # [B, out_len]
#         return y.unsqueeze(-1)          # [B, out_len, 1]，与目标形状匹配
#
# model = Seq2SeqRNN(hidden_size=16, out_len=7)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = nn.MSELoss()
#
# # =============== 4. 训练 ===============
# for epoch in range(500):
#     model.train()
#     optimizer.zero_grad()
#     pred = model(X_train)
#     loss = criterion(pred, y_train)
#     loss.backward()
#     optimizer.step()
#
#     if (epoch+1) % 50 == 0:
#         with torch.no_grad():
#             val_loss = criterion(model(X_test), y_test).item()
#         print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Test Loss: {val_loss:.4f}")
#
# # =============== 5. 测试预测 + 反标准化 ===============
# with torch.no_grad():
#     last_seq = torch.tensor(series[-14:-7]).unsqueeze(0).unsqueeze(-1)  # [1,7,1]
#     pred_seq = model(last_seq).squeeze(0).squeeze(-1).numpy()           # [7]
#     pred_real = pred_seq * std + mean
#     print("\n预测接下来的7个:")
#     print(np.round(pred_real, 2))


# 很多布和很多特征
#   self.fc = nn.Linear(hidden_size, out_len * num_targets)

# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
#
# # =============== 1. 读取并预处理 ===============
# data = pd.read_csv("temperature.csv")
# data = data.dropna().astype(np.float32)
#
# # 假设第一列是时间戳，不参与训练
# series = data.iloc[:, 1:].values  # shape [T, num_features]
# mean = series.mean(axis=0, keepdims=True)
# std = series.std(axis=0, keepdims=True)
# series = (series - mean) / (std + 1e-8)  # 每个特征独立标准化
#
# # =============== 2. 构造窗口7 -> 窗口7（输入输出都多特征） ===============
# def make_dataset(series, in_len=7, out_len=7):
#     X, y = [], []
#     total_len = in_len + out_len
#     for i in range(len(series) - total_len):
#         X.append(series[i:i+in_len])           # [in_len, num_features]
#         y.append(series[i+in_len:i+total_len]) # [out_len, num_features]
#     X = torch.tensor(np.array(X))              # [N, in_len, num_features]
#     y = torch.tensor(np.array(y))              # [N, out_len, num_features]
#     return X, y
#
# split = int(len(series) * 0.8)
# train_series = series[:split]
# test_series  = series[split - 7:]
#
# X_train, y_train = make_dataset(train_series, 7, 7)
# X_test,  y_test  = make_dataset(test_series, 7, 7)
#
# num_features = X_train.shape[2]
#
# # =============== 3. LSTM 模型：多输入 -> 多输出 ===============
# class Seq2SeqLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size=64, out_len=7, num_targets=None):
#         super().__init__()
#         if num_targets is None:
#             num_targets = input_size  # 默认输出维度与输入特征数相同
#         self.out_len = out_len
#         self.num_targets = num_targets
#
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
#         # 一次性输出所有未来步 × 特征数
#         self.fc = nn.Linear(hidden_size, out_len * num_targets)
#
#     def forward(self, x):
#         # x: [B, in_len, input_size]
#         out, _ = self.lstm(x)           # [B, in_len, hidden]
#         h_last = out[:, -1, :]          # [B, hidden]
#         y = self.fc(h_last)             # [B, out_len * num_targets]
#         y = y.view(-1, self.out_len, self.num_targets)  # [B, out_len, num_targets]
#         return y
#
# model = Seq2SeqLSTM(input_size=num_features, hidden_size=64, out_len=7)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss()
#
# # =============== 4. 训练 ===============
# for epoch in range(300):
#     optimizer.zero_grad()
#     pred = model(X_train)
#     loss = criterion(pred, y_train)
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 50 == 0:
#         val_loss = criterion(model(X_test), y_test).item()
#         print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Test Loss: {val_loss:.4f}")
#
# # =============== 5. 测试预测 + 反标准化（全部特征） ===============
# with torch.no_grad():
#     last_seq = torch.tensor(series[-14:-7]).unsqueeze(0)   # [1,7,num_features]
#     pred_seq = model(last_seq).squeeze(0).numpy()          # [7,num_features]
#     pred_real = pred_seq * std + mean
#     print("\n预测接下来的7步（每步包含所有特征）:")
#     print(np.round(pred_real, 3))
