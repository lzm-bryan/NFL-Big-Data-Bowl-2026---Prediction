import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# =============== 1. 数据预处理 ===============
data = pd.read_csv("temperature.csv").dropna().astype(np.float32)

# 假设第1列是时间戳
series = data.iloc[:, 1:].values  # shape [T, num_features]
mean = series.mean(axis=0, keepdims=True)
std = series.std(axis=0, keepdims=True)
series = (series - mean) / (std + 1e-8)

# =============== 2. 构造窗口数据 ===============
def make_dataset(series, in_len=7, out_len=7):
    X, y = [], []
    total_len = in_len + out_len
    for i in range(len(series) - total_len):
        X.append(series[i:i+in_len])           # [in_len, num_features]
        y.append(series[i+in_len:i+total_len]) # [out_len, num_features]
    X = torch.tensor(np.array(X))              # [N, in_len, num_features]
    y = torch.tensor(np.array(y))              # [N, out_len, num_features]
    return X, y

split = int(len(series) * 0.8)
train_series = series[:split]
test_series  = series[split - 7:]
X_train, y_train = make_dataset(train_series, 7, 7)
X_test,  y_test  = make_dataset(test_series, 7, 7)

num_features = X_train.shape[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============== 3. Encoder–Decoder 模型定义 ===============
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x: [B, in_len, input_size]
        outputs, (h, c) = self.lstm(x)   # h, c: [num_layers, B, hidden_size]
        return h, c

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, out_len, num_layers=1):
        super().__init__()
        self.out_len = out_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, h, c, decoder_input):
        # decoder_input: 初始输入（通常为上一步输出），shape [B,1,input_size]
        preds = []
        for _ in range(self.out_len):
            out, (h, c) = self.lstm(decoder_input, (h, c))  # out: [B,1,hidden]
            y = self.fc(out)                                # [B,1,output_size]
            preds.append(y)
            decoder_input = y                               # 下一步输入 = 当前预测
        preds = torch.cat(preds, dim=1)                     # [B,out_len,output_size]
        return preds

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, out_len, num_layers=1):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(input_size, hidden_size, input_size, out_len, num_layers)

    def forward(self, x):
        h, c = self.encoder(x)
        # decoder 初始输入 = 上一个时间步的最后输入
        decoder_input = x[:, -1:, :]     # [B,1,input_size]
        out = self.decoder(h, c, decoder_input)
        return out                       # [B,out_len,input_size]

model = Seq2Seq(input_size=num_features, hidden_size=64, out_len=7).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# =============== 4. 训练 ===============
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train.to(device))
    loss = criterion(pred, y_train.to(device))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            val_loss = criterion(model(X_test.to(device)), y_test.to(device)).item()
        print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Test Loss: {val_loss:.4f}")

# =============== 5. 推理（预测未来7步） ===============
with torch.no_grad():
    last_seq = torch.tensor(series[-14:-7]).unsqueeze(0).to(device)  # [1,7,num_features]
    pred_seq = model(last_seq).squeeze(0).cpu().numpy()              # [7,num_features]
    pred_real = pred_seq * std + mean
    print("\n预测接下来的7步（多特征）:")
    print(np.round(pred_real, 3))




# 能并行吗？
# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
#
# # =============== 1. 数据预处理 ===============
# data = pd.read_csv("temperature.csv").dropna().astype(np.float32)
#
# # 假设第1列为时间戳
# series = data.iloc[:, 1:].values  # [T, num_features]
# mean = series.mean(axis=0, keepdims=True)
# std = series.std(axis=0, keepdims=True)
# series = (series - mean) / (std + 1e-8)
#
# # =============== 2. 构造窗口数据 ===============
# def make_dataset(series, in_len=7, out_len=7):
#     X, y = [], []
#     total_len = in_len + out_len
#     for i in range(len(series) - total_len):
#         X.append(series[i:i+in_len])           # [in_len, num_features]
#         y.append(series[i+in_len:i+total_len]) # [out_len, num_features]
#     X = torch.tensor(np.array(X))
#     y = torch.tensor(np.array(y))
#     return X, y
#
# split = int(len(series) * 0.8)
# train_series = series[:split]
# test_series  = series[split - 7:]
#
# X_train, y_train = make_dataset(train_series, 7, 7)
# X_test,  y_test  = make_dataset(test_series, 7, 7)
# num_features = X_train.shape[2]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # =============== 3. Encoder + 并行 Decoder 模型定义 ===============
# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#
#     def forward(self, x):
#         _, (h, c) = self.lstm(x)
#         return h[-1]  # [B, hidden_size]
#
# class ParallelDecoder(nn.Module):
#     def __init__(self, hidden_size, out_len, num_features):
#         super().__init__()
#         self.fc = nn.Linear(hidden_size, out_len * num_features)
#         self.out_len = out_len
#         self.num_features = num_features
#
#     def forward(self, context):
#         # context: [B, hidden_size]
#         out = self.fc(context)  # [B, out_len * num_features]
#         return out.view(-1, self.out_len, self.num_features)
#
# class Seq2SeqParallel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, out_len=7):
#         super().__init__()
#         self.encoder = Encoder(input_size, hidden_size)
#         self.decoder = ParallelDecoder(hidden_size, out_len, input_size)
#
#     def forward(self, x):
#         context = self.encoder(x)
#         out = self.decoder(context)
#         return out  # [B, out_len, num_features]
#
# model = Seq2SeqParallel(num_features, hidden_size=64, out_len=7).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss()
#
# # =============== 4. 训练 ===============
# for epoch in range(300):
#     model.train()
#     optimizer.zero_grad()
#     pred = model(X_train.to(device))
#     loss = criterion(pred, y_train.to(device))
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 50 == 0:
#         with torch.no_grad():
#             val_loss = criterion(model(X_test.to(device)), y_test.to(device)).item()
#         print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Test Loss: {val_loss:.4f}")
#
# # =============== 5. 预测 + 反标准化 ===============
# with torch.no_grad():
#     last_seq = torch.tensor(series[-14:-7]).unsqueeze(0).to(device)  # [1,7,num_features]
#     pred_seq = model(last_seq).squeeze(0).cpu().numpy()              # [7,num_features]
#     pred_real = pred_seq * std + mean
#     print("\n预测接下来的7步（每步包含所有特征）:")
#     print(np.round(pred_real, 3))
