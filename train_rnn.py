# train_rnn.py
import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# ---------------- Utils ----------------
def set_seed(seed=2025):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def masked_mse(pred, target, y_mask):
    """
    pred:   [B, Hp, 2]
    target: [B, Ht, 2]
    y_mask: [B, Hm]
    自动对齐到共同长度 L = min(Hp, Ht, Hm)，只在 y_mask==1 的位置计算。
    """
    Hp, Ht, Hm = pred.size(1), target.size(1), y_mask.size(1)
    L = min(Hp, Ht, Hm)
    pred   = pred[:, :L]
    target = target[:, :L]
    w = y_mask[:, :L].unsqueeze(-1).float()           # [B,L,1]
    num = (w.sum() * target.size(-1)).clamp_min(1)    # 有效元素个数（×2，因为(x,y)）
    return ((pred - target) ** 2 * w).sum() / num

def masked_rmse(pred, target, y_mask):
    return torch.sqrt(masked_mse(pred, target, y_mask) + 1e-12)

# ---------------- Model ----------------
class SimpleGRU2H(nn.Module):
    """
    编码器：单层GRU，读 [B,T,F]（用pack忽略padding）。
    解码：用最后隐藏态 h 线性映射到 [H_max, 2]（并行预测），训练时用 Y_mask。
    """
    def __init__(self, in_dim, hidden=128, h_max=30, dropout=0.0):
        super().__init__()
        self.h_max = h_max
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, h_max * 2)

    def forward(self, x, lengths):
        # x: [B,T,F], lengths: [B]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)            # h_n: [1,B,H]
        h = self.drop(h_n.squeeze(0))        # [B,H]
        out = self.head(h).view(-1, self.h_max, 2)  # [B,Hmax,2]
        return out

# ---------------- Data ----------------
def make_loader(pack_path, batch_size=256, shuffle=False):
    pack = torch.load(pack_path)
    X      = pack["X"].float()         # [N,T_in,F]
    X_mask = pack["X_mask"].float()    # [N,T_in]
    Y      = pack["Y"].float()         # [N,Hmax,2]
    Y_mask = pack["Y_mask"].float()    # [N,Hmax]
    H      = pack["H"].long()          # [N]
    dataset = TensorDataset(X, X_mask, Y, Y_mask, H)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    meta = dict(T_in=pack["T_in"], H_max=pack["H_max"], F=X.shape[-1], feature_names=pack.get("feature_names"))
    return loader, meta

# ---------------- Train / Eval ----------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_rmse, total_n = 0.0, 0.0, 0
    for X, X_mask, Y, Y_mask, H in loader:
        X, X_mask, Y, Y_mask = X.to(device), X_mask.to(device), Y.to(device), Y_mask.to(device)
        lengths = X_mask.sum(dim=1).long()
        Y_pred = model(X, lengths)              # [B, Hmax_model, 2]
        loss = masked_mse(Y_pred, Y, Y_mask)    # 自动对齐长度
        rmse = torch.sqrt(loss + 1e-12)
        bs = X.size(0)
        total_loss += loss.item() * bs
        total_rmse += rmse.item() * bs
        total_n += bs
    return total_loss/total_n, total_rmse/total_n

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # loaders
    train_loader, meta_tr = make_loader(os.path.join(args.data_dir, "train.pt"), args.batch_size, shuffle=True)
    val_loader,   meta_va = make_loader(os.path.join(args.data_dir, "val.pt"),   args.batch_size, shuffle=False)
    test_loader,  meta_te = make_loader(os.path.join(args.data_dir, "test.pt"),  args.batch_size, shuffle=False)

    F      = meta_tr["F"]
    # 用“训练集 H_max”初始化模型（验证/测试若不同，损失会自动对齐）
    H_max_model  = meta_tr["H_max"]
    print(f"F={F}, T_in={meta_tr['T_in']}, H_max(train)={H_max_model}, "
          f"H_max(val)={meta_va['H_max']}, H_max(test)={meta_te['H_max']}, "
          f"features={meta_tr['feature_names']}")

    model = SimpleGRU2H(in_dim=F, hidden=args.hidden, h_max=H_max_model, dropout=args.dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, verbose=True)

    best_val = math.inf
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, "best_gru.pt")

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss, total_rmse, total_n = 0.0, 0.0, 0

        for X, X_mask, Y, Y_mask, H in train_loader:
            X, X_mask, Y, Y_mask = X.to(device), X_mask.to(device), Y.to(device), Y_mask.to(device)
            lengths = X_mask.sum(dim=1).long()

            Y_pred = model(X, lengths)          # [B, Hmax_model, 2]
            loss = masked_mse(Y_pred, Y, Y_mask)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            rmse = torch.sqrt(loss + 1e-12)
            bs = X.size(0)
            total_loss += loss.item() * bs
            total_rmse += rmse.item() * bs
            total_n += bs

        tr_loss = total_loss/total_n
        tr_rmse = total_rmse/total_n
        val_loss, val_rmse = evaluate(model, val_loader, device)
        sched.step(val_loss)

        print(f"Epoch {epoch:03d} | Train RMSE {tr_rmse:.4f} | Val RMSE {val_rmse:.4f} | LR {opt.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "meta": meta_tr}, ckpt)
            print(f"  -> saved: {ckpt}")

    # Test with best ckpt
    ckpt_data = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_data["model"])
    test_loss, test_rmse = evaluate(model, test_loader, device)
    print(f"[TEST] RMSE: {test_rmse:.4f}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_norm", help="包含 train.pt/val.pt/test.pt 的目录")
    ap.add_argument("--out_dir",  default="runs_gru")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)

# python train_rnn.py --data_dir data_norm --out_dir runs_gru --epochs 30 --batch_size 512 --hidden 128 --lr 1e-3
