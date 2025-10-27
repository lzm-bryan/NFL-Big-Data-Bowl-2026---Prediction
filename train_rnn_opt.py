# train_rnn_opt.py
import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from math import sqrt

# ---------------- Utils ----------------
def set_seed(seed=2025):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def right_pad(X, X_mask):
    """
    把每个样本的有效历史挪到序列前端，右侧补零（pack 只会看前 L 步）
    X: [B,T,F], X_mask: [B,T] (0/1)
    return X_rpad, lengths  (lengths = 每条样本有效步数)
    """
    lengths = X_mask.sum(dim=1).long()  # [B]
    B, T, F = X.shape
    Xr = torch.zeros_like(X)
    for i in range(B):
        L = int(lengths[i])
        if L > 0:
            Xr[i, :L] = X[i, T-L:]
    return Xr, lengths.clamp(min=1)  # 防止 L=0 的边角炸掉

def masked_mse_weighted(pred, target, y_mask, gamma=0.97):
    """
    pred:   [B, Hp, 2]
    target: [B, Ht, 2]
    y_mask: [B, Hm]
    L = min(Hp, Ht, Hm)
    按步权重: w_step[t] = gamma^t (t从0开始)，近步权重大，远步权重小
    """
    Hp, Ht, Hm = pred.size(1), target.size(1), y_mask.size(1)
    L = min(Hp, Ht, Hm)
    pred   = pred[:, :L]
    target = target[:, :L]
    w_mask = y_mask[:, :L].unsqueeze(-1).float()     # [B,L,1]
    step_w = (gamma ** torch.arange(L, device=pred.device, dtype=pred.dtype)).view(1, L, 1)  # [1,L,1]
    w = w_mask * step_w
    num = (w.sum() * target.size(-1)).clamp_min(1)   # 有效权重个数 ×2
    return ((pred - target)**2 * w).sum() / num

@torch.no_grad()
def masked_rmse(pred, target, y_mask):
    Hp, Ht, Hm = pred.size(1), target.size(1), y_mask.size(1)
    L = min(Hp, Ht, Hm)
    w = y_mask[:, :L].unsqueeze(-1).float()
    num = (w.sum() * target.size(-1)).clamp_min(1)
    return torch.sqrt((((pred[:, :L] - target[:, :L])**2) * w).sum() / num + 1e-12)

# ---------------- Model ----------------
class StackedGRU(nn.Module):
    """
    两层 GRU，读 [B,T,F]（pack 忽略右侧 padding）
    头部一次性输出 [H_max, 2]（并行多步）
    """
    def __init__(self, in_dim, hidden=192, layers=2, h_max=30, dropout=0.2):
        super().__init__()
        self.h_max = h_max
        self.gru = nn.GRU(
            input_size=in_dim, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=(dropout if layers > 1 else 0.0)
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, h_max * 2)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)            # h_n: [layers, B, H]
        h = self.drop(h_n[-1])               # 取顶层: [B,H]
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

def stepwise_rmse(pred, target, y_mask, steps):
    """返回 {k: rmse@k}，k 为截断步数"""
    res = {}
    for k in steps:
        k = min(k, pred.size(1), target.size(1), y_mask.size(1))
        if k <= 0:
            res[k] = float('nan'); continue
        rmse_k = masked_rmse(pred[:, :k], target[:, :k], y_mask[:, :k])
        res[k] = rmse_k.item()
    return res

# ---------------- Train / Eval ----------------
@torch.no_grad()
def evaluate(model, loader, device, gamma):
    model.eval()
    total_loss, total_rmse, total_n = 0.0, 0.0, 0
    sw_acc = {1:0.0, 5:0.0, 10:0.0, 20:0.0}
    for X, X_mask, Y, Y_mask, H in loader:
        X, X_mask, Y, Y_mask = X.to(device), X_mask.to(device), Y.to(device), Y_mask.to(device)
        # 右补零对齐
        Xr, lengths = right_pad(X, X_mask)   # [B,T,F], [B]
        Y_pred = model(Xr, lengths)          # [B, Hmax_model, 2]
        loss = masked_mse_weighted(Y_pred, Y, Y_mask, gamma=gamma)
        rmse = masked_rmse(Y_pred, Y, Y_mask)
        bs = X.size(0)
        total_loss += loss.item() * bs
        total_rmse += rmse.item() * bs
        total_n += bs
        # 分步
        ks = stepwise_rmse(Y_pred, Y, Y_mask, steps=sw_acc.keys())
        for k in sw_acc:
            sw_acc[k] += ks[k] * bs
    out = {
        "loss": total_loss/total_n,
        "rmse": total_rmse/total_n,
        "rmse@1": sw_acc[1]/total_n,
        "rmse@5": sw_acc[5]/total_n,
        "rmse@10": sw_acc[10]/total_n,
        "rmse@20": sw_acc[20]/total_n,
    }
    return out

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # loaders
    train_loader, meta_tr = make_loader(os.path.join(args.data_dir, "train.pt"), args.batch_size, shuffle=True)
    val_loader,   meta_va = make_loader(os.path.join(args.data_dir, "val.pt"),   args.batch_size, shuffle=False)
    test_loader,  meta_te = make_loader(os.path.join(args.data_dir, "test.pt"),  args.batch_size, shuffle=False)

    F      = meta_tr["F"]
    H_max_model  = meta_tr["H_max"]  # 模型用 train 的 H_max；不同 split 在损失里自动对齐
    print(f"F={F}, T_in={meta_tr['T_in']}, H_max(train)={H_max_model}, "
          f"H_max(val)={meta_va['H_max']}, H_max(test)={meta_te['H_max']}, "
          f"features={meta_tr['feature_names']}")

    model = StackedGRU(in_dim=F, hidden=args.hidden, layers=args.layers,
                       h_max=H_max_model, dropout=args.dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 余弦退火 + warmup
    warmup = max(1, int(0.1 * args.epochs))
    def lr_lambda(e):
        if e < warmup:
            return (e+1)/warmup
        # cos 从 1 -> 0.1
        progress = (e - warmup) / max(1, args.epochs - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    best_val = float('inf')
    best_epoch = -1
    es_count = 0
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, "best_gru.pt")

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss, total_rmse, total_n = 0.0, 0.0, 0

        for X, X_mask, Y, Y_mask, H in train_loader:
            X, X_mask, Y, Y_mask = X.to(device), X_mask.to(device), Y.to(device), Y_mask.to(device)
            # 右补零
            Xr, lengths = right_pad(X, X_mask)

            Y_pred = model(Xr, lengths)
            loss = masked_mse_weighted(Y_pred, Y, Y_mask, gamma=args.gamma)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            rmse = masked_rmse(Y_pred, Y, Y_mask)
            bs = X.size(0)
            total_loss += loss.item() * bs
            total_rmse += rmse.item() * bs
            total_n += bs

        tr_loss = total_loss/total_n
        tr_rmse = total_rmse/total_n

        val_stats = evaluate(model, val_loader, device, gamma=args.gamma)
        sched.step()

        print(f"Epoch {epoch:03d} | "
              f"Train RMSE {tr_rmse:.4f} | Val RMSE {val_stats['rmse']:.4f} "
              f"(@1 {val_stats['rmse@1']:.4f} @5 {val_stats['rmse@5']:.4f} "
              f"@10 {val_stats['rmse@10']:.4f} @20 {val_stats['rmse@20']:.4f}) "
              f"| LR {opt.param_groups[0]['lr']:.2e}")

        # 早停 & 保存
        if val_stats["loss"] < best_val - 1e-6:
            best_val = val_stats["loss"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "meta": meta_tr}, ckpt)
            es_count = 0
            print(f"  -> saved: {ckpt}")
        else:
            es_count += 1
            if es_count >= args.patience:
                print(f"Early stopping at epoch {epoch} (best {best_epoch})")
                break

    # Test with best ckpt
    ckpt_data = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_data["model"])
    test_stats = evaluate(model, test_loader, device, gamma=args.gamma)
    print(f"[TEST] RMSE: {test_stats['rmse']:.4f} "
          f"( @1 {test_stats['rmse@1']:.4f} @5 {test_stats['rmse@5']:.4f} "
          f"@10 {test_stats['rmse@10']:.4f} @20 {test_stats['rmse@20']:.4f} )")

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_norm", help="包含 train.pt/val.pt/test.pt 的目录")
    ap.add_argument("--out_dir",  default="runs_gru_opt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.97, help="步权重衰减系数，越小越偏向近步")
    ap.add_argument("--patience", type=int, default=6, help="早停耐心")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)
#
# python train_rnn_opt.py --data_dir data_norm --out_dir runs_gru_opt --epochs 40 --batch_size 512 --hidden 192 --layers 2 --dropout 0.2 --lr 2e-3 --gamma 0.97
#
