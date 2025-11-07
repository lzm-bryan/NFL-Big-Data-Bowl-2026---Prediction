# data_process.py
import os, glob, argparse, json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ----------------- helpers -----------------
def to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true","1","t","yes"])

def find_outputs(root: str, in_file: str) -> str:
    return os.path.join(root, os.path.basename(in_file).replace("input_", "output_"))

def split_files(files, val=0.1, test=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(files)); rng.shuffle(idx)
    ntest, nval = int(len(files)*test), int(len(files)*val)
    test_set = set(files[i] for i in idx[:ntest])
    val_set  = set(files[i] for i in idx[ntest:ntest+nval])
    train = [f for f in files if f not in test_set and f not in val_set]
    val   = [f for f in files if f in val_set]
    test  = [f for f in files if f in test_set]
    return train, val, test

def read_numeric(df: pd.DataFrame, cols):
    used = [c for c in cols if c in df.columns]
    arr = df[used].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), used

# ----------------- pass 1: collect raw samples -----------------
def collect_samples(files, root, feature_cols):
    """返回样本列表，每个元素: dict(hist: [Ti,F], H: int, label: [H,2] or None, meta: dict)"""
    samples = []
    for f in tqdm(files, desc="scan csv", leave=False):
        in_df = pd.read_csv(f)
        if "player_to_predict" not in in_df.columns:
            continue
        in_df["player_to_predict"] = to_bool(in_df["player_to_predict"])
        tgt = in_df[in_df["player_to_predict"]].copy()
        if tgt.empty:
            continue

        # 输出文件（训练/验证可能有；测试可能没有）
        out_file = find_outputs(root, f)
        out_df = pd.read_csv(out_file) if os.path.exists(out_file) else None

        # 按 (game_id, play_id, nfl_id) 分组
        for (gid, pid, nid), g in tgt.groupby(["game_id","play_id","nfl_id"]):
            g = g.sort_values("frame_id")

            if g["num_frames_output"].isna().all():
                continue
            H = int(g["num_frames_output"].iloc[0])

            hist_np, used_cols = read_numeric(g, feature_cols)  # [Ti, F]
            if hist_np.shape[0] == 0:  # 没历史，跳
                continue

            label = None
            if out_df is not None:
                sub = out_df[(out_df["game_id"]==gid) & (out_df["play_id"]==pid) & (out_df["nfl_id"]==nid)].sort_values("frame_id")
                if not sub.empty:
                    lab = sub[["x","y"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)[:H]  # [<=H,2]
                    lab = np.nan_to_num(lab, nan=0.0, posinf=0.0, neginf=0.0)
                    if lab.shape[0] < H:  # 右侧 pad
                        pad = np.repeat(lab[-1:], H-lab.shape[0], axis=0) if lab.shape[0]>0 else np.zeros((H,2), np.float32)
                        lab = np.vstack([lab, pad])
                    label = lab  # [H,2]

            samples.append(dict(
                hist=hist_np, H=H, label=label,
                meta=dict(game_id=int(gid), play_id=int(pid), nfl_id=int(nid),
                          file=os.path.basename(f), features=used_cols)
            ))
    return samples

# ----------------- pad/unify -----------------
def unify_split(samples, lookback, infer_tin_when_zero=True):
    """返回统一后的张量 X, X_mask, Y, Y_mask, H, metas, feature_names"""
    if not samples:
        return None

    # 目标 T_in
    if lookback and lookback > 0:
        T_in = int(lookback)
    else:
        # 取该 split 内最大历史长度
        T_in = max(s["hist"].shape[0] for s in samples) if infer_tin_when_zero else 0

    # 目标 H_max
    H_max = max(int(s["H"]) for s in samples)

    F = samples[0]["hist"].shape[1]
    N = len(samples)

    X = np.zeros((N, T_in, F), np.float32)
    X_mask = np.zeros((N, T_in), np.float32)
    Y = np.zeros((N, H_max, 2), np.float32)
    Y_mask = np.zeros((N, H_max), np.float32)
    H_vec = np.zeros((N,), np.int64)
    metas = []

    for i, s in enumerate(samples):
        h = s["hist"]
        Ti = h.shape[0]
        # 右对齐：左侧补零，最新帧在最右边
        if Ti >= T_in:
            X[i] = h[-T_in:]
            X_mask[i] = 1.0
        else:
            X[i, -Ti:] = h
            X_mask[i, -Ti:] = 1.0

        H = int(s["H"]); H_vec[i] = H
        if s["label"] is not None:
            Y[i, :H] = s["label"][:H]
            Y_mask[i, :H] = 1.0
        else:
            # 测试没有标签
            pass

        metas.append(s["meta"])

    return dict(
        X=torch.from_numpy(X),            # [N, T_in, F]
        X_mask=torch.from_numpy(X_mask),  # [N, T_in]
        Y=torch.from_numpy(Y),            # [N, H_max, 2]
        Y_mask=torch.from_numpy(Y_mask),  # [N, H_max]
        H=torch.from_numpy(H_vec),        # [N]
        metas=metas,
        T_in=T_in, H_max=H_max,
        feature_names=samples[0]["meta"]["features"]
    )

# ----------------- normalization -----------------
def normalize_inplace(pack, mean=None, std=None):
    """只对 X 做标准化；mask==0 的位置不参与统计；返回 mean,std"""
    X = pack["X"].float()
    M = pack["X_mask"].float()
    # 统计（按特征）
    if mean is None or std is None:
        # 展平 batch 和时间，按 mask 过滤
        valid = M.unsqueeze(-1)  # [N,T,1]
        num = valid.sum(dim=(0,1))            # [1]
        num_feat = torch.clamp(num, min=1.0)  # 广播用不到，先占位
        # 特征维统计：对 mask 位置求和
        sum_feat = (X * valid).sum(dim=(0,1))           # [F]
        sqr_feat = (X * X * valid).sum(dim=(0,1))       # [F]
        mean = (sum_feat / torch.clamp(valid.sum(dim=(0,1)), min=1.0)).numpy()
        var = (sqr_feat / torch.clamp(valid.sum(dim=(0,1)), min=1.0)).numpy() - mean**2
        std = np.sqrt(np.clip(var, 1e-12, None))
    # 应用
    mean_t = torch.from_numpy(mean.astype(np.float32))
    std_t  = torch.from_numpy((std.astype(np.float32) + 1e-8))
    pack["X"] = ((X - mean_t) / std_t).float()
    return mean, std

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="包含 input_*.csv 与 output_*.csv 的目录")
    ap.add_argument("--outdir", required=True, help="保存目录")
    ap.add_argument("--features", nargs="+", required=True,
                    help="作为输入的特征列名（来自 input CSV）")
    ap.add_argument("--lookback", type=int, default=0, help="输入统一长度；0=按该split最大长度")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 找文件
    inputs = sorted(glob.glob(os.path.join(args.root, "input_*.csv")))
    if not inputs:
        raise FileNotFoundError(f"{args.root} 下没有 input_*.csv")
    train_files, val_files, test_files = split_files(inputs, args.val_ratio, args.test_ratio, args.seed)

    # 收集样本（两次：train/val/test 各一次），带进度条
    print("Collect train samples...")
    S_tr = collect_samples(train_files, args.root, args.features)
    print("Collect val samples...")
    S_va = collect_samples(val_files, args.root, args.features)
    print("Collect test samples...")
    S_te = collect_samples(test_files, args.root, args.features)

    # 统一维度
    print("Unify dims (train)...")
    P_tr = unify_split(S_tr, args.lookback)
    print("Unify dims (val)...")
    P_va = unify_split(S_va, args.lookback)
    print("Unify dims (test)...")
    P_te = unify_split(S_te, args.lookback)

    # 规范化：只用 train 的统计量
    mean = std = None
    if args.normalize and P_tr is not None:
        print("Normalize (fit on train)...")
        mean, std = normalize_inplace(P_tr, None, None)
        if P_va is not None:
            normalize_inplace(P_va, mean, std)
        if P_te is not None:
            normalize_inplace(P_te, mean, std)

    # 保存
    meta = dict(
        root=args.root,
        features=args.features,
        lookback=args.lookback,
        splits=dict(
            train=[os.path.basename(x) for x in train_files],
            val=[os.path.basename(x) for x in val_files],
            test=[os.path.basename(x) for x in test_files],
        ),
        stats=dict(
            train=len(S_tr), val=len(S_va), test=len(S_te)
        ),
        norm=dict(mean=None if mean is None else mean.tolist(),
                  std=None if std is None else std.tolist())
    )

    if P_tr: torch.save(P_tr, os.path.join(args.outdir, "train.pt"))
    if P_va: torch.save(P_va, os.path.join(args.outdir, "val.pt"))
    if P_te: torch.save(P_te, os.path.join(args.outdir, "test.pt"))
    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 打印形状
    def _shape(pack, name):
        if pack is None:
            print(f"{name}: 0 samples")
            return
        print(f"{name}: X{tuple(pack['X'].shape)}  Y{tuple(pack['Y'].shape)}  "
              f"T_in={pack['T_in']}  H_max={pack['H_max']}  N={pack['X'].shape[0]}")
    _shape(P_tr, "train")
    _shape(P_va, "val")
    _shape(P_te, "test")

if __name__ == "__main__":
    main()

# python data_process_norm.py --root train --outdir data_norm --features x y s a dir o absolute_yardline_number ball_land_x ball_land_y --lookback 0 --val_ratio 0.1 --test_ratio 0.1 --normalize
