# build_dataset_xy.py
import os, glob, json, argparse
import numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional

# ------------------ 工具 ------------------
def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true","1","t","yes"])

def _out_csv(root: str, in_csv: str) -> str:
    return os.path.join(root, os.path.basename(in_csv).replace("input_", "output_"))

def _split_files(files: List[str], val=0.1, test=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(files)); rng.shuffle(idx)
    ntest, nval = int(len(files)*test), int(len(files)*val)
    test_set = set(files[i] for i in idx[:ntest])
    val_set  = set(files[i] for i in idx[ntest:ntest+nval])
    train = [f for f in files if f not in test_set and f not in val_set]
    val   = [f for f in files if f in val_set]
    test  = [f for f in files if f in test_set]
    return train, val, test

# ------------------ 读一对 input/output 生成样本 ------------------
def _load_pair(
    in_csv: str,
    out_csv: Optional[str],
    feat_cols: List[str],
    lookback: int,
) -> List[Dict]:
    """
    只保留 player_to_predict==True 的 (game_id,play_id,nfl_id) 组。
    hist: [Tin, Fin]  (来自 input 的特征)
    H:    num_frames_output
    label:[H, 2]      (来自 output 的 x,y；测试可为 None)
    """
    df = pd.read_csv(in_csv)
    if "player_to_predict" not in df.columns:
        raise ValueError(f"{in_csv} 缺少 player_to_predict 列")
    df["player_to_predict"] = _to_bool(df["player_to_predict"])

    tgt = df[df["player_to_predict"]].copy()
    if tgt.empty:
        return []

    # 必要列检查
    need = {"game_id","play_id","nfl_id","frame_id","num_frames_output"}
    miss = [c for c in (need | set(feat_cols)) if c not in df.columns]
    if miss:
        raise ValueError(f"{in_csv} 缺少必要列: {miss}")

    out_df = pd.read_csv(out_csv) if (out_csv and os.path.exists(out_csv)) else None
    if out_df is not None:
        for c in ["game_id","play_id","nfl_id","frame_id","x","y"]:
            if c not in out_df.columns:
                raise ValueError(f"{out_csv} 缺少列 {c}")

    samples = []
    for (gid,pid,nid), g in tgt.groupby(["game_id","play_id","nfl_id"]):
        g = g.sort_values("frame_id")

        # 历史特征 -> 数值
        feats = g[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        hist = feats if lookback <= 0 else feats[-lookback:]
        if hist.shape[0] == 0:
            continue

        # 各自的预测窗口
        if g["num_frames_output"].isna().all():
            raise ValueError(f"{in_csv} 该组缺少 num_frames_output: {(gid,pid,nid)}")
        H = int(g["num_frames_output"].iloc[0])

        # 标签 (x,y)
        label = None
        if out_df is not None:
            sub = out_df.query("game_id==@gid and play_id==@pid and nfl_id==@nid").sort_values("frame_id")
            if not sub.empty:
                lab = sub[["x","y"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)[:H]
                if lab.shape[0] < H:
                    # 右侧 pad（极少见）
                    pad = np.repeat(lab[-1:], H-lab.shape[0], axis=0)
                    lab = np.vstack([lab, pad])
                label = lab  # [H,2]

        samples.append(dict(
            meta=dict(game_id=int(gid), play_id=int(pid), nfl_id=int(nid),
                      file=os.path.basename(in_csv)),
            hist=hist,    # [Tin, Fin]
            H=H,          # int
            label=label   # [H,2] 或 None
        ))
    return samples

# ------------------ Dataset & collate（变长 padding + mask） ------------------
class PlayerXY(Dataset):
    def __init__(self, samples: List[Dict], normalize: bool=False, mean=None, std=None):
        self.samples = samples
        self.normalize = normalize
        if normalize:
            allf = np.concatenate([s["hist"] for s in samples], axis=0) if samples else np.zeros((1,1),np.float32)
            mean = allf.mean(axis=0) if mean is None else np.asarray(mean)
            std  = allf.std(axis=0)  if std  is None else np.asarray(std)
            std  = std + 1e-8
            self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
            for s in self.samples:
                s["hist"] = ((s["hist"] - self.mean) / self.std).astype(np.float32)
        else:
            self.mean, self.std = None, None

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        item = dict(
            hist=torch.from_numpy(s["hist"]),   # [Tin, Fin]
            H=torch.tensor(s["H"]).long(),
            meta=s["meta"]
        )
        item["label"] = None if s["label"] is None else torch.from_numpy(s["label"])  # [H,2]
        return item

def collate_xy(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    Tin_max = max(b["hist"].shape[0] for b in batch)
    Fin = batch[0]["hist"].shape[1]
    Hmax = max(int(b["H"]) for b in batch)

    hist = torch.zeros(B, Tin_max, Fin, dtype=torch.float32)
    hist_mask = torch.zeros(B, Tin_max, dtype=torch.float32)
    H_vec = torch.zeros(B, dtype=torch.long)
    metas = []

    has_label = all(b["label"] is not None for b in batch)
    label = torch.zeros(B, Hmax, 2, dtype=torch.float32) if has_label else None
    label_mask = torch.zeros(B, Hmax, dtype=torch.float32) if has_label else None

    for i,b in enumerate(batch):
        t = b["hist"].shape[0]
        hist[i, :t] = b["hist"]
        hist_mask[i, :t] = 1.0
        H = int(b["H"]); H_vec[i] = H
        metas.append(b["meta"])
        if has_label:
            L = b["label"].shape[0]
            label[i, :L] = b["label"]
            label_mask[i, :L] = 1.0

    return dict(hist=hist, hist_mask=hist_mask, H=H_vec,
                label=label, label_mask=label_mask, metas=metas)

# ------------------ 主流程 ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="包含 input_*.csv 与 output_*.csv 的目录，例如 kaggle-10/train")
    ap.add_argument("--outdir", default="learn_rnn/processed_xy")
    # 默认只用数值列；如果你想“全部列都当特征”，请把字符串列先数值化后再传进来
    ap.add_argument("--features", nargs="+",
                    default=["x","y","s","a","dir","o","absolute_yardline_number","ball_land_x","ball_land_y"])
    ap.add_argument("--lookback", type=int, default=0, help="0=用全部历史；>0=只取最后 L 帧")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    inputs = sorted(glob.glob(os.path.join(args.root, "input_*.csv")))
    if not inputs:
        raise FileNotFoundError(f"{args.root} 下没有 input_*.csv")

    tr_files, va_files, te_files = _split_files(inputs, args.val_ratio, args.test_ratio, args.seed)

    def build(files):
        res=[]
        for f in files:
            out_csv = _out_csv(args.root, f)
            try:
                res += _load_pair(f, out_csv, args.features, args.lookback)
            except Exception as e:
                print(f"[SKIP] {os.path.basename(f)}: {e}")
        return res

    S_tr, S_va, S_te = build(tr_files), build(va_files), build(te_files)

    # 保存样本
    torch.save(S_tr, os.path.join(args.outdir, "train.pt"))
    torch.save(S_va, os.path.join(args.outdir, "val.pt"))
    torch.save(S_te, os.path.join(args.outdir, "test.pt"))

    # 保存 meta
    meta = dict(
        root=args.root,
        features=args.features,
        lookback=args.lookback,
        splits=dict(
            train=[os.path.basename(x) for x in tr_files],
            val=[os.path.basename(x) for x in va_files],
            test=[os.path.basename(x) for x in te_files],
        ),
        stats=dict(train=len(S_tr), val=len(S_va), test=len(S_te))
    )
    json.dump(meta, open(os.path.join(args.outdir,"meta.json"),"w"), ensure_ascii=False, indent=2)
    print("Saved:", args.outdir, "| stats:", meta["stats"])

    # 演示 DataLoader（可删）
    if S_tr:
        ds = PlayerXY(S_tr, normalize=args.normalize)
        dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_xy)
        b = next(iter(dl))
        print("hist", b["hist"].shape, "hist_mask", b["hist_mask"].shape)
        if b["label"] is not None:
            print("label", b["label"].shape, "label_mask", b["label_mask"].shape)

if __name__ == "__main__":
    main()

# 目录：
# kaggle-10/train/
#   input_2023_w01.csv
#   output_2023_w01.csv
#   input_2023_w02.csv
#   output_2023_w02.csv
#   ...

# python data_process.py --root train --outdir data --features x y s a dir o absolute_yardline_number ball_land_x ball_land_y --lookback 0  --val_ratio 0.1 --test_ratio 0.1 --normalize
