import torch

pack = torch.load("data_norm/train.pt")
X, X_mask = pack["X"], pack["X_mask"]          # [N, T_in, F], [N, T_in]
Y, Y_mask = pack["Y"], pack["Y_mask"]          # [N, H_max, 2], [N, H_max]
H = pack["H"]

print("== Shapes ==")
print("X:", X.shape, "X_mask:", X_mask.shape)
print("Y:", Y.shape, "Y_mask:", Y_mask.shape, "H:", H.shape)
print("T_in(meta):", pack["T_in"], "H_max(meta):", pack["H_max"])
print("F (feature dim):", X.shape[-1], "feature_names:", pack.get("feature_names"))

# 找到第一个“左侧非零”的样本
bad_idx = None
for i in range(X.shape[0]):
    ti = int(X_mask[i].sum().item())
    left = X[i, :X.shape[1]-ti]
    if left.numel() > 0 and not torch.allclose(left, torch.zeros_like(left)):
        bad_idx = i
        break

if bad_idx is None:
    print("没找到异常样本，说明左侧都是0（或全部样本Ti==T_in）。")
else:
    i = bad_idx
    ti = int(X_mask[i].sum().item())
    Tin = X.shape[1]
    print(f"\n=== 异常样本 idx={i} ===")
    print("Ti(有效历史长度) =", ti, "  左侧长度 =", Tin - ti)

    # 设置打印精度/阈值
    torch.set_printoptions(precision=5, sci_mode=False, edgeitems=3, linewidth=140)

    # 打印左侧补齐区的数值（只看前3行、前6个特征，避免太长）
    left = X[i, :Tin-ti, :]
    print("\n左侧补齐区(前3行, 前6个特征):")
    print(left[:3])

    # 打印左侧对应的 mask（应该都是0）
    print("\n左侧 X_mask：")
    print(X_mask[i, :Tin-ti])

    # 打印有效区前3步用于对比
    right = X[i, Tin-ti:Tin, :]
    print("\n有效区(最后Ti步中的前3行, 前6个特征):")
    print(right[:3])

    # 看看这些“左侧非零”是不是因为标准化导致
    # 计算左侧绝对值的最大/均值
    print("\n左侧区 |X| 的最大/均值：",
          left.abs().max().item() if left.numel() else 0.0,
          left.abs().mean().item() if left.numel() else 0.0)
