# -*- coding: utf-8 -*-
"""对比: 原版10年回测(2016-10起)截取2020-01~2026-09  vs  2020窗口重跑版
口径: 两边月收益序列均为 open-to-open、后复权、单边万一
注意: 10年版贝叶斯校准/年度重训窗口与重跑版不同(训练样本随SIG_MIN前移),
      组合权重因此不同 —— 对比的是"同一策略体系, 不同校准窗口"的表现差异
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Microsoft YaHei"

def perf(r, dates):
    r = np.asarray(r, float)
    n = len(r)
    cum = np.cumprod(1 + r)
    ann = (1 + cum[-1] - 1) ** (12 / n) - 1
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min()
    return {"期数": n, "区间": f"{dates[0]}~{dates[-1]}", "累计收益": round(cum[-1]-1, 4),
            "年化收益": round(ann, 4), "夏普": round(sharpe, 3), "最大回撤": round(mdd, 4),
            "月胜率": round((r > 0).mean(), 3)}

# ---- 原版10年序列截取2020起
df10 = pd.read_csv(r"E:\quant-data\backtest\monthly_returns_paperv2.csv", encoding="utf-8-sig")
d10 = df10[df10["月份"] >= "2020-01"].reset_index(drop=True)
# ---- 2020重跑版
df20 = pd.read_csv(r"E:\quant-data\backtest\monthly_returns_paperv2_2020.csv", encoding="utf-8-sig")
d20 = df20[df20["月份"] >= "2020-01"].reset_index(drop=True)

assert (d10["月份"].values == d20["月份"].values).all(), "月份不对齐"
months = d10["月份"].tolist()
print(f"对比区间: {months[0]} ~ {months[-1]}  共{len(months)}个月\n")

rows = {}
for label, df in [("10年版(截取)", d10), ("2020重跑版", d20)]:
    for port in ["LGBv2_mw", "LGBv2_ew10"]:
        rows[f"{label}·{port}"] = perf(df[port].values, months)
rows["沪深300"] = perf(d10["沪深300"].values, months)

print(f"{'组合':22s} {'年化':>7s} {'夏普':>6s} {'回撤':>7s} {'胜率':>6s} {'累计':>7s}")
for k, v in rows.items():
    print(f"{k:22s} {v['年化收益']*100:6.1f}% {v['夏普']:6.2f} {v['最大回撤']*100:6.1f}% {v['月胜率']*100:5.1f}% {v['累计收益']*100:6.1f}%")

# 月度超额差(重跑 - 截取) 稳定性
for port in ["LGBv2_mw", "LGBv2_ew10"]:
    diff = d20[port].values - d10[port].values
    t = diff.mean() / diff.std() * np.sqrt(len(diff)) if diff.std() > 0 else np.nan
    print(f"\n{port}: 重跑-截取 月均差异 {diff.mean()*100:+.2f}%  (t={t:.2f})")

# ---- 对比图
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(months))
series = [("10年版截取·mw", d10["LGBv2_mw"].values, "tab:blue", "-"),
          ("2020重跑·mw", d20["LGBv2_mw"].values, "tab:cyan", "-"),
          ("10年版截取·ew10", d10["LGBv2_ew10"].values, "tab:red", "-"),
          ("2020重跑·ew10", d20["LGBv2_ew10"].values, "tab:orange", "-"),
          ("CSI 300", d10["沪深300"].values, "black", "--")]
for name, r, c, ls in series:
    ax.plot(x, np.cumprod(1 + np.nan_to_num(r, nan=0.0)), label=name, color=c, ls=ls, linewidth=1.4)
xt = list(range(0, len(months), 12))
ax.set_xticks(xt); ax.set_xticklabels([months[i] for i in xt], fontsize=8)
ax.set_title("同一策略体系对比: 10年版截取2020起 vs 2020窗口重跑 (fee 1bp/side)")
ax.set_ylabel("NAV"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
txt = "\n".join([f"{k}: 夏普{v['夏普']:.2f} 年化{v['年化收益']*100:.1f}%" for k, v in rows.items() if k != "沪深300"])
ax.text(0.985, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8e1", edgecolor="#e0c060", alpha=0.95))
fig.tight_layout()
out = r"E:\quant-data\backtest\compare_2020_window.png"
fig.savefig(out, dpi=130)
print(f"\n图已存 {out}")
