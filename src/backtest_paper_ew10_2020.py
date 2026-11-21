# -*- coding: utf-8 -*-
"""论文复现 · 2020起窗口版：论文温和参数 + 仅 Top10 等权(ew10)组合
================================================================
因子: 11因子(or_yoy/np_yoy/roa/ps/pcf/ep/turn20/std60/revs20/accrual/gpm)
模型: LightGBM 二分类(论文 Table 5 温和最优参数, 不跑贝叶斯优化)
标签: 前向12月后复权收益符号(论文口径)
组合: 仅 Top10 等权, 每3个月调仓(原文60交易日形式)
窗口: 执行 2020-01 起; 首个模型相位对齐全窗口版(2019-10重训), 每年10月重训
纪律: PIT(首披版f_ann_date + asof) / 行业历史成分 / 剔ST·北交所·次新·停牌
      / 一字涨停不可买 / winsorize 1%-99% → z-score / t+1开盘成交 / 单边万一
数据: 全部本地库(CH quant / PG quantdb), 无外部调用
================================================================
运行: E:\\quant-data\\venv\\Scripts\\python.exe backtest_paper_ew10.py
"""
import sys, os, json, time, warnings, datetime, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
import polars as pl
import backtest_v3 as V3

# ---- 配置 ----
FEAT = ["or_yoy", "np_yoy", "roa", "ps", "pcf",
        "ep", "turn20", "std60", "revs20",
        "accrual", "gpm"]
SIG_MIN = "2020-01-01"
FEE_SIDE = 1e-4
TOPN_EW = 10
EW_REBAL_EVERY = 3          # 月(≈原文60交易日)
TRAIN_M = 12
LABEL_LAG = 13              # 标签前向12月须完整实现(j+12<i)
TAG = "paperv2_ew10_2020"

# 论文 Table 5 温和最优参数(固定, 不跑贝叶斯优化)
BEST = {"max_depth": 10, "n_estimators": 163, "subsample": 0.764,
        "min_split_gain": 0.266, "reg_alpha": 1.0, "reg_lambda": 1.451}

V3.FACTORS = FEAT
t0 = time.time()

def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

# ---------------------------------------------------------------- 数据
log("加载数据...")
data = V3.load_all()

# 论文补充财务: cogs_ttm + np_yoy + ocf_ttm
def load_paper_fund_extra():
    pg = V3._pg()
    inc = pl.read_database(
        "SELECT symbol, end_date, f_ann_date, total_cogs, n_income_attr_p "
        "FROM fin.income_ts WHERE end_date >= '2009-01-01'", pg)
    pg.close()
    inc = (inc.sort(["symbol", "end_date", "f_ann_date"])
              .group_by(["symbol", "end_date"], maintain_order=True)
              .first()
              .sort(["symbol", "end_date"]))
    inc = inc.with_columns([
        pl.col("total_cogs").shift(1).over("symbol", pl.col("end_date").dt.year()).alias("_cogs_prev"),
        pl.col("n_income_attr_p").shift(1).over("symbol", pl.col("end_date").dt.year()).alias("_np_prev"),
    ]).with_columns([
        (pl.col("end_date").dt.quarter() == 1).alias("_q1"),
    ]).with_columns([
        pl.when(pl.col("_q1")).then(pl.col("total_cogs"))
          .otherwise(pl.col("total_cogs") - pl.col("_cogs_prev")).alias("cogs_q"),
        pl.when(pl.col("_q1")).then(pl.col("n_income_attr_p"))
          .otherwise(pl.col("n_income_attr_p") - pl.col("_np_prev")).alias("np_q"),
    ]).with_columns([
        pl.col("cogs_q").rolling_sum(4).over("symbol").alias("cogs_ttm"),
        pl.col("np_q").rolling_sum(4).over("symbol").alias("np_ttm2"),
    ]).with_columns([
        (pl.col("np_ttm2") / pl.col("np_ttm2").shift(4).over("symbol") - 1).alias("np_yoy"),
    ])
    pg2 = V3._pg()
    cf = pl.read_database(
        "SELECT symbol, end_date, f_ann_date, n_cashflow_act "
        "FROM fin.cashflow_ts WHERE end_date >= '2009-01-01'", pg2)
    pg2.close()
    cf = (cf.sort(["symbol", "end_date", "f_ann_date"])
            .group_by(["symbol", "end_date"], maintain_order=True).first()
            .sort(["symbol", "end_date"]))
    cf = cf.with_columns([
        pl.col("n_cashflow_act").shift(1).over("symbol", pl.col("end_date").dt.year()).alias("_ocf_prev"),
        (pl.col("end_date").dt.quarter() == 1).alias("_q1b"),
    ]).with_columns([
        pl.when(pl.col("_q1b")).then(pl.col("n_cashflow_act"))
          .otherwise(pl.col("n_cashflow_act") - pl.col("_ocf_prev")).alias("ocf_q"),
    ]).with_columns([
        pl.col("ocf_q").rolling_sum(4).over("symbol").alias("ocf_ttm"),
    ])
    inc = inc.join(cf.select(["symbol", "end_date", "ocf_ttm"]),
                   on=["symbol", "end_date"], how="left")
    return inc.select(["symbol", "end_date", "cogs_ttm", "np_yoy", "ocf_ttm"])

extra = load_paper_fund_extra()
fund = (data["fund_report"]
        .join(extra, on=["symbol", "end_date"], how="left")
        .sort(["symbol", "end_date"]))

log("构建面板...")
panel = V3.build_panel(data)
panel = V3.attach_industry(panel, data["sw"])
val = data["valuation"].select(["symbol", "trade_date", "pe_ttm", "total_mv"])
panel = panel.join(val, on=["symbol", "trade_date"], how="left")

# ---------------------------------------------------------------- 标签
cal_dates, sig_dates = V3.get_signal_dates(panel)
cal = [str(x) for x in cal_dates]
months_all = [str(sd) for sd in sig_dates]
lab_m = V3.build_labels(cal_dates, sig_dates, panel)   # 月频执行标签

log("构建论文标签(前向12月收益符号)...")
sig_idx = {sd: cal.index(sd) for sd in months_all}
po = panel.select(["symbol", "trade_date", "hfq_open"])
lf = []
for i in range(len(months_all) - TRAIN_M):
    sd = months_all[i]
    j = i + TRAIN_M
    if j >= len(months_all):
        break
    sd_out = months_all[j]
    if sig_idx[sd] + 1 >= len(cal_dates) or sig_idx[sd_out] + 1 >= len(cal_dates):
        break
    e_in = cal_dates[sig_idx[sd] + 1]
    e_out = cal_dates[sig_idx[sd_out] + 1]
    left = panel.filter(pl.col("trade_date") == datetime.date.fromisoformat(sd)).select("symbol")
    l1 = po.filter(pl.col("trade_date") == e_in).rename({"hfq_open": "px_in"})
    l2 = po.filter(pl.col("trade_date") == e_out).rename({"hfq_open": "px_out"})
    lab = (left.join(l1, on="symbol", how="left").join(l2, on="symbol", how="left")
           .with_columns([
               pl.lit(sd).alias("sig_date"),
               (pl.col("px_out") / pl.col("px_in") - 1).alias("ret_fwd12"),
               (pl.col("px_out") > pl.col("px_in")).cast(pl.Int8).alias("y12"),
           ]))
    lf.append(lab)
lab12 = pl.concat(lf)
log(f"  12月标签 {lab12.shape[0]:,} 行, 正例占比 {lab12['y12'].mean():.3f}")

# ---------------------------------------------------------------- 截面
FUND_COLS = ["symbol", "end_date", "rev_ttm", "np_ttm", "or_yoy_ttm",
             "roa_ttm", "ocf_ttm", "cogs_ttm", "np_yoy",
             "accrual_ttm", "gpm_ttm"]
fund_rep = fund.select(FUND_COLS)

def make_cs(sd):
    f = panel.filter(pl.col("trade_date") == sd)
    f = f.filter(
        (pl.col("volume") > 0)
        & (~pl.col("is_st"))
        & (~pl.col("symbol").str.starts_with("4"))
        & (~pl.col("symbol").str.starts_with("8"))
        & (pl.col("list_days") >= 370)
        & pl.col("hfq_close_lag60").is_not_null())
    f = (f.sort("symbol").join_asof(
        fund_rep.rename({"end_date": "eff_d"}).sort(["symbol", "eff_d"]),
        left_on="trade_date", right_on="eff_d", by="symbol", strategy="backward"))
    f = f.with_columns([
        (pl.col("total_mv") * 1e4 / pl.col("rev_ttm")).alias("ps"),
        pl.when(pl.col("ocf_ttm") > 0)
          .then(pl.col("total_mv") * 1e4 / pl.col("ocf_ttm")).otherwise(None).alias("pcf"),
        pl.col("roa_ttm").alias("roa"),
        pl.col("or_yoy_ttm").alias("or_yoy"),
        pl.col("np_yoy"),
        pl.col("accrual_ttm").alias("accrual"),
        pl.col("gpm_ttm").alias("gpm"),
        (1.0 / pl.col("pe_ttm")).alias("ep"),
    ])
    out = {"symbol": f["symbol"].to_numpy()}
    for k in FEAT:
        col = f[k].to_numpy().astype(float)
        w = V3.winsorize(col)
        m = np.isfinite(w)
        z = np.full(len(col), np.nan)
        if m.sum() > 30 and w[m].std() > 0:
            z[m] = (w[m] - w[m].mean()) / w[m].std()
        out["z_" + k] = z
    for c in ["trade_date", "open", "close", "hfq_open", "hfq_close",
              "volume", "zt_onezi", "dt_onezi", "pct_chg", "mktcap", "circmv"]:
        out[c] = f[c].to_numpy()
    return pl.DataFrame(out)

cs = {}
for i, (sd, sd_dt) in enumerate(zip(months_all, sig_dates)):
    cs[sd] = make_cs(sd_dt)
    if (i + 1) % 24 == 0:
        print(f"  截面 {i+1}/{len(months_all)} ({sd}) n={cs[sd].shape[0]}")

# ---------------------------------------------------------------- 训练窗
def collect_train(i):
    Xl, yl = [], []
    for j in range(i - TRAIN_M - LABEL_LAG + 1, i - LABEL_LAG + 1):
        sjd = months_all[j]
        if sjd not in cs:
            continue
        fj = cs[sjd]
        lj = lab12.filter(pl.col("sig_date") == sjd)
        jj = fj.join(lj.select(["symbol", "y12"]), on="symbol", how="inner")
        Xj = np.column_stack([jj["z_" + k].to_numpy() for k in FEAT])
        yj = jj["y12"].to_numpy()
        mm = np.isfinite(yj) & np.isfinite(Xj).all(axis=1)
        if mm.sum() > 100:
            Xl.append(Xj[mm]); yl.append(yj[mm])
    if not Xl:
        return None, None
    return np.vstack(Xl), np.concatenate(yl)

# ---------------------------------------------------------------- 回测(仅 ew10)
from lightgbm import LGBMClassifier
from scipy import stats
log(f"回测开始(论文温和参数 {BEST}, 仅 ew10)...")

holdings = []
port_ret = []
port_dates, bench_ret = [], []
prev_w = {}
last_ew_rebal = -99
fitted = None
ic_log = []
i0 = next(i for i, sd in enumerate(months_all) if sd >= SIG_MIN)
exec_log = []
bench_open = {str(r0[0]): r0[1] for r0 in data["bench"].iter_rows()}

for i, sd in enumerate(months_all):
    if sd < SIG_MIN:
        continue
    if sd not in cs or cs[sd].shape[0] < 500:
        continue
    f = cs[sd]
    syms = f["symbol"].to_list()
    if i < TRAIN_M + LABEL_LAG:
        continue
    if fitted is None:
        # 相位对齐: 首个模型等效全窗口(2016起)版在2019-10重训的模型(训练窗 2017-10..2018-09)
        Xtr, ytr = collect_train(i - 3)
        _trained = True
    elif (i - i0) % 12 == 9:
        # 每年10月重训, 与全窗口版保持同一重训相位(2019-10, 2020-10, ...)
        Xtr, ytr = collect_train(i)
        _trained = True
    else:
        _trained = False
    if _trained:
        if Xtr is not None and len(ytr) > 500:
            fitted = LGBMClassifier(learning_rate=0.05, n_jobs=-1, random_state=42,
                                    verbose=-1, **BEST).fit(Xtr, ytr)
            print(f"  [{sd}] 重训 样本={len(ytr):,} 正例={ytr.mean():.3f}")
    if fitted is None:
        continue
    Z = np.column_stack([f["z_" + k].to_numpy() for k in FEAT])
    pred = fitted.predict_proba(np.nan_to_num(Z, nan=np.nan))[:, 1]
    lab_cur = lab_m.filter(pl.col("sig_date") == sd)
    if lab_cur.shape[0] == 0:
        continue
    exe = f.join(lab_cur, on="symbol", how="inner")
    # 月度 Rank IC(预测概率 vs 下月前向收益)
    _lj = f.join(lab_cur.select(["symbol", "label"]), on="symbol", how="left")
    _pv = _lj["label"].to_numpy()
    _m = np.isfinite(_pv) & np.isfinite(pred)
    if _m.sum() > 30:
        ic_log.append((sd, float(stats.spearmanr(pred[_m], _pv[_m]).statistic)))
    # Top10 等权, 每3月调仓
    if i - last_ew_rebal >= EW_REBAL_EVERY:
        top10 = [syms[j] for j in np.argsort(-pred)[:TOPN_EW]]
        prev_w = {s: 1.0 / TOPN_EW for s in top10}
        last_ew_rebal = i
    w10 = prev_w
    r = V3.realize(exe, list(w10), w10, sd, exec_log)
    port_ret.append(r)   # 等权持有期无换手费
    holdings.append({"sig": sd, "rebalanced": i == last_ew_rebal,
                     "w": {s: round(w, 4) for s, w in w10.items()}})
    # 基准
    e_i = cal[cal.index(sd) + 1]
    e_next = None
    if i + 1 < len(months_all) and cal.index(months_all[i + 1]) + 1 < len(cal):
        e_next = cal[cal.index(months_all[i + 1]) + 1]
    if e_next and str(e_i) in bench_open and str(e_next) in bench_open:
        bench_ret.append(bench_open[str(e_next)] / bench_open[str(e_i)] - 1)
    else:
        bench_ret.append(np.nan)
    port_dates.append(sd)

# ---------------------------------------------------------------- 输出
log("写结果...")
out_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(out_dir, f"monthly_returns_{TAG}.csv"), "w", newline="", encoding="utf-8-sig") as fp:
    w = csv.writer(fp)
    w.writerow(["月份", "沪深300", "ew10", "ew10_超额"])
    for i2, d in enumerate(port_dates):
        w.writerow([d[:7], round(bench_ret[i2], 5),
                    round(port_ret[i2], 5),
                    round(port_ret[i2] - bench_ret[i2], 5)])
with open(os.path.join(out_dir, f"holdings_detailed_{TAG}.json"), "w", encoding="utf-8") as fp:
    json.dump(holdings, fp, ensure_ascii=False, indent=1, default=V3._np_json_default)

stat = V3.perf_stats(port_ret, port_dates)
bench_stat = V3.perf_stats(bench_ret, port_dates)

# Rank IC 汇总
ic_arr = np.array([x[1] for x in ic_log]) if ic_log else np.array([np.nan])
ic_mean = float(np.nanmean(ic_arr))
ic_std = float(np.nanstd(ic_arr)) if len(ic_arr) > 1 else np.nan
ic_ir = ic_mean / ic_std if ic_std and ic_std > 0 else float("nan")
ic_t = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std and ic_std > 0 else float("nan")
ic_pos = float((ic_arr > 0).mean()) if len(ic_arr) else float("nan")

print("\n================ 组合绩效 (ew10, 论文温和参数) ================")
print("ew10     ", stat)
print("沪深300  ", bench_stat)
print(f"RankIC  均值={ic_mean:.4f}  ICIR={ic_ir:.3f} t={ic_t:.2f}  >0占比={ic_pos:.3f} (n={len(ic_arr)})")

with open(os.path.join(out_dir, f"backtest_result_{TAG}.json"), "w", encoding="utf-8") as fp:
    json.dump({"组合绩效": {"ew10": stat}, "沪深300": bench_stat,
               "RankIC": {"IC均值": round(ic_mean, 4), "ICIR": round(ic_ir, 3),
                          "t值": round(ic_t, 2), ">0占比": round(ic_pos, 3),
                          "期数": len(ic_arr)},
               "月度IC序列": [[d[:7], round(v, 4)] for d, v in ic_log],
               "参数": BEST, "特征集": FEAT,
               "组合月收益": [round(x, 5) for x in port_ret],
               "基准月收益": [round(x, 5) for x in bench_ret],
               "月份": port_dates}, fp, ensure_ascii=False, indent=2, default=V3._np_json_default)

# ---------------------------------------------------------------- 画图 (ew10 vs 沪深300)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Microsoft YaHei"

r_p = np.array(port_ret, float)
r_b = np.array(bench_ret, float)
wp = np.cumprod(1 + r_p)
wb = np.cumprod(1 + r_b)
xs = list(range(len(port_dates)))
xt = [d[:7] for d in port_dates]
step = max(1, len(xs) // 12)
fig, ax = plt.subplots(figsize=(11.5, 6))
ax.plot(xs, wp, label=f"ew10 (Top10等权, 季调, 论文温和参数)", lw=1.6, color="#d62728")
ax.plot(xs, wb, label="沪深300", lw=1.4, color="#1f77b4", alpha=0.85)
ax.set_xticks(xs[::step]); ax.set_xticklabels([xt[i] for i in xs[::step]], rotation=45, fontsize=8)
ax.set_title(f"paper-v2 ew10 (paper hyperparameters, no bayes)  {port_dates[0][:7]} ~ {port_dates[-1][:7]},  单边万一", fontsize=12)
ax.set_ylabel("净值(起点=1)")
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
txt = (f"夏普 {stat['夏普']:.2f}   年化 {stat['年化收益']*100:.1f}%   回撤 {stat['最大回撤']*100:.1f}%\n"
       f"IC {ic_mean:.4f}   ICIR {ic_ir:.2f}\n"
       f"沪深300 夏普 {bench_stat['夏普']:.2f}  年化 {bench_stat['年化收益']*100:.1f}%")
ax.text(0.985, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=0.9))
fig.tight_layout()
fig.savefig(os.path.join(out_dir, f"wealth_{TAG}.png"), dpi=130)
print(f"[图] wealth_{TAG}.png 已保存")
print(f"\n[done] 总耗时 {time.time()-t0:.0f}s")
