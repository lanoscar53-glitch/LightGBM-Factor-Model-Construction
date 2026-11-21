# -*- coding: utf-8 -*-
"""论文复现V2-2020窗口重跑(扩展因子集+收益集中度治理):
与 backtest_paper_v2.py 逻辑完全一致, 仅回测窗口改为 2020-01~2026-10
(训练数据仍用窗口之前的历史, 年度重训+标签滞后13月, 无前视):
1. 特征扩展: 剔除论文FM不显著因子(pe/eps/ps_sales/oi/tc/npf/sharpe12/md12),
   保留 |NW_t|>=1.5 的 or_yoy/np_yoy/roa/ps/pcf, 并入我们16因子体系中显著的
   ep/turn20/std60/revs20/accrual/gpm → 共11因子
2. 收益集中度治理(诊断发现ew10超额收益top1月2024-09占+25.2%, top5月>100%对数贡献):
   MW风格盒约束加入 beta250(250日对沪深300beta, 截面z) — 直接钳制beta押注
   (标签严格遵循原论文: 前向12个月后复权收益符号, 不用超额标签 — 用户指定)
3. 新增: 月度Rank IC(预测概率 vs 下月open-to-open后复权收益, Spearman)
其余设定与V2完全一致: LightGBM+贝叶斯200迭代, 年度重训+标签滞后13月(防前视),
Top20 MW月调(阈值0.5%) + Top10等权季调(原文形式), 全A剔ST/北交所/次新/停牌,
t+1开盘成交, 后复权, 单边万一, 窗口2020-01~2026-10
"""
import sys, os, json, time, warnings, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
import polars as pl
import backtest_v3 as V3

# V2 扩展特征集: 论文保留(|NW_t|>=1.5) + 我们16因子显著项 → 11因子
FEAT = ["or_yoy", "np_yoy", "roa", "ps", "pcf",          # 论文保留
        "ep", "turn20", "std60", "revs20",               # 量价/价值(16因子体系显著)
        "accrual", "gpm"]                                # 质量(16因子体系显著)
STYLE = ["size", "ep", "revs20", "std60", "turn20", "beta250"]   # MW盒约束风格(+beta钳制)
SIG_MIN = "2020-01-01"
FEE_SIDE = 1e-4
REBAL_THR = 0.005
TOPN_MW = 20
TOPN_EW = 10
EW_REBAL_EVERY = 3                                            # 月, ≈原文60交易日
TRAIN_M = 12
TAG = "paperv2_2020"
t0 = time.time()

# 约束参数注入v3模块(pick_markowitz_constrained读模块全局)
V3.IND_LIM = 0.10
V3.STYLE_LIM = 0.35
V3.STOCK_CAP = 0.12
V3.CAND_N = 120
V3.FACTORS = FEAT                                             # factor_mimicking_returns用

# ---------------------------------------------------------------- 1. 数据
print("=" * 60)
print("[paper] 论文复现V2-2020窗口: LightGBM-Bayes 11因子选股 (2020-01~2026-10)")
print("=" * 60)
data = V3.load_all()

# 1a. paper补充财务: cogs_ttm(营业总成本) + np_yoy(净利同比) —— 其余复用fund_report
def load_paper_fund_extra():
    print("[fund] 补拉 cogs_ttm / np_yoy (PG income_ts, 首披版PIT) ...")
    pg = V3._pg()
    inc = pl.read_database(
        "SELECT symbol, end_date, f_ann_date, total_cogs, n_income_attr_p "
        "FROM fin.income_ts WHERE end_date >= '2009-01-01'", pg)
    pg.close()
    inc = (inc.sort(["symbol", "end_date", "f_ann_date"])
              .group_by(["symbol", "end_date"], maintain_order=True)
              .first())                                   # 首披版本(防前视)
    inc = inc.sort(["symbol", "end_date"])
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
    # 经营现金流TTM(PCF因子用, fund_report未输出该列)
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
    ]).with_columns([
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

# 1b. v3日面板 + 估值列 + Sharpe12/MD12 (250日后复权)
print("[panel] 构建日面板(v3复用) + 论文风险因子 ...")
panel = V3.build_panel(data)
panel = V3.attach_industry(panel, data["sw"])              # PIT申万行业(l2_code)
val = data["valuation"].select(["symbol", "trade_date", "pe_ttm", "total_mv"])
panel = panel.join(val, on=["symbol", "trade_date"], how="left")
panel = (panel.sort(["symbol", "trade_date"])
    .with_columns((pl.col("hfq_close") / pl.col("hfq_close").shift(1) - 1).alias("_dret"))
    .with_columns([
        pl.col("_dret").rolling_mean(250).over("symbol").alias("_m"),
        pl.col("_dret").rolling_std(250).over("symbol").alias("_s"),
        pl.col("hfq_close").rolling_max(250).over("symbol").alias("_cmax"),
    ]).with_columns([
        (pl.col("_m") / pl.col("_s") * np.sqrt(250)).alias("sharpe12"),
        (pl.col("hfq_close") / pl.col("_cmax") - 1).alias("_dd"),
    ]).with_columns(
        pl.col("_dd").rolling_min(250).over("symbol").alias("md12"))
    .drop(["_m", "_s", "_cmax", "_dd"]).with_columns(pl.col("_dret").alias("_x")))

# 1b+. beta250: 250日滚动对沪深300beta(收益集中度治理: MW风格盒约束用)
print("[panel] 计算 beta250 (250日滚动 vs 沪深300) ...")
_bch = data["bench"].sort("trade_date").with_columns(
    (pl.col("close") / pl.col("close").shift(1) - 1).alias("_mkt"))
panel = panel.join(_bch.select(["trade_date", "_mkt"]), on="trade_date", how="left")
_W = 250
panel = panel.with_columns([
    (pl.col("_x") * pl.col("_mkt")).rolling_sum(_W).over("symbol").alias("_Sxm"),
    pl.col("_x").rolling_sum(_W).over("symbol").alias("_Sx"),
    pl.col("_mkt").rolling_sum(_W).over("symbol").alias("_Sm2"),
    (pl.col("_mkt") * pl.col("_mkt")).rolling_sum(_W).over("symbol").alias("_Smm2"),
    pl.col("_mkt").is_not_null().cast(pl.Int32).rolling_sum(_W).over("symbol").alias("_nb"),
]).with_columns([
    pl.when((pl.col("_nb") >= 200) & (pl.col("_Smm2") - pl.col("_Sm2") ** 2 / _W > 1e-16))
      .then((_W * pl.col("_Sxm") - pl.col("_Sx") * pl.col("_Sm2")) /
            (_W * pl.col("_Smm2") - pl.col("_Sm2") ** 2))
      .otherwise(None).alias("beta250"),
]).drop(["_x", "_mkt", "_Sxm", "_Sx", "_Sm2", "_Smm2", "_nb"])

# 1c. 标签: ①月频open-to-open(执行/对照FM用, 复用v3) ②前向12月符号(论文训练标签)
cal_dates, sig_dates = V3.get_signal_dates(panel)           # Date对象(panel列类型匹配)
cal = [str(x) for x in cal_dates]
months_all = [str(sd) for sd in sig_dates]
lab_m = V3.build_labels(cal_dates, sig_dates, panel)        # 月频: px_in/label/zt_in

print("[label] 构建论文标签: 前向12月后复权收益符号(原论文口径) ...")
sig_idx = {sd: cal.index(sd) for sd in months_all}
po = panel.select(["symbol", "trade_date", "hfq_open"])
lf = []
for i in range(len(months_all) - TRAIN_M):                  # 需前向12个月数据
    sd = months_all[i]
    j = i + TRAIN_M
    if j >= len(months_all):
        break
    sd_out = months_all[j]
    if sig_idx[sd] + 1 >= len(cal_dates) or sig_idx[sd_out] + 1 >= len(cal_dates):
        break                                                  # 日历末尾无t+1开盘价
    e_in = cal_dates[sig_idx[sd] + 1]; e_out = cal_dates[sig_idx[sd_out] + 1]
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
print(f"  12月标签覆盖: {lab12.shape[0]:,} 行, 正样本占比 {lab12['y12'].mean():.3f}")

# ---------------------------------------------------------------- 2. 截面
print("[cs] 生成信号日截面(全A剔ST/北交所/新股/停牌) + 论文11因子z值 ...")
FUND_COLS = ["symbol", "end_date", "rev_ttm", "np_ttm", "or_yoy_ttm",
             "roa_ttm", "ocf_ttm", "cogs_ttm", "np_yoy",
             "accrual_ttm", "gpm_ttm"]                       # V2: +accrual/gpm
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
    # PIT asof: 最新已公告报告(ann+1口径由首披版end_date保证)
    f = (f.sort("symbol").join_asof(
        fund_rep.rename({"end_date": "eff_d"}).sort(["symbol", "eff_d"]),
        left_on="trade_date", right_on="eff_d", by="symbol", strategy="backward"))
    # 11因子原始值(V2扩展集: 论文保留5 + 我们16因子显著6)
    f = f.with_columns([
        (pl.col("total_mv") * 1e4 / pl.col("rev_ttm")).alias("ps"),
        pl.when(pl.col("ocf_ttm") > 0)
          .then(pl.col("total_mv") * 1e4 / pl.col("ocf_ttm")).otherwise(None).alias("pcf"),
        pl.col("roa_ttm").alias("roa"),
        pl.col("or_yoy_ttm").alias("or_yoy"),
        pl.col("np_yoy"),
        pl.col("accrual_ttm").alias("accrual"),
        pl.col("gpm_ttm").alias("gpm"),
    ])
    # 风格/量价因子: ep(估值) + turn20/std60/revs20/beta250(面板已有)
    f = f.with_columns([
        (1.0 / pl.col("pe_ttm")).alias("ep"),
    ])
    out = {"symbol": f["symbol"].to_numpy(),
           "industry": f["l2_code"].fill_null("NA").to_numpy()}
    for k in FEAT + STYLE:
        col = f[k].to_numpy().astype(float)
        w = V3.winsorize(col)                                # 1%/99%缩尾
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

# ---------------------------------------------------------------- 3. 模型对比+贝叶斯优化
LABEL_LAG = 13   # PIT铁律: 12月前向标签须完整实现才可用于训练 → 训练月j需满足 j+12 < i(即j≤i-13)
                 # (直接月频重训[i-12,i)会让最近11个月标签覆盖未来=前视泄漏; 论文为年度步进故无此问题)

def collect_train(i):
    """训练窗取最近12个标签已完整实现的月份: [i-12-LABEL_LAG, i-LABEL_LAG)
    标签=前向12月后复权收益符号(原论文口径)"""
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
        mm = np.isfinite(yj) & np.isfinite(Xj).all(axis=1)   # 训练去缺失(原文缺失值处理)
        if mm.sum() > 100:
            Xl.append(Xj[mm]); yl.append(yj[mm])
    if not Xl:
        return None, None
    return np.vstack(Xl), np.concatenate(yl)

def metrics(y_true, y_prob, y_pred):
    from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
    return {"Precision": round(precision_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "AUC": round(roc_auc_score(y_true, y_prob), 4)}

i0 = next(i for i, sd in enumerate(months_all) if sd >= SIG_MIN)
X0, y0 = collect_train(i0)
print(f"[model] 首窗训练集: {X0.shape[0]:,} 样本 × {X0.shape[1]} 因子, 正例占比 {y0.mean():.3f}")

# 3a. 三算法对比(原文Table 2, 默认参数)
print("[model] RF / XGB / LightGBM 默认参数对比 ...")
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
cv = StratifiedKFold(5, shuffle=True, random_state=42)
model_cmp = {}
for name, mdl in [("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
                  ("XGBoost", XGBClassifier(n_estimators=200, max_depth=6, n_jobs=-1, random_state=42, verbosity=0)),
                  ("LightGBM", LGBMClassifier(n_estimators=200, n_jobs=-1, random_state=42, verbose=-1))]:
    prob = cross_val_predict(mdl, X0, y0, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    pred = (prob > 0.5).astype(int)
    model_cmp[name] = metrics(y0, prob, pred)
    print(f"  {name:14s} {model_cmp[name]}")

# 3b. LightGBM贝叶斯优化(原文Table 5范围, 平均P/R/Acc/AUC为目标)
print("[model] LightGBM 贝叶斯优化 (skopt gp_minimize, 200 calls, 以原文最优值起步) ...")
from skopt import gp_minimize
from skopt.space import Integer, Real
SPACE = [Integer(1, 20, name="max_depth"),
         Integer(1, 400, name="n_estimators"),
         Real(0.4, 1.0, name="subsample"),
         Real(0.0, 1.0, name="min_split_gain"),
         Real(0.005, 30.0, name="reg_alpha"),
         Real(0.005, 30.0, name="reg_lambda")]
X_tr_s, X_te_s, y_tr_s, y_te_s = (None,) * 4
def bayes_obj(p):
    mdl = LGBMClassifier(max_depth=int(p[0]), n_estimators=int(p[1]), subsample=p[2],
                         min_split_gain=p[3], reg_alpha=p[4], reg_lambda=p[5],
                         learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1)
    prob = cross_val_predict(mdl, X0, y0, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    pred = (prob > 0.5).astype(int)
    from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
    return -(precision_score(y0, pred) + recall_score(y0, pred)
             + accuracy_score(y0, pred) + roc_auc_score(y0, prob)) / 4
res_bayes = gp_minimize(bayes_obj, SPACE, n_calls=200, n_initial_points=15,
                        x0=[[10, 163, 0.764, 0.266, 1.0, 1.451]],      # 原文最优值起步
                        random_state=42, verbose=False)
BEST = dict(zip(["max_depth", "n_estimators", "subsample", "min_split_gain",
                 "reg_alpha", "reg_lambda"], res_bayes.x))
BEST = {k: (int(v) if k in ("max_depth", "n_estimators") else float(v)) for k, v in BEST.items()}
print(f"  最优参数: {BEST}  目标={-res_bayes.fun:.4f}  (耗时 {time.time()-t0:.0f}s)")
mdl_best = LGBMClassifier(learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1, **BEST)
prob_b = cross_val_predict(mdl_best, X0, y0, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
model_cmp["LightGBM-bayes"] = metrics(y0, prob_b, (prob_b > 0.5).astype(int))
print(f"  LightGBM-bayes {model_cmp['LightGBM-bayes']}")

# ---------------------------------------------------------------- 4. 滚动回测
print("[bt] 滚动回测 (LightGBM-bayes, 年度重训[论文形式]+训练窗滞后13月, 12月窗) ...")
ret_panel = panel.select(["symbol", "trade_date",
    (pl.col("hfq_close") / pl.col("hfq_close").shift(1) - 1).alias("dret")])
ret_panel = {str(dd): g for (dd,), g in ret_panel.group_by("trade_date")}
bench_open = {str(r[0]): r[1] for r in data["bench"].iter_rows()}

holdings = {"LGBv2_mw": [], "LGBv2_ew10": []}
port_ret = {k: [] for k in holdings}
prev_w = {k: {} for k in holdings}
port_dates, bench_ret = [], []
hold_log, exec_log, fm_coefs = [], [], []
ic_log = []                                                   # 月度Rank IC(新增)
fitted, lgb_imp = None, {}
last_ew_rebal = -99

for i, sd in enumerate(months_all):
    if sd < SIG_MIN:
        continue
    if sd not in cs or cs[sd].shape[0] < 500:
        continue
    f = cs[sd]
    syms = f["symbol"].to_list()
    # FM 月截面(用月频收益标签, 与16因子版对照口径一致)
    lj_m = lab_m.filter(pl.col("sig_date") == sd)
    fm = f.join(lj_m.select(["symbol", "label"]), on="symbol", how="inner")
    yfm = fm["label"].to_numpy()
    Zfm = np.column_stack([fm["z_" + k].to_numpy() for k in FEAT])
    mm = np.isfinite(yfm) & np.isfinite(Zfm).all(axis=1)
    if mm.sum() > 200:
        Xm = np.column_stack([np.ones(mm.sum()), Zfm[mm]])
        b, *_ = np.linalg.lstsq(Xm, yfm[mm], rcond=None)
        fm_coefs.append(b)
    if i < TRAIN_M + LABEL_LAG:
        continue
    # 重训: 每年一次(论文年度滚动), 训练窗严格滞后13个月(标签完整实现, 无前视)
    if fitted is None or (i - i0) % 12 == 0:
        Xtr, ytr = collect_train(i)
        if Xtr is None or len(ytr) <= 500:
            pass
        else:
            fitted = LGBMClassifier(learning_rate=0.05, n_jobs=-1, random_state=42,
                                    verbose=-1, **BEST).fit(Xtr, ytr)
            lgb_imp = dict(zip(FEAT,
                               np.round(fitted.feature_importances_ / max(fitted.feature_importances_.sum(), 1e-12), 4).tolist()))
            print(f"  [{sd}] 重训 样本={len(ytr):,} 正例={ytr.mean():.3f}")
    if fitted is None:
        continue
    Z = np.column_stack([f["z_" + k].to_numpy() for k in FEAT])
    pred = fitted.predict_proba(np.nan_to_num(Z, nan=np.nan))[:, 1]   # LGBM原生支持NaN
    lab_cur = lab_m.filter(pl.col("sig_date") == sd)
    if lab_cur.shape[0] == 0:
        continue
    exe = f.join(lab_cur, on="symbol", how="inner")
    # ---- 月度Rank IC(新增): 预测概率 vs 下月open-to-open后复权收益(Spearman)
    _lj = f.join(lab_cur.select(["symbol", "label"]), on="symbol", how="left")
    _r1 = _lj["label"].to_numpy()
    _m = np.isfinite(_r1) & np.isfinite(pred)
    if _m.sum() > 50:
        from scipy.stats import spearmanr
        ic_log.append({"sig": sd, "ic": float(spearmanr(pred[_m], _r1[_m]).statistic),
                       "n": int(_m.sum())})
    ind_arr = f["industry"].to_numpy()
    uu, cc = np.unique(ind_arr, return_counts=True)
    u_ind = dict(zip(uu, cc / len(ind_arr)))
    Zstyle = np.column_stack([f["z_" + k].to_numpy() for k in STYLE])
    ret12 = V3.last12m_ret_matrix(ret_panel, cal, sd, syms)

    # ---- 组合1: LGBv2_mw (Top20约束Markowitz, 含beta250盒约束, 阈值0.5%)
    wsel, expo = V3.pick_markowitz_constrained(pred, syms, ret12, Zstyle,
                                               ind_arr, u_ind, holdings["LGBv2_mw"],
                                               topn=TOPN_MW)
    mu_all = dict(zip(syms, pred))
    adv = sum(w * mu_all.get(s, 0) for s, w in wsel.items()) - \
          sum(w * mu_all.get(s, 0) for s, w in prev_w["LGBv2_mw"].items())
    if prev_w["LGBv2_mw"] and adv < REBAL_THR:
        wsel = prev_w["LGBv2_mw"]
    else:
        prev_w["LGBv2_mw"] = wsel
    holdings["LGBv2_mw"] = list(wsel.keys())
    r = V3.realize(exe, list(wsel), wsel, sd, exec_log)
    tw = sum(abs(wsel.get(s, 0) - prev_w["LGBv2_mw"].get(s, 0))
             for s in set(wsel) | set(prev_w["LGBv2_mw"]))
    port_ret["LGBv2_mw"].append(r - tw * FEE_SIDE)
    rec = {"sig": sd, "port": "LGBv2_mw", "adv": round(float(adv), 5),
           "w": {s: round(w, 4) for s, w in wsel.items()}}
    rec.update(expo)
    hold_log.append(rec)

    # ---- 组合2: LGBv2_ew10 (原文形式: Top10等权, 每3个月调仓)
    if i - last_ew_rebal >= EW_REBAL_EVERY:
        top10 = [syms[j] for j in np.argsort(-pred)[:TOPN_EW]]
        w10 = {s: 1.0 / TOPN_EW for s in top10}
        prev_w["LGBv2_ew10"] = w10
        holdings["LGBv2_ew10"] = top10
        last_ew_rebal = i
    w10 = prev_w["LGBv2_ew10"]
    r2 = V3.realize(exe, list(w10), w10, sd, exec_log)
    tw2 = 0.0 if i == last_ew_rebal else 0.0        # 等权持有期无换手费(月初已换)
    port_ret["LGBv2_ew10"].append(r2 - tw2 * FEE_SIDE)
    hold_log.append({"sig": sd, "port": "LGBv2_ew10",
                     "rebalanced": i == last_ew_rebal,
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

# ---------------------------------------------------------------- 5. 统计输出
# Rank IC 汇总(新增)
_ics = np.array([x["ic"] for x in ic_log], float)
ic_stat = {"IC均值": round(float(_ics.mean()), 4),
           "ICIR": round(float(_ics.mean() / _ics.std()), 3) if _ics.std() > 0 else None,
           "IC_t值": round(float(_ics.mean() / _ics.std() * np.sqrt(len(_ics))), 2) if _ics.std() > 0 else None,
           "IC>0占比": round(float((_ics > 0).mean()), 3), "月份数": int(len(_ics))}
print("\n================ 月度Rank IC(预测概率 vs 下月收益) ================")
print(ic_stat)

print("\n================ 组合绩效 ================")
rows = {k: V3.perf_stats(v, port_dates) for k, v in port_ret.items()}
for k, v in rows.items():
    print(k, v)
bench_stat = V3.perf_stats(bench_ret, port_dates)
print("沪深300", bench_stat)

out_dir = os.path.dirname(os.path.abspath(__file__))
import csv as _csv
with open(os.path.join(out_dir, f"monthly_returns_{TAG}.csv"), "w", newline="", encoding="utf-8-sig") as fp:
    w = _csv.writer(fp)
    hdr = ["月份", "沪深300"] + [k for k in port_ret] + [k + "_超额" for k in port_ret]
    w.writerow(hdr)
    for i2, d in enumerate(port_dates):
        row = [d[:7], round(bench_ret[i2], 5)]
        row += [round(port_ret[k][i2], 5) for k in port_ret]
        row += [round(port_ret[k][i2] - bench_ret[i2], 5) for k in port_ret]
        w.writerow(row)
with open(os.path.join(out_dir, f"holdings_detailed_{TAG}.json"), "w", encoding="utf-8") as fp:
    json.dump(hold_log, fp, ensure_ascii=False, indent=1, default=V3._np_json_default)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(port_dates))
    for k, v in port_ret.items():
        ax.plot(x, np.cumprod(1 + np.array(v)), label=k, linewidth=1.4)
    ax.plot(x, np.cumprod(1 + np.nan_to_num(np.array(bench_ret, float), nan=0.0)),
            label="CSI 300", linewidth=1.4, color="black", linestyle="--")
    xt = [i for i in range(0, len(port_dates), 12)]
    ax.set_xticks(xt)
    ax.set_xticklabels([port_dates[i][:7] for i in xt], rotation=0, fontsize=8)
    ax.set_title(f"paper-v2 2020 rerun: LightGBM-Bayes 11-factor ({SIG_MIN[:7]}~2026.10, fee 1bp/side)")
    ax.set_ylabel("NAV")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    # 3个核心指标标注(主组合 LGBv2_mw)
    _mw = rows["LGBv2_mw"]
    _txt = (f"LGBv2_mw 核心指标\n"
            f"夏普: {_mw['夏普']:.2f}\n"
            f"年化收益: {_mw['年化收益']*100:.1f}%\n"
            f"Rank IC均值: {ic_stat['IC均值']:.4f} (ICIR {ic_stat['ICIR']})")
    ax.text(0.985, 0.03, _txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, family="Microsoft YaHei",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8e1", edgecolor="#e0c060", alpha=0.95))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"wealth_{TAG}.png"), dpi=130)
    print(f"图已存 wealth_{TAG}.png")
except Exception as e:
    print("画图失败:", e)

print("\n================ FM因子回归 (Newey-West t, 月频收益×11因子V2扩展集) ================")
fm = np.array(fm_coefs)
fm_rows = {}
names = ["alpha"] + FEAT
for j, nm in enumerate(names):
    t, mu = V3.nw_tstat(fm[:, j])
    fm_rows[nm] = {"月均": round(mu, 5), "NW_t": round(t, 2)}
    print(f"{nm:10s} 月均={mu:+.5f}  NW_t={t:+.2f}")

print("\n================ GRS 检验 ================")
months_used = port_dates
Fmim = V3.factor_mimicking_returns(cs, lab_m, months_used)
Fmat = np.column_stack([Fmim[k] for k in FEAT])
Rports = np.column_stack([np.array(port_ret[k]) for k in port_ret])
gm = np.isfinite(Rports).all(axis=1) & np.isfinite(Fmat).all(axis=1)
stat, p, alpha = V3.grs_test(Rports[gm], Fmat[gm])
print(f"GRS stat={stat:.3f}  p={p:.4f}")

print("\nLightGBM特征重要性(gain/分裂, 末窗):", lgb_imp)

# ---- 收益集中度指标(问题诊断对照): beta/相关/top月贡献
print("\n================ 收益集中度诊断 ================")
conc = {}
barr = np.array(bench_ret, float)
for k, v in port_ret.items():
    r = np.array(v, float)
    ok = np.isfinite(r) & np.isfinite(barr)
    beta_m = np.cov(r[ok], barr[ok])[0, 1] / np.var(barr[ok])
    corr_m = np.corrcoef(r[ok], barr[ok])[0, 1]
    lr = np.log1p(r[ok]); lex = np.log1p(r[ok]) - np.log1p(barr[ok])
    o_abs = np.argsort(-lr); o_ex = np.argsort(-lex)
    conc[k] = {"月beta": round(float(beta_m), 2), "月corr": round(float(corr_m), 2),
               "top5月绝对贡献占比": round(float(lr[o_abs[:5]].sum() / lr.sum()), 3),
               "top5月超额贡献占比": round(float(lex[o_ex[:5]].sum() / lex.sum()), 3),
               "超额最大单月": round(float(np.expm1(lex[o_ex[0]])), 4),
               "超额最大单月序号": int(o_ex[0])}
    print(f"{k:12s} beta={beta_m:.2f} corr={corr_m:.2f} top5绝对贡献={conc[k]['top5月绝对贡献占比']:.0%} "
          f"top5超额贡献={conc[k]['top5月超额贡献占比']:.0%} 最大超额月={conc[k]['超额最大单月']:+.1%}")
conc["沪深300"] = {"note": "基准自身top5月对数贡献>100%属A股绝对收益固有事件驱动特征"}

result = {"组合绩效": rows, "沪深300": bench_stat, "FM因子": fm_rows,
          "RankIC": ic_stat, "月度IC序列": [{"sig": x["sig"], "ic": round(x["ic"], 4)} for x in ic_log],
          "模型对比_首窗": model_cmp, "贝叶斯最优参数": BEST,
          "GRS": {"stat": stat, "p": p},
          "LGB特征重要性": lgb_imp, "集中度诊断": conc,
          "组合月收益": {k: [round(x, 5) for x in v] for k, v in port_ret.items()},
          "基准月收益": [round(x, 5) for x in bench_ret],
          "月份": months_used,
          "费率": "单边万一(原文千一)", "持仓数": f"MW Top{TOPN_MW}(含beta250盒约束) / 原文形式 Top{TOPN_EW}等权季调",
          "窗口": f"{SIG_MIN[:7]}-2026.10",
          "特征集": "V2扩展11因子: 论文保留or_yoy/np_yoy/roa/ps/pcf + 16因子体系ep/turn20/std60/revs20/accrual/gpm",
          "论文": "Li,Xu,Li (2022) Procedia CS 214:1234-1240",
          "标签": "前向12月后复权收益>0为1(原论文口径)",
          "耗时秒": round(time.time() - t0, 1)}
out_json = os.path.join(out_dir, f"backtest_result_{TAG}.json")
with open(out_json, "w", encoding="utf-8") as fp:
    json.dump(result, fp, ensure_ascii=False, indent=2, default=V3._np_json_default)
print(f"\n[done] 结果已存 {out_json}  总耗时 {time.time()-t0:.0f}s")
