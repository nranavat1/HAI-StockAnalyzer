import os, glob, numpy as np, pandas as pd

def load_one_csv(fp):
    df = pd.read_csv(fp)
    # normalize column names & date
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else next((c for c in df.columns if "date" in c), None)
    if date_col is None: return None
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    # unify names
    ren = {}
    for k in ["open","high","low","close","volume","adj close","adj_close"]:
        if k not in df.columns:
            alt = next((c for c in df.columns if k.replace("_","") in c.replace("_","")), None)
            if alt: ren[alt] = k.replace(" ","_")
    if ren: df.rename(columns=ren, inplace=True)
    if "adj close" in df.columns and "adj_close" not in df.columns:
        df.rename(columns={"adj close":"adj_close"}, inplace=True)

    needed = {"open","high","low","close","volume"}
    if not needed.issubset(df.columns): return None

    df["ticker"] = os.path.splitext(os.path.basename(fp))[0].upper()
    df.rename(columns={date_col:"date"}, inplace=True)
    return df[["ticker","date","open","high","low","close","volume"]]

def add_feats(x: pd.DataFrame, window_size):
    x = x.sort_values("date").copy()
    # safe numerics
    for c in ["open","high","low","close","volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(x) < window_size + 2: return None

    # features
    x["ret_1d"]   = x["close"].pct_change(1)
    x["ret_5d"]   = x["close"].pct_change(5)
    x["spread"]   = (x["high"] - x["low"]) / x["open"]
    x["vol_ma5"]  = x["volume"].rolling(5, min_periods=3).mean()
    x["vol_ratio"]= x["volume"] / x["vol_ma5"]
    # log-volume for stability
    x["volume"]   = np.log1p(np.clip(x["volume"], a_min=0, a_max=None))
    # target = next-day return
    x["y_true"]   = x["close"].shift(-1) / x["close"] - 1

    x = x.dropna(subset=["ret_1d","ret_5d","spread","vol_ratio","y_true"]).reset_index(drop=True)
    if len(x) < window_size + 1: return None
    return x

def build_windows(block: pd.DataFrame, feat_cols, win=20):
    vals = block[feat_cols].to_numpy(dtype=np.float32)
    yv   = block["y_true"].to_numpy(dtype=np.float32)
    Xs, Ys, meta = [], [], []
    for i in range(len(block) - win):
        # window = rows [i .. i+win-1], predict return at i+win
        W = vals[i:i+win, :].T  # (C, T)
        # optional detrend: subtract last close in window from price cols (0..3)
        # W[0:4, :] = W[0:4, :] - W[3, -1]  # (open,high,low,close) minus last close
        Xs.append(W)
        Ys.append(yv[i+win])
        snap = block.iloc[i+win-1]
        meta.append({
            "ticker": snap["ticker"], "date": snap["date"],
            "open": snap["open"], "high": snap["high"],
            "low": snap["low"], "close": snap["close"], "volume": snap["volume"]
        })
    if not Xs: return None, None, None
    return np.stack(Xs), np.array(Ys), pd.DataFrame(meta)