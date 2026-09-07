"""
================================================================================
TradingView 多图表庄家资金指标 (L3 Banker Fund Oscillator) 算法共振策略程序
Multi-Chart Confluence Trading Engine for TradingView Datasets
================================================================================
支持4图表跨周期/跨形态共振分析：
1. 10分钟普通K线图 (10m Regular K-Line) -> 宏观周期与庄家资金基底
2. 4分钟0.1688%砖形图 (4m Renko 0.1688%) -> 中波段势能与WaveTrend确认
3. 0.09%小周期砖形图 (Renko 0.09% / 2m) -> 极致灵敏微观拐点扳机
4. 4R Range图 (Range 4R) -> 非时间纯结构反转与订单块阻力支撑确认

核心目标：
- 抄底在启动前 (Bottom fishing before takeoff, High MFE, Minimal MAE)
- 卖在还没跌的最高值 (Peak selling before plunge, Max captured profit)
"""

import os
import sys
import glob
import json
import re
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# 适配 Windows 终端编码，防止中文字符乱码或编码报错
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def read_csv_robust(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """具备多重中文编码容错的 CSV 读取器，兼容 utf-8-sig, utf-8, gb18030, gbk。"""
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            df = pd.read_csv(filepath, nrows=nrows, encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    df = pd.read_csv(filepath, nrows=nrows, encoding="utf-8", errors="replace")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_timestamp_to_epoch(val) -> float:
    """将 TradingView 导出的 ISO 字符串或 Unix 时间戳转换为 float 秒数，兼容各类格式与空值。"""
    if val is None or pd.isna(val) or val == "":
        return np.nan
    try:
        f = float(val)
        if f > 1e14:
            return f / 1e9
        elif f > 1e11:
            return f / 1e3
        return f
    except (ValueError, TypeError):
        try:
            dt = pd.to_datetime(val, format="mixed")
            if pd.isna(dt):
                return np.nan
            return dt.timestamp()
        except Exception:
            return np.nan


def parse_series_to_epoch(s: pd.Series) -> pd.Series:
    """批量超高效时间戳序列转换（支持毫秒/纳秒自适应、ISO字符串及混合格式）。"""
    s_num = pd.to_numeric(s, errors="coerce")
    valid_mask = s_num.notna()
    if valid_mask.sum() > 0 and (valid_mask.sum() >= len(s) * 0.8):
        s_float = s_num.astype(float)
        median_val = s_float[valid_mask].median()
        if median_val > 1e14:      # 纳秒 (19位)
            s_float = s_float / 1e9
        elif median_val > 1e11:    # 毫秒 (13位)
            s_float = s_float / 1e3
        return s_float

    try:
        s_dt = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
        if s_dt.notna().any():
            s_int = s_dt.astype("int64")
            unit = str(s_dt.dtype).split("[")[-1].split(",")[0].rstrip("]")
            divisor = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}.get(unit, 1e9)
            s_epoch = s_int / divisor
            s_epoch[s_dt.isna()] = np.nan
            return s_epoch
    except Exception:
        pass

    return s.apply(parse_timestamp_to_epoch)


class MultiChartDataManager:
    """负责加载、清洗并无未来函数地对其四张图表的时间序列数据。"""

    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.raw_dfs: Dict[str, pd.DataFrame] = {}
        self.merged_df: Optional[pd.DataFrame] = None

    @staticmethod
    def is_raw_chart_candidate(filepath: str) -> bool:
        """过滤掉回测/对齐导出的结果文件，仅保留包含时间戳的原始图表 CSV。"""
        basename = os.path.basename(filepath).lower()
        exclude_kw = ["signal", "trade", "result", "aligned", "export", "summary", "log", "backtest", "optimize"]
        if any(kw in basename for kw in exclude_kw):
            return False
        try:
            head = read_csv_robust(filepath, nrows=5)
            if head is None or head.empty or len(head.columns) < 2:
                return False

            cols = [str(c).strip() for c in head.columns]
            cols_lower = [c.lower() for c in cols]

            output_indicators = [
                "raw_buy", "raw_sell", "renko_009_rg_close", "10m_rg_close",
                "range_4r_rg_close", "4m_01688_rg_close", "cumulative_pnl",
                "pnl_pct", "bars_held", "exit_type", "mfe_pct", "mae_pct"
            ]
            if any(col in cols for col in output_indicators):
                return False

            has_time = any(c in cols_lower for c in ["time", "timestamp", "datetime", "date_time", "date", "epoch", "时间", "日期", "成交时间"])
            if not has_time:
                return False

            return True
        except Exception:
            return False

    def auto_detect_files(self) -> Dict[str, str]:
        """根据文件名和特征自动匹配四张图表文件，隔离历史结果导出文件。"""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        candidates = [f for f in csv_files if self.is_raw_chart_candidate(f)]
        if not candidates:
            return {}

        # 若存在同目录下多个标的，自动锁定文件数最多的标的前缀（如 SSE_603993）
        ticker_counts = {}
        for f in candidates:
            b = os.path.basename(f)
            if "," in b:
                prefix = b.split(",")[0].strip()
                ticker_counts[prefix] = ticker_counts.get(prefix, 0) + 1
        if ticker_counts:
            best_prefix = max(ticker_counts.items(), key=lambda x: x[1])[0]
            prefix_candidates = [f for f in candidates if os.path.basename(f).startswith(best_prefix)]
            if len(prefix_candidates) >= 2:
                candidates = prefix_candidates

        file_map = {}

        # 1. 优先根据文件名模式匹配
        for f in candidates:
            basename = os.path.basename(f)
            b_lower = basename.lower()
            # Range 纯价格结构图：如 2R_xxx.csv, 4R_xxx.csv, 10R_xxx.csv, 或带 range
            if re.search(r'(?:^|[\s_,])\d+r(?:$|[\s_.,])', b_lower) or "range" in b_lower:
                if "range_4r" not in file_map:
                    file_map["range_4r"] = f
            # 10m 宏观普通K线
            elif "10_" in basename or "10m" in b_lower or re.search(r'(?:^|[\s_,])10(?:$|[_.,])', basename):
                if "10m" not in file_map:
                    file_map["10m"] = f
            # 0.09% 微观砖形图 (TV命名如 2_xxxx.csv，注意排除 2R)
            elif (re.search(r'(?:^|[\s_,])2(?:$|[_.,])', basename) or "0.09" in basename) and not re.search(r'(?:^|[\s_,])\d+r', b_lower):
                if "renko_009" not in file_map:
                    file_map["renko_009"] = f
            # 4m 0.1688% 动量砖形图 (TV命名如 4_xxxx.csv，注意排除 4R)
            elif (re.search(r'(?:^|[\s_,])4(?:$|[_.,])', basename) or "0.1688" in basename or "4m" in b_lower) and not re.search(r'(?:^|[\s_,])\d+r', b_lower):
                if "4m_01688" not in file_map:
                    file_map["4m_01688"] = f

        # 2. 如果自动匹配缺失，按指标特征内容推断
        if len(file_map) < 4:
            for f in candidates:
                if f in file_map.values():
                    continue
                try:
                    df_temp = read_csv_robust(f, nrows=15)
                    cols_str = " ".join([str(c).strip() for c in df_temp.columns])
                    if any(k in cols_str for k in ["WT1", "WT2", "WaveTrend", "中轴线"]) and "4m_01688" not in file_map:
                        file_map["4m_01688"] = f
                    elif any(k in cols_str for k in ["通道位置", "需求强度", "供给强度", "成交量中心线", "成交量压力"]) and "renko_009" not in file_map:
                        file_map["renko_009"] = f
                    elif any(k in cols_str for k in ["反转形状", "买入形状", "卖出形状", "多头反转形状", "快速 MA", "慢速 MA"]) and "range_4r" not in file_map:
                        file_map["range_4r"] = f
                    elif any(k in cols_str for k in ["EMA 曲线", "Cloud Reach", "Cloud Candle", "EMA"]) and "10m" not in file_map:
                        file_map["10m"] = f
                except Exception:
                    pass

        return file_map

    def load_and_preprocess(self, file_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """加载四张表并使用因果 merge_asof 对齐时间戳。"""
        if file_map is None:
            file_map = self.auto_detect_files()

        if not file_map:
            raise FileNotFoundError(f"在目录 [{os.path.abspath(self.data_dir)}] 中未识别到任何有效的 TradingView 图表数据 CSV 文件！")

        print("--> 识别到的图表数据文件映射:")
        for k, v in file_map.items():
            print(f"    [{k:<12}] -> {os.path.basename(v)}")

        dfs = {}
        for key, path in file_map.items():
            df = read_csv_robust(path)

            time_col = None
            for c in df.columns:
                if c.strip().lower() in ["time", "timestamp", "datetime", "date_time", "date", "epoch", "时间", "日期", "成交时间"]:
                    time_col = c
                    break

            if time_col is None:
                raise ValueError(
                    f"图表 [{key}] 对应文件 [{os.path.basename(path)}] 未找到有效的时间戳列 (如 time, timestamp, datetime, epoch)！"
                    f"现有列名: {list(df.columns[:10])}"
                )

            df["epoch"] = parse_series_to_epoch(df[time_col])
            df = df.dropna(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)
            df["dt"] = (
                pd.to_datetime(df["epoch"], unit="s", utc=True)
                .dt.tz_convert("Asia/Shanghai")
            )

            ohlc_map = {}
            for col in df.columns:
                if col.strip().lower() in ["open", "high", "low", "close", "volume"]:
                    ohlc_map[col] = col.strip().lower()
            if ohlc_map:
                df = df.rename(columns=ohlc_map)

            # 规范化 Banker Fund 指标列名
            rename_dict = {}
            for col in df.columns:
                col_s = col.strip()
                if "red_green" in col_s:
                    if "关" in col_s or "close" in col_s.lower():
                        rename_dict[col] = f"{key}_rg_close"
                    elif "开" in col_s or "open" in col_s.lower():
                        rename_dict[col] = f"{key}_rg_open"
                    elif "高" in col_s or "high" in col_s.lower():
                        rename_dict[col] = f"{key}_rg_high"
                    elif "低" in col_s or "low" in col_s.lower():
                        rename_dict[col] = f"{key}_rg_low"
                elif "cyan_magenta" in col_s:
                    if "关" in col_s or "close" in col_s.lower():
                        rename_dict[col] = f"{key}_cm_close"
                    elif "开" in col_s or "open" in col_s.lower():
                        rename_dict[col] = f"{key}_cm_open"
                elif "WT1" in col_s and "WT1 — 快线信号" not in df.columns:
                    rename_dict[col] = "WT1 — 快线信号"
                elif "WT2" in col_s and "WT2 — 慢线确认" not in df.columns:
                    rename_dict[col] = "WT2 — 慢线确认"

            df = df.rename(columns=rename_dict)
            dfs[key] = df
            self.raw_dfs[key] = df

        # 以最高灵敏度的时间序列 (Renko 0.09%) 作为基底进行对齐
        base_key = "renko_009" if "renko_009" in dfs else list(dfs.keys())[0]
        base_cols = ["epoch", "dt"]
        for c in ["open", "high", "low", "close", f"{base_key}_rg_close", f"{base_key}_cm_close"]:
            if c in dfs[base_key].columns:
                base_cols.append(c)

        base = dfs[base_key][base_cols].copy()
        if "close" in base.columns:
            if "high" not in base.columns:
                base["high"] = base["close"]
            if "low" not in base.columns:
                base["low"] = base["close"]
            if "open" not in base.columns:
                base["open"] = base["close"]

        # 添加 Renko 特有指标
        for extra_col in ["需求强度 %", "供给强度 %", "成交量压力 %", "趋势力度 %", "通道位置"]:
            if extra_col in dfs[base_key].columns:
                base[f"renko_{extra_col}"] = dfs[base_key][extra_col]

        # 依次因果向前合并（direction='backward'，完全无未来函数）
        for key in ["10m", "4m_01688", "range_4r"]:
            if key in dfs and key != base_key:
                other_df = dfs[key]
                merge_cols = ["epoch"]
                if f"{key}_rg_close" in other_df.columns:
                    merge_cols.append(f"{key}_rg_close")
                if f"{key}_cm_close" in other_df.columns:
                    merge_cols.append(f"{key}_cm_close")

                # 提取特定辅助指标
                if key == "4m_01688":
                    for c in ["WT1 — 快线信号", "WT2 — 慢线确认", "WT 差值（动量）"]:
                        if c in other_df.columns:
                            merge_cols.append(c)
                elif key == "range_4r":
                    for c in ["买入形状", "卖出形状", "多头反转形状", "空头反转形状", "快速 MA", "慢速 MA", "支撑", "阻力"]:
                        if c in other_df.columns:
                            merge_cols.append(c)
                elif key == "10m":
                    for c in ["Cloud Reach", "EMA 曲线", "Red %", "Blue %"]:
                        if c in other_df.columns:
                            merge_cols.append(c)

                sub_other = other_df[merge_cols].dropna(subset=["epoch"]).sort_values("epoch")
                base = pd.merge_asof(base, sub_other, on="epoch", direction="backward")

        # 计算速度与差分
        for rg_col in ["10m_rg_close", "4m_01688_rg_close", "renko_009_rg_close", "range_4r_rg_close"]:
            if rg_col in base.columns:
                base[f"{rg_col}_d1"] = base[rg_col].diff()

        if "renko_009_rg_close" in base.columns:
            base["renko_min_past5"] = base["renko_009_rg_close"].rolling(6, min_periods=1).min()

        # 计算微观金叉死叉
        if "renko_009_rg_close" in base.columns and "renko_009_cm_close" in base.columns:
            base["renko_cross_up"] = (base["renko_009_rg_close"] > base["renko_009_cm_close"]) & (
                base["renko_009_rg_close"].shift(1) <= base["renko_009_cm_close"].shift(1)
            )
            base["renko_cross_dn"] = (base["renko_009_rg_close"] < base["renko_009_cm_close"]) & (
                base["renko_009_rg_close"].shift(1) >= base["renko_009_cm_close"].shift(1)
            )

        if "range_4r_rg_close" in base.columns and "range_4r_cm_close" in base.columns:
            base["range_cross_up"] = (base["range_4r_rg_close"] > base["range_4r_cm_close"]) & (
                base["range_4r_rg_close"].shift(1) <= base["range_4r_cm_close"].shift(1)
            )
            base["range_cross_dn"] = (base["range_4r_rg_close"] < base["range_4r_cm_close"]) & (
                base["range_4r_rg_close"].shift(1) >= base["range_4r_cm_close"].shift(1)
            )

        self.merged_df = base
        print(f"--> 数据对齐成功！共对齐 {len(base)} 行数据，覆盖时间: {base['dt'].iloc[0]} 至 {base['dt'].iloc[-1]}")
        return base


class MultiChartAlgorithm:
    """
    四图表共振算法执行引擎：
    包括：
    1. 买入抄底共振规则 (Buy Bottom Confluence)
    2. 卖出逃顶共振规则 (Sell Top Confluence)
    3. 全流程回测与统计
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def get_default_parameters() -> dict:
        """根据网格优化与统计分析得出的黄金共振阈值参数。"""
        return {
            # === 买入抄底共振参数 (抄底在启动前) ===
            "buy_10m_rg_max": 42.0,      # 10m宏观底过滤（庄家低位吸筹区，必须<=42）
            "buy_4m_rg_max": 30.0,       # 4m中波段底确认（必须<=30，历史大底均在16~27）
            "buy_renko_rg_max": 20.0,    # Renko微观极值探底（必须触及或低于20，极端底<10）
            "buy_renko_turn_up": True,   # Renko开始拐头向上（d1 > 0 或金叉，杜绝飞刀）
            "buy_range_rg_max": 30.0,    # Range 4R极值共振确认（若有数据，<=30）
            "buy_wt1_max": -20.0,        # 4m WaveTrend快线处于超卖区 (<= -20)
            
            # === 卖出逃顶共振参数 (卖在还没跌的最高值) ===
            "sell_renko_rg_min": 80.0,   # Renko微观冲顶（必须>=80，逼近顶部线96~100）
            "sell_range_rg_min": 78.0,   # Range 4R冲顶共振确认（>=78）
            "sell_renko_turn_dn": True,  # Renko微观拐头向下或死叉（高位滞涨第一根信号）
            "sell_4m_rg_min": 35.0,      # 4m中波段进入高风险区（>=35）
            "sell_wt1_min": 35.0,        # 4m WaveTrend快线处于超买冲顶区 (>= 35)
            "stop_loss_pct": 1.2,        # 严格风险截断止损阈值 (1.2%)
        }

    def evaluate_signals(self, params: Optional[dict] = None) -> pd.DataFrame:
        """对整张时序数据评估买入和卖出信号。"""
        p = {**self.get_default_parameters(), **(params or {})}
        df = self.df.copy()
        
        # 1. 滚动微观极值窗口
        if "renko_009_rg_close" in df.columns:
            renko_min = df["renko_min_past5"] if "renko_min_past5" in df.columns else df["renko_009_rg_close"].rolling(6, min_periods=1).min()
            cond_renko_oversold = (df["renko_009_rg_close"] <= p["buy_renko_rg_max"]) | (
                renko_min <= 15.0
            )
            d1 = df.get("renko_009_rg_close_d1", pd.Series(0, index=df.index))
            cond_renko_trigger = (d1 > 0) | df.get("renko_cross_up", False)
        else:
            cond_renko_oversold = True
            cond_renko_trigger = True

        # 2. 买入条件合成
        cond_10m_buy = (df["10m_rg_close"] <= p["buy_10m_rg_max"]) if "10m_rg_close" in df.columns else True
        cond_4m_buy = (df["4m_01688_rg_close"] <= p["buy_4m_rg_max"]) if "4m_01688_rg_close" in df.columns else True

        # Range 确认（若缺失则跳过）
        if "range_4r_rg_close" in df.columns:
            cond_range_buy = (df["range_4r_rg_close"].isna()) | (
                df["range_4r_rg_close"] <= p["buy_range_rg_max"]
            ) | (df.get("range_4r_rg_close_d1", 0) > 0)
        else:
            cond_range_buy = True

        # WT1 辅助过滤
        if "WT1 — 快线信号" in df.columns:
            cond_wt_buy = (df["WT1 — 快线信号"].isna()) | (df["WT1 — 快线信号"] <= p["buy_wt1_max"])
        else:
            cond_wt_buy = True

        df["raw_buy"] = (
            cond_10m_buy
            & cond_4m_buy
            & cond_renko_oversold
            & cond_renko_trigger
            & cond_range_buy
            & cond_wt_buy
        )

        # 3. 卖出条件合成
        if "renko_009_rg_close" in df.columns:
            d1 = df.get("renko_009_rg_close_d1", pd.Series(0, index=df.index))
            cond_renko_top = (df["renko_009_rg_close"] >= p["sell_renko_rg_min"]) & (
                (d1 < 0) | df.get("renko_cross_dn", False)
            )
            cond_death_cross_top = df.get("renko_cross_dn", False) & (df["renko_009_rg_close"] >= 65.0)
        else:
            cond_renko_top = False
            cond_death_cross_top = False

        if "range_4r_rg_close" in df.columns:
            cond_range_top = (
                df["range_4r_rg_close"].notna()
                & (df["range_4r_rg_close"] >= p["sell_range_rg_min"])
                & (df.get("range_4r_rg_close_d1", 0) < 0)
            )
        else:
            cond_range_top = False

        df["raw_sell"] = cond_renko_top | cond_range_top | cond_death_cross_top

        return df

    def run_backtest(self, params: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
        """运行无未来函数的动态状态机回测。"""
        p = {**self.get_default_parameters(), **(params or {})}
        df_sig = self.evaluate_signals(p)
        stop_loss_pct = p.get("stop_loss_pct", 1.2)

        n = len(df_sig)
        if n == 0 or "close" not in df_sig.columns:
            summary = {
                "total_trades": 0, "win_count": 0, "loss_count": 0,
                "win_rate_%": 0.0, "cumulative_pnl_%": 0.0, "profit_factor": 0.0,
                "avg_win_%": 0.0, "avg_loss_%": 0.0, "avg_mfe_%": 0.0, "avg_mae_%": 0.0,
                "profit_risk_ratio": 0.0
            }
            return pd.DataFrame(), summary

        # 极致向量化加速：提取 numpy 1D 数组
        raw_buy = df_sig["raw_buy"].to_numpy(dtype=bool)
        raw_sell = df_sig["raw_sell"].to_numpy(dtype=bool)
        close = df_sig["close"].to_numpy(dtype=float)
        high = df_sig["high"].to_numpy(dtype=float) if "high" in df_sig.columns else close
        low = df_sig["low"].to_numpy(dtype=float) if "low" in df_sig.columns else close
        dt = df_sig["dt"].to_numpy()

        trades = []
        position = 0
        entry_price = 0.0
        entry_time = None
        entry_idx = 0
        peak_price = 0.0
        trough_price = 1e9

        for i in range(n):
            if position == 0:
                if raw_buy[i]:
                    position = 1
                    entry_price = close[i]
                    entry_time = dt[i]
                    entry_idx = i
                    peak_price = high[i]
                    trough_price = low[i]
            elif position == 1:
                if high[i] > peak_price:
                    peak_price = high[i]
                if low[i] < trough_price:
                    trough_price = low[i]

                # 止损检测
                current_pnl = (close[i] - entry_price) / entry_price * 100.0
                is_stop_loss = current_pnl <= -stop_loss_pct

                if raw_sell[i] or is_stop_loss:
                    exit_price = close[i]
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                    mfe_pct = (peak_price - entry_price) / entry_price * 100.0
                    mae_pct = (entry_price - trough_price) / entry_price * 100.0

                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": dt[i],
                        "bars_held": int(i - entry_idx),
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price),
                        "pnl_pct": round(pnl_pct, 2),
                        "mfe_pct": round(mfe_pct, 2),
                        "mae_pct": round(mae_pct, 2),
                        "exit_type": "止损离场" if is_stop_loss else "四图逃顶"
                    })
                    position = 0

        df_trades = pd.DataFrame(trades)

        # 统计指标
        if not df_trades.empty:
            total_pnl = df_trades["pnl_pct"].sum()
            win_count = (df_trades["pnl_pct"] > 0).sum()
            loss_count = (df_trades["pnl_pct"] <= 0).sum()
            win_rate = (win_count / len(df_trades)) * 100.0
            wins = df_trades[df_trades["pnl_pct"] > 0]["pnl_pct"]
            losses = df_trades[df_trades["pnl_pct"] <= 0]["pnl_pct"]
            avg_win = wins.mean() if len(wins) > 0 else 0.0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
            profit_factor = wins.sum() / (abs(losses.sum()) + 1e-6)
            avg_mfe = df_trades["mfe_pct"].mean()
            avg_mae = df_trades["mae_pct"].mean()
        else:
            total_pnl = win_rate = avg_win = avg_loss = profit_factor = avg_mfe = avg_mae = 0.0
            win_count = loss_count = 0

        summary = {
            "total_trades": len(df_trades),
            "win_count": int(win_count),
            "loss_count": int(loss_count),
            "win_rate_%": round(win_rate, 2),
            "cumulative_pnl_%": round(total_pnl, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win_%": round(avg_win, 2),
            "avg_loss_%": round(avg_loss, 2),
            "avg_mfe_%": round(avg_mfe, 2),
            "avg_mae_%": round(avg_mae, 2),
            "profit_risk_ratio": round(avg_mfe / (avg_mae + 1e-4), 2)
        }

        return df_trades, summary

    def run_optimization(self) -> pd.DataFrame:
        """执行四图表共振超参数网格扫描，寻找胜率与盈亏比最优解。"""
        print("\n--> 正在执行四图表共振网格寻优计算 (Grid Search Optimization)...")
        grid_10m = [35.0, 42.0, 50.0]
        grid_4m = [25.0, 30.0, 35.0]
        grid_renko = [12.0, 18.0, 25.0]
        grid_sell_renko = [78.0, 82.0, 88.0]
        grid_sell_range = [75.0, 80.0, 85.0]

        candidates = []
        for t10 in grid_10m:
            for t4 in grid_4m:
                for trenko in grid_renko:
                    for s_renko in grid_sell_renko:
                        for s_range in grid_sell_range:
                            p = self.get_default_parameters()
                            p["buy_10m_rg_max"] = t10
                            p["buy_4m_rg_max"] = t4
                            p["buy_renko_rg_max"] = trenko
                            p["sell_renko_rg_min"] = s_renko
                            p["sell_range_rg_min"] = s_range

                            trades, smm = self.run_backtest(p)
                            if smm["total_trades"] >= 1:
                                candidates.append({
                                    "10m_max": t10,
                                    "4m_max": t4,
                                    "renko_max": trenko,
                                    "sell_renko": s_renko,
                                    "sell_range": s_range,
                                    "trades": smm["total_trades"],
                                    "win_rate_%": smm["win_rate_%"],
                                    "pnl_%": smm["cumulative_pnl_%"],
                                    "avg_mfe_%": smm["avg_mfe_%"],
                                    "avg_mae_%": smm["avg_mae_%"],
                                    "profit_risk": smm["profit_risk_ratio"],
                                })

        df_cand = pd.DataFrame(candidates)
        if not df_cand.empty:
            df_cand = df_cand.sort_values(by=["profit_risk", "pnl_%"], ascending=False).reset_index(drop=True)
            print("\n" + "=" * 90)
            print("                 四图表共振网格寻优 TOP 10 最佳参数配置")
            print("=" * 90)
            print(df_cand.drop_duplicates(subset=["trades", "pnl_%", "profit_risk"]).head(10).to_string(index=False))
            print("=" * 90 + "\n")
        else:
            print("--> 网格搜索未找到有效组合。")
        return df_cand


def print_strategy_report(summary: dict, df_trades: pd.DataFrame, params: dict):
    """打印清晰美观的算法决策与绩效报告。"""
    print("\n" + "=" * 90)
    print("      TRADINGVIEW 四图表庄家资金 (L3 BANKER FUND) 算法共振决策模型报告")
    print("=" * 90)

    print("\n[一、核心图表定位与关键参数设置]")
    print("-" * 90)
    print("1. 10分钟普通K线图 (宏观蓄势过滤):")
    print(f"   * 抄底启动前判定值 : red_green <= {params['buy_10m_rg_max']} (主力吸筹蓄势区)")
    print(f"   * 逃顶防诱多判定值 : red_green >= 58.0 (主力高位滞涨风险区)")
    print("2. 4分钟0.1688%砖形图 (中波段势能与WaveTrend):")
    print(f"   * 抄底启动前判定值 : red_green <= {params['buy_4m_rg_max']} 且 WT1 <= {params['buy_wt1_max']}")
    print(f"   * 逃顶高位判定值   : red_green >= {params['sell_4m_rg_min']} 且 WT1 >= {params['sell_wt1_min']}")
    print("3. 0.09%微观砖形图 (极速微观拐点扳机 - 2m/Renko 0.02):")
    print(f"   * 抄底启动前扳机值 : 曾触及 <= {params['buy_renko_rg_max']} (底带-1~4)，且拐头向上 (d1 > 0 或金叉)")
    print(f"   * 逃顶最高点扳机值 : 冲至 >= {params['sell_renko_rg_min']} (顶带96~101)，且第一根拐头向下或死叉")
    print("4. Range 4R图 (纯价格结构反转与订单块确认):")
    print(f"   * 抄底启动前确认值 : red_green <= {params['buy_range_rg_max']} 且出现多头反转/支撑")
    print(f"   * 逃顶最高点确认值 : red_green >= {params['sell_range_rg_min']} 且出现空头反转/阻力")

    print("\n[二、回测统计绩效与收益风险比]")
    print("-" * 90)
    for k, v in summary.items():
        print(f"  * {k:<20}: {v}")

    print("\n[三、交易记录详情]")
    print("-" * 90)
    if not df_trades.empty:
        cols_disp = ["entry_time", "exit_time", "entry_price", "exit_price", "pnl_pct", "mfe_pct", "mae_pct", "exit_type"]
        print(df_trades[cols_disp].to_string(index=False))
    else:
        print("  (当前参数下无触发交易)")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="TradingView 四图表庄家资金算法共振策略")
    parser.add_argument("--dir", default=".", help="CSV文件所在目录路径 (默认为当前目录)")
    parser.add_argument("--optimize", action="store_true", help="是否执行网格超参数扫描寻优")
    parser.add_argument("--export-csv", default=None, help="导出对齐时序与信号至指定CSV文件")
    args = parser.parse_args()

    # 1. 加载并对齐数据
    manager = MultiChartDataManager(data_dir=args.dir)
    merged_df = manager.load_and_preprocess()

    # 2. 执行算法策略
    engine = MultiChartAlgorithm(merged_df)
    params = engine.get_default_parameters()

    if args.optimize:
        engine.run_optimization()

    df_trades, summary = engine.run_backtest(params)

    # 3. 输出报告
    print_strategy_report(summary, df_trades, params)

    if args.export_csv:
        df_sig = engine.evaluate_signals(params)
        df_sig.to_csv(args.export_csv, index=False, encoding="utf-8-sig")
        print(f"--> 已成功将带买卖信号的时序数据导出至: {args.export_csv}")


if __name__ == "__main__":
    main()
