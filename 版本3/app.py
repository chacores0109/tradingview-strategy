# -*- coding: utf-8 -*-
"""
TradingView 四图表庄家资金 (L3 Banker Fund) 算法共振策略系统 - 版本3 (数据自适应与真实顶底归因版)
核心升级：
1. 真实数据分布与范围全景透视 (Min, Q05, Q25, Median, Q75, Q95, Max)
2. 历史价格绝对波谷与波峰逆向归因分析 (提取每次真实起爆与见顶时四张图的精确数值画像)
3. 真实极值自适应推荐参数 (彻底告别凭空猜测阈值)
4. 自定义网格寻优范围与步长控制 (支持用户任意设置搜索区间与精细步长，NumPy秒级加速)
"""

import os
import sys
import glob
import re
import math
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

# Windows控制台UTF-8编码保护
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QDoubleSpinBox, QSpinBox, QCheckBox, QProgressBar, QPlainTextEdit,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QFrame, QSplitter, QScrollArea, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor


# ==============================================================================
# 工具函数：鲁棒 CSV 读取与时间戳解析
# ==============================================================================
def read_csv_robust(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """自动探测文件编码并读取 CSV，兼容 utf-8, utf-8-sig, gb18030, gbk。"""
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    for enc in encodings:
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
    """将各类格式的时间戳转换为 Unix 浮点秒数。"""
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
    """批量高效转换时间戳序列，自适应纳秒/毫秒/秒与 ISO 字符串。"""
    s_num = pd.to_numeric(s, errors="coerce")
    valid_mask = s_num.notna()
    if valid_mask.sum() > 0 and (valid_mask.sum() >= len(s) * 0.8):
        s_float = s_num.astype(float)
        median_val = s_float[valid_mask].median()
        if median_val > 1e14:
            s_float = s_float / 1e9
        elif median_val > 1e11:
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


# ==============================================================================
# 数据管理器与分布探测器
# ==============================================================================
class MultiChartDataManager:
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.raw_dfs: Dict[str, pd.DataFrame] = {}
        self.merged_df: Optional[pd.DataFrame] = None
        self.indicator_distributions: Dict[str, Dict[str, float]] = {}
        self.swings_low: pd.DataFrame = pd.DataFrame()
        self.swings_high: pd.DataFrame = pd.DataFrame()

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
            return has_time
        except Exception:
            return False

    def auto_detect_files(self) -> Dict[str, str]:
        """智能多策略匹配四张图表原始数据文件，隔离历史导出文件并防止同目录多标的污染。"""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        candidates = [f for f in csv_files if self.is_raw_chart_candidate(f)]
        if not candidates:
            return {}

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
            if re.search(r'(?:^|[\s_,])\d+r(?:$|[\s_.,])', b_lower) or "range" in b_lower:
                if "range_4r" not in file_map:
                    file_map["range_4r"] = f
            elif "10_" in basename or "10m" in b_lower or re.search(r'(?:^|[\s_,])10(?:$|[_.,])', basename):
                if "10m" not in file_map:
                    file_map["10m"] = f
            elif (re.search(r'(?:^|[\s_,])2(?:$|[_.,])', basename) or "0.09" in basename) and not re.search(r'(?:^|[\s_,])\d+r', b_lower):
                if "renko_009" not in file_map:
                    file_map["renko_009"] = f
            elif (re.search(r'(?:^|[\s_,])4(?:$|[_.,])', basename) or "0.1688" in basename or "4m" in b_lower) and not re.search(r'(?:^|[\s_,])\d+r', b_lower):
                if "4m_01688" not in file_map:
                    file_map["4m_01688"] = f

        # 2. 内容推断补足
        if len(file_map) < 4:
            for f in candidates:
                if f in file_map.values():
                    continue
                try:
                    df_temp = read_csv_robust(f, nrows=15)
                    cols_str = " ".join([str(c).strip() for c in df_temp.columns])
                    if any(k in cols_str for k in ["WT1", "WT2", "WaveTrend", "中轴线"]) and "4m_01688" not in file_map:
                        file_map["4m_01688"] = f
                    elif any(k in cols_str for k in ["通道位置", "需求强度", "供给强度", "成交量中心线"]) and "renko_009" not in file_map:
                        file_map["renko_009"] = f
                    elif any(k in cols_str for k in ["反转形状", "买入形状", "卖出形状", "多头反转形状"]) and "range_4r" not in file_map:
                        file_map["range_4r"] = f
                    elif any(k in cols_str for k in ["EMA 曲线", "Cloud Reach", "Cloud Candle"]) and "10m" not in file_map:
                        file_map["10m"] = f
                except Exception:
                    pass

        return file_map

    def load_and_preprocess(self, file_map: Optional[Dict[str, str]] = None, log_fn=None, progress_fn=None) -> pd.DataFrame:
        if file_map is None:
            file_map = self.auto_detect_files()

        if not file_map:
            raise FileNotFoundError(f"在目录 [{os.path.abspath(self.data_dir)}] 中未识别到任何有效的 TradingView 图表数据 CSV 文件！")

        if log_fn:
            log_fn(f"开始加载四图表数据文件，目标目录: {os.path.abspath(self.data_dir)}", "INFO")
            if len(file_map) < 4:
                log_fn(f"提示：当前检测到 {len(file_map)}/4 个图表 (已识别: {list(file_map.keys())})", "WARN")

        dfs = {}
        step = 0
        total_steps = len(file_map) + 2

        for key, path in file_map.items():
            step += 1
            if progress_fn:
                progress_fn(int(step / total_steps * 35), f"正在读取并解析: {os.path.basename(path)}")
            if log_fn:
                log_fn(f"-> 识别到图表 [{key}]: {os.path.basename(path)}", "INFO")

            df = read_csv_robust(path)

            time_col = None
            for c in df.columns:
                if c.strip().lower() in ["time", "timestamp", "datetime", "date_time", "date", "epoch", "时间", "日期", "成交时间"]:
                    time_col = c
                    break

            if time_col is None:
                raise ValueError(f"图表 [{key}] 对应文件 [{os.path.basename(path)}] 未找到有效的时间戳列！")

            df["epoch"] = parse_series_to_epoch(df[time_col])
            df = df.dropna(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)
            df["dt"] = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")

            ohlc_map = {col: col.strip().lower() for col in df.columns if col.strip().lower() in ["open", "high", "low", "close", "volume"]}
            if ohlc_map:
                df = df.rename(columns=ohlc_map)

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

        if progress_fn:
            progress_fn(40, "正在进行因果无未来函数时间对齐 (merge_asof)...")
        if log_fn:
            log_fn("执行多周期时间轴因果同步...", "INFO")

        base_key = "renko_009" if "renko_009" in dfs else list(dfs.keys())[0]
        base_cols = ["epoch", "dt"]
        for c in ["open", "high", "low", "close", f"{base_key}_rg_close", f"{base_key}_cm_close"]:
            if c in dfs[base_key].columns:
                base_cols.append(c)

        merged = dfs[base_key][base_cols].copy().sort_values("epoch").reset_index(drop=True)

        for key, df_other in dfs.items():
            if key == base_key:
                continue
            cols_to_merge = ["epoch"]
            for c in df_other.columns:
                if c.startswith(f"{key}_rg_") or c.startswith(f"{key}_cm_") or c in ["WT1 — 快线信号", "WT2 — 慢线确认"]:
                    cols_to_merge.append(c)
                elif any(shape_kw in c for shape_kw in ["反转形状", "买入形状", "卖出形状", "多头反转形状"]):
                    cols_to_merge.append(c)

            df_sub = df_other[cols_to_merge].sort_values("epoch").drop_duplicates(subset=["epoch"])
            merged = pd.merge_asof(merged, df_sub, on="epoch", direction="backward")

        # 补全 OHLC 缺失列
        if "close" in merged.columns:
            for c in ["open", "high", "low"]:
                if c not in merged.columns:
                    merged[c] = merged["close"]
                else:
                    merged[c] = merged[c].fillna(merged["close"])

        # 衍生辅助列
        if "renko_009_rg_close" in merged.columns:
            merged["renko_diff"] = merged["renko_009_rg_close"].diff().fillna(0)
            merged["renko_min_past5"] = merged["renko_009_rg_close"].rolling(5, min_periods=1).min()
            if "renko_009_cm_close" in merged.columns:
                merged["renko_cross_up"] = (merged["renko_009_rg_close"] > merged["renko_009_cm_close"]) & \
                                           (merged["renko_009_rg_close"].shift(1) <= merged["renko_009_cm_close"].shift(1))
                merged["renko_cross_dn"] = (merged["renko_009_rg_close"] < merged["renko_009_cm_close"]) & \
                                           (merged["renko_009_rg_close"].shift(1) >= merged["renko_009_cm_close"].shift(1))

        self.merged_df = merged

        # 自动计算指标全景分布与真实波段极值归因
        if progress_fn:
            progress_fn(48, "正在计算四图数据全景分布与真实波谷/波峰归因...")
        self.compute_indicator_distributions()
        self.detect_real_swings()

        if log_fn:
            log_fn(f"时间对齐完成！有效行数: {len(merged)} 行，数据分布与顶底画像已提取就绪。", "SUCCESS")

        return merged

    def compute_indicator_distributions(self) -> Dict[str, Dict[str, float]]:
        """计算各图表指标在当前数据集上的真实分位数全景分布。"""
        if self.merged_df is None or self.merged_df.empty:
            return {}

        dist = {}
        target_cols = [
            ("10m_rg_close", "10分钟红绿值"),
            ("4m_01688_rg_close", "4分钟红绿值"),
            ("renko_009_rg_close", "0.09%微观红绿值"),
            ("range_4r_rg_close", "Range 4R红绿值"),
            ("WT1 — 快线信号", "4分钟 WT1快线")
        ]

        for col, label in target_cols:
            if col in self.merged_df.columns:
                s = self.merged_df[col].dropna()
                if len(s) > 0:
                    dist[col] = {
                        "label": label,
                        "min": float(s.min()),
                        "q05": float(s.quantile(0.05)),
                        "q10": float(s.quantile(0.10)),
                        "q25": float(s.quantile(0.25)),
                        "median": float(s.median()),
                        "q75": float(s.quantile(0.75)),
                        "q90": float(s.quantile(0.90)),
                        "q95": float(s.quantile(0.95)),
                        "max": float(s.max())
                    }

        self.indicator_distributions = dist
        return dist

    def detect_real_swings(self, window: int = 15, min_swing_pct: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        逆向极值归因：找出价格序列上的所有真实波谷（起爆点）和真实波峰（逃顶点），
        并精确提取出在每个顶/底发生的瞬间，四张图的具体数值到底是多少！
        """
        if self.merged_df is None or len(self.merged_df) < window * 2:
            return pd.DataFrame(), pd.DataFrame()

        df = self.merged_df
        lows = []
        highs = []
        n = len(df)

        low_arr = df['low'].to_numpy()
        high_arr = df['high'].to_numpy()
        close_arr = df['close'].to_numpy()

        for i in range(window, n - window):
            is_low = (low_arr[i] == low_arr[i - window:i + window + 1].min())
            is_high = (high_arr[i] == high_arr[i - window:i + window + 1].max())

            post_high = high_arr[i:min(i + 40, n)].max()
            post_low = low_arr[i:min(i + 40, n)].min()

            rally_pct = (post_high - close_arr[i]) / close_arr[i] * 100.0
            drop_pct = (close_arr[i] - post_low) / close_arr[i] * 100.0

            row_info = {
                "idx": i,
                "dt": str(df['dt'].iloc[i]),
                "price": round(float(close_arr[i]), 2),
                "10m": round(float(df['10m_rg_close'].iloc[i]), 1) if '10m_rg_close' in df.columns and pd.notna(df['10m_rg_close'].iloc[i]) else np.nan,
                "4m": round(float(df['4m_01688_rg_close'].iloc[i]), 1) if '4m_01688_rg_close' in df.columns and pd.notna(df['4m_01688_rg_close'].iloc[i]) else np.nan,
                "renko": round(float(df['renko_009_rg_close'].iloc[i]), 1) if 'renko_009_rg_close' in df.columns and pd.notna(df['renko_009_rg_close'].iloc[i]) else np.nan,
                "range": round(float(df['range_4r_rg_close'].iloc[i]), 1) if 'range_4r_rg_close' in df.columns and pd.notna(df['range_4r_rg_close'].iloc[i]) else np.nan,
                "wt1": round(float(df['WT1 — 快线信号'].iloc[i]), 1) if 'WT1 — 快线信号' in df.columns and pd.notna(df['WT1 — 快线信号'].iloc[i]) else np.nan,
            }

            if is_low and rally_pct >= min_swing_pct:
                row_info["swing_type"] = "波谷起爆点"
                row_info["change_pct"] = round(rally_pct, 2)
                lows.append(row_info)

            if is_high and drop_pct >= min_swing_pct:
                row_info["swing_type"] = "波峰逃顶点"
                row_info["change_pct"] = round(-drop_pct, 2)
                highs.append(row_info)

        self.swings_low = pd.DataFrame(lows)
        self.swings_high = pd.DataFrame(highs)
        return self.swings_low, self.swings_high

    def get_data_adaptive_recommendations(self) -> Dict[str, float]:
        """根据真实历史大底与大顶的统计画像，生成完全贴合本数据集的推荐阈值。"""
        defaults = {
            "buy_10m_max": 42.0,
            "buy_4m_max": 30.0,
            "buy_wt1_max": -20.0,
            "buy_renko_max": 22.0,
            "buy_range_max": 30.0,
            "sell_renko_min": 80.0,
            "sell_range_min": 78.0,
            "stop_loss_pct": 1.2
        }

        # 如果提取出了真实大底，取真实大底的 35%~50% 分位数（严格处于超跌区间）
        if not self.swings_low.empty and len(self.swings_low) >= 3:
            if "10m" in self.swings_low.columns and self.swings_low["10m"].notna().sum() >= 3:
                defaults["buy_10m_max"] = round(float(self.swings_low["10m"].quantile(0.40)), 1)
            if "4m" in self.swings_low.columns and self.swings_low["4m"].notna().sum() >= 3:
                defaults["buy_4m_max"] = round(float(self.swings_low["4m"].quantile(0.35)), 1)
            if "renko" in self.swings_low.columns and self.swings_low["renko"].notna().sum() >= 3:
                defaults["buy_renko_max"] = round(float(self.swings_low["renko"].quantile(0.30)), 1)
            if "range" in self.swings_low.columns and self.swings_low["range"].notna().sum() >= 3:
                defaults["buy_range_max"] = round(float(self.swings_low["range"].quantile(0.35)), 1)
            if "wt1" in self.swings_low.columns and self.swings_low["wt1"].notna().sum() >= 3:
                defaults["buy_wt1_max"] = round(float(self.swings_low["wt1"].quantile(0.35)), 1)

        # 真实大顶的 60%~75% 分位数（进入冲顶超买区间）
        if not self.swings_high.empty and len(self.swings_high) >= 3:
            if "renko" in self.swings_high.columns and self.swings_high["renko"].notna().sum() >= 3:
                defaults["sell_renko_min"] = round(float(self.swings_high["renko"].quantile(0.65)), 1)
            if "range" in self.swings_high.columns and self.swings_high["range"].notna().sum() >= 3:
                defaults["sell_range_min"] = round(float(self.swings_high["range"].quantile(0.65)), 1)

        return defaults


# ==============================================================================
# 量化共振与高速向量化回测算法引擎
# ==============================================================================
class MultiChartAlgorithm:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "buy_10m_max": 42.0,
            "buy_4m_max": 30.0,
            "buy_wt1_max": -20.0,
            "buy_renko_max": 20.0,
            "buy_renko_turn_up": True,
            "buy_range_max": 30.0,
            "sell_renko_min": 80.0,
            "sell_renko_turn_dn": True,
            "sell_range_min": 78.0,
            "stop_loss_pct": 1.2
        }

    def evaluate_signals(self, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        p = {**self.get_default_parameters(), **(params or {})}
        df = self.merged_df_copy = self.df.copy()
        n = len(df)

        cond_buy = pd.Series(True, index=df.index)
        if "10m_rg_close" in df.columns:
            cond_buy &= (df["10m_rg_close"] <= p["buy_10m_max"])
        if "4m_01688_rg_close" in df.columns:
            cond_buy &= (df["4m_01688_rg_close"] <= p["buy_4m_max"])
        if "WT1 — 快线信号" in df.columns and "buy_wt1_max" in p:
            cond_buy &= (df["WT1 — 快线信号"] <= p["buy_wt1_max"])
        if "range_4r_rg_close" in df.columns:
            cond_buy &= (df["range_4r_rg_close"] <= p["buy_range_max"])

        if "renko_009_rg_close" in df.columns:
            renko_s = df["renko_009_rg_close"]
            cond_buy &= (renko_s <= p["buy_renko_max"])
            if p.get("buy_renko_turn_up", True):
                cond_turn = (df.get("renko_diff", renko_s.diff()) > 0)
                if "renko_cross_up" in df.columns:
                    cond_turn |= df["renko_cross_up"]
                cond_buy &= cond_turn

        cond_sell = pd.Series(False, index=df.index)
        if "renko_009_rg_close" in df.columns:
            renko_s = df["renko_009_rg_close"]
            sell_base = (renko_s >= p["sell_renko_min"])
            if p.get("sell_renko_turn_dn", True):
                cond_sell_turn = (df.get("renko_diff", renko_s.diff()) < 0)
                if "renko_cross_dn" in df.columns:
                    cond_sell_turn |= df["renko_cross_dn"]
                sell_base &= cond_sell_turn
            cond_sell |= sell_base

        if "range_4r_rg_close" in df.columns:
            range_sell = (df["range_4r_rg_close"] >= p["sell_range_min"])
            cond_sell = cond_sell & range_sell if "renko_009_rg_close" in df.columns else range_sell

        df["raw_buy"] = cond_buy
        df["raw_sell"] = cond_sell
        return df

    def run_backtest(self, params: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        p = {**self.get_default_parameters(), **(params or {})}
        df_sig = self.evaluate_signals(p)

        n = len(df_sig)
        if n == 0:
            return [], self._empty_summary()

        close_arr = df_sig["close"].to_numpy()
        high_arr = df_sig["high"].to_numpy()
        low_arr = df_sig["low"].to_numpy()
        raw_buy = df_sig["raw_buy"].to_numpy()
        raw_sell = df_sig["raw_sell"].to_numpy()
        dt_list = df_sig["dt"].tolist()

        stop_loss_frac = p["stop_loss_pct"] / 100.0
        trades = []
        in_pos = False
        entry_idx = 0
        entry_price = 0.0
        trade_id = 0

        for i in range(n):
            c_price = close_arr[i]
            h_price = high_arr[i]
            l_price = low_arr[i]

            if not in_pos:
                if raw_buy[i]:
                    in_pos = True
                    entry_idx = i
                    entry_price = c_price
                    trade_id += 1
            else:
                mfe_high = high_arr[entry_idx:i+1].max()
                mae_low = low_arr[entry_idx:i+1].min()
                mfe_pct = (mfe_high - entry_price) / entry_price * 100.0
                mae_pct = (entry_price - mae_low) / entry_price * 100.0

                is_stop_loss = (c_price <= entry_price * (1.0 - stop_loss_frac))
                is_exit_signal = raw_sell[i]
                is_last_bar = (i == n - 1)

                if is_stop_loss or is_exit_signal or is_last_bar:
                    exit_price = c_price
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                    exit_reason = "硬止损" if is_stop_loss else ("逃顶信号" if is_exit_signal else "回测收盘离场")

                    trades.append({
                        "trade_id": trade_id,
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_time": str(dt_list[entry_idx]),
                        "exit_time": str(dt_list[i]),
                        "bars_held": i - entry_idx,
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(exit_price, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "mfe_pct": round(mfe_pct, 2),
                        "mae_pct": round(mae_pct, 2),
                        "exit_type": exit_reason
                    })
                    in_pos = False

        summary = self._compute_summary(trades)
        return trades, summary

    def _compute_summary(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trades:
            return self._empty_summary()

        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        mfes = [t["mfe_pct"] for t in trades]
        maes = [t["mae_pct"] for t in trades]

        total_trades = len(trades)
        win_rate = len(wins) / total_trades * 100.0
        cum_pnl = sum(pnls)
        avg_pnl = np.mean(pnls)
        avg_mfe = np.mean(mfes)
        avg_mae = np.mean(maes)
        profit_risk = round(avg_mfe / (avg_mae + 1e-5), 2)

        return {
            "total_trades": total_trades,
            "win_rate_%": round(win_rate, 1),
            "cumulative_pnl_%": round(cum_pnl, 2),
            "avg_pnl_%": round(avg_pnl, 2),
            "avg_mfe_%": round(avg_mfe, 2),
            "avg_mae_%": round(avg_mae, 2),
            "profit_risk_ratio": profit_risk,
            "max_drawdown_%": round(max(maes), 2) if maes else 0.0
        }

    def _empty_summary(self) -> Dict[str, Any]:
        return {
            "total_trades": 0, "win_rate_%": 0.0, "cumulative_pnl_%": 0.0,
            "avg_pnl_%": 0.0, "avg_mfe_%": 0.0, "avg_mae_%": 0.0,
            "profit_risk_ratio": 0.0, "max_drawdown_%": 0.0
        }


# ==============================================================================
# 后台异步工作线程
# ==============================================================================
class StrategyWorkerThread(QThread):
    sig_progress = pyqtSignal(int, str)
    sig_log = pyqtSignal(str, str)
    sig_backtest_done = pyqtSignal(list, dict, object)
    sig_optimize_done = pyqtSignal(list)
    sig_data_analyzed = pyqtSignal(dict, object, object)
    sig_error = pyqtSignal(str)

    def __init__(self, data_dir: str, mode: str = "backtest", params: Optional[Dict[str, Any]] = None, custom_grid: Optional[Dict[str, List[float]]] = None):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode  # "backtest" | "optimize" | "analyze"
        self.params = params or {}
        self.custom_grid = custom_grid

    def run(self):
        try:
            manager = MultiChartDataManager(self.data_dir)
            df = manager.load_and_preprocess(
                log_fn=lambda msg, lvl: self.sig_log.emit(msg, lvl),
                progress_fn=lambda val, txt: self.sig_progress.emit(val, txt)
            )

            # 发送数据全景分布与波段归因结果
            self.sig_data_analyzed.emit(
                manager.indicator_distributions,
                manager.swings_low,
                manager.swings_high
            )

            algo = MultiChartAlgorithm(df)

            if self.mode == "analyze":
                self.sig_progress.emit(100, "数据全景与波段归因分析完成")
                return

            elif self.mode == "backtest":
                self.sig_progress.emit(60, "执行四图共振回测...")
                trades, summary = algo.run_backtest(self.params)
                self.sig_progress.emit(100, "回测计算完成")
                self.sig_backtest_done.emit(trades, summary, df)

            elif self.mode == "optimize":
                self.sig_progress.emit(50, "开始基于真实范围进行精细化超参数寻优...")
                self.sig_log.emit("启动超参数网格扫描...", "INFO")

                grid = self.custom_grid
                if not grid:
                    # 自适应根据数据分布生成寻优空间
                    dist = manager.indicator_distributions
                    r_10m = dist.get("10m_rg_close", {})
                    r_4m = dist.get("4m_01688_rg_close", {})
                    r_renko = dist.get("renko_009_rg_close", {})
                    r_range = dist.get("range_4r_rg_close", {})

                    p_10m = [round(r_10m.get("q10", 35.0), 1), round(r_10m.get("q25", 42.0), 1), round(r_10m.get("median", 50.0), 1)]
                    p_4m = [round(r_4m.get("q10", 25.0), 1), round(r_4m.get("q25", 30.0), 1), round(r_4m.get("median", 35.0), 1)]
                    p_renko_buy = [round(r_renko.get("q05", 10.0), 1), round(r_renko.get("q10", 18.0), 1), round(r_renko.get("q25", 25.0), 1)]
                    p_renko_sell = [round(r_renko.get("q75", 75.0), 1), round(r_renko.get("q90", 82.0), 1), round(r_renko.get("q95", 88.0), 1)]
                    p_range_sell = [round(r_range.get("q75", 72.0), 1), round(r_range.get("q90", 80.0), 1), round(r_range.get("q95", 85.0), 1)]

                    grid = {
                        "10m": sorted(list(set(p_10m))),
                        "4m": sorted(list(set(p_4m))),
                        "renko_buy": sorted(list(set(p_renko_buy))),
                        "renko_sell": sorted(list(set(p_renko_sell))),
                        "range_sell": sorted(list(set(p_range_sell)))
                    }

                opt_results = []
                combos = []
                for b_10 in grid["10m"]:
                    for b_4 in grid["4m"]:
                        for b_ren in grid["renko_buy"]:
                            for s_ren in grid["renko_sell"]:
                                for s_ran in grid["range_sell"]:
                                    combos.append((b_10, b_4, b_ren, s_ren, s_ran))

                total_combos = len(combos)
                self.sig_log.emit(f"总计扫描参数组合数: {total_combos} 组", "INFO")

                for idx, (b_10, b_4, b_ren, s_ren, s_ran) in enumerate(combos):
                    if idx % max(1, total_combos // 20) == 0:
                        prog = 50 + int(idx / total_combos * 48)
                        self.sig_progress.emit(prog, f"寻优进度: {idx}/{total_combos} ({prog}%)")

                    test_p = {
                        "buy_10m_max": b_10,
                        "buy_4m_max": b_4,
                        "buy_wt1_max": self.params.get("buy_wt1_max", -20.0),
                        "buy_renko_max": b_ren,
                        "buy_renko_turn_up": self.params.get("buy_renko_turn_up", True),
                        "buy_range_max": self.params.get("buy_range_max", 30.0),
                        "sell_renko_min": s_ren,
                        "sell_renko_turn_dn": self.params.get("sell_renko_turn_dn", True),
                        "sell_range_min": s_ran,
                        "stop_loss_pct": self.params.get("stop_loss_pct", 1.2)
                    }

                    _, s = algo.run_backtest(test_p)
                    opt_results.append({
                        "10m": b_10, "4m": b_4, "renko_buy": b_ren,
                        "renko_sell": s_ren, "range_sell": s_ran,
                        "trades": s["total_trades"],
                        "win_rate": s["win_rate_%"],
                        "cum_pnl": s["cumulative_pnl_%"],
                        "mfe": s["avg_mfe_%"],
                        "mae": s["avg_mae_%"],
                        "profit_risk": s["profit_risk_ratio"]
                    })

                opt_results.sort(key=lambda x: (x["profit_risk"], x["cum_pnl"], x["win_rate"]), reverse=True)
                self.sig_progress.emit(100, f"网格寻优完成，评估了 {total_combos} 组组合")
                self.sig_optimize_done.emit(opt_results)

        except Exception as e:
            import traceback
            err_details = traceback.format_exc()
            self.sig_log.emit(f"运行发生异常: {str(e)}\n{err_details}", "ERROR")
            self.sig_error.emit(str(e))


# ==============================================================================
# PyQt5 主窗口交互界面 (版本3)
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TradingView 四图表庄家资金指标共振系统 - 版本3 (数据自适应与真实顶底归因版)")
        self.resize(1400, 920)
        self.worker: Optional[StrategyWorkerThread] = None
        self.latest_aligned_df: Optional[pd.DataFrame] = None
        self.latest_trades: List[Dict[str, Any]] = []
        self.latest_opt_results: List[Dict[str, Any]] = []
        self.latest_distributions: Dict[str, Dict[str, float]] = {}

        self.init_ui()
        self.apply_dark_theme()
        # 启动时自动做一次数据探测
        self.run_data_analysis()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # ---------------- 顶部控制栏 ----------------
        top_bar = QGroupBox("数据源设置与全局操作 (Data Source & Control)")
        top_layout = QHBoxLayout(top_bar)

        top_layout.addWidget(QLabel("数据目录:"))
        self.txt_data_dir = QLineEdit(os.path.abspath("."))
        top_layout.addWidget(self.txt_data_dir, stretch=2)

        btn_browse = QPushButton("📁 浏览目录")
        btn_browse.clicked.connect(self.browse_directory)
        top_layout.addWidget(btn_browse)

        btn_detect = QPushButton("🔍 自动检测四图")
        btn_detect.clicked.connect(self.detect_files)
        top_layout.addWidget(btn_detect)

        btn_analyze = QPushButton("🎯 探测数据分布与顶底归因")
        btn_analyze.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        btn_analyze.clicked.connect(self.run_data_analysis)
        top_layout.addWidget(btn_analyze)

        btn_backtest = QPushButton("▶ 启动四图共振回测")
        btn_backtest.setStyleSheet("background-color: #0d47a1; color: white; font-weight: bold;")
        btn_backtest.clicked.connect(self.run_backtest)
        top_layout.addWidget(btn_backtest)

        btn_optimize = QPushButton("⚡ 精细化网格寻优")
        btn_optimize.setStyleSheet("background-color: #e65100; color: white; font-weight: bold;")
        btn_optimize.clicked.connect(self.run_optimization)
        top_layout.addWidget(btn_optimize)

        btn_export = QPushButton("💾 导出信号CSV")
        btn_export.clicked.connect(self.export_signals_csv)
        top_layout.addWidget(btn_export)

        main_layout.addWidget(top_bar)

        # ---------------- 主工作区分割器 ----------------
        splitter = QSplitter(Qt.Horizontal)

        # 左侧面板：参数设定控制台
        left_panel = self.create_left_parameter_panel()
        splitter.addWidget(left_panel)
        splitter.setStretchFactor(0, 0)

        # 右侧面板：多 Tab 视图
        right_panel = self.create_right_tab_panel()
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, stretch=1)

        # ---------------- 底部进度条与状态栏 ----------------
        status_bar = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        status_bar.addWidget(self.progress_bar, stretch=1)

        self.lbl_status = QLabel("就绪")
        status_bar.addWidget(self.lbl_status)
        main_layout.addLayout(status_bar)

    def create_left_parameter_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(390)
        scroll.setMaximumWidth(450)

        container = QWidget()
        param_layout = QVBoxLayout(container)
        param_layout.setContentsMargins(5, 5, 5, 5)
        param_layout.setSpacing(6)

        lbl_title = QLabel("四图表阈值控制面板 (Parameter Panel)")
        lbl_title.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        lbl_title.setStyleSheet("color: #4fc3f7; padding: 4px;")
        param_layout.addWidget(lbl_title)

        # 1. 10m
        grp_10m = QGroupBox("1. 10分钟普通K线图 (宏观蓄势过滤)")
        g10_lay = QGridLayout(grp_10m)
        lbl_10m = QLabel("宏观吸筹门槛 (必须低于 <=):")
        lbl_10m.setToolTip("指标 0~100。设为 42 表示 10m 必须处于 42 以下的吸筹蓄势区才允许做多。")
        g10_lay.addWidget(lbl_10m, 0, 0)
        self.spin_buy_10m = QDoubleSpinBox()
        self.spin_buy_10m.setRange(0, 100)
        self.spin_buy_10m.setValue(42.0)
        g10_lay.addWidget(self.spin_buy_10m, 0, 1)
        param_layout.addWidget(grp_10m)

        # 2. 4m
        grp_4m = QGroupBox("2. 4分钟 0.1688% 砖形图 (中周期 / 每砖0.05)")
        g4_lay = QGridLayout(grp_4m)
        g4_lay.addWidget(QLabel("4分钟波段超跌门槛 (必须 <=):"), 0, 0)
        self.spin_buy_4m = QDoubleSpinBox()
        self.spin_buy_4m.setRange(0, 100)
        self.spin_buy_4m.setValue(30.0)
        g4_lay.addWidget(self.spin_buy_4m, 0, 1)

        g4_lay.addWidget(QLabel("4分钟 WT1超卖门槛 (必须 <=):"), 1, 0)
        self.spin_buy_wt1 = QDoubleSpinBox()
        self.spin_buy_wt1.setRange(-100, 100)
        self.spin_buy_wt1.setValue(-20.0)
        g4_lay.addWidget(self.spin_buy_wt1, 1, 1)
        param_layout.addWidget(grp_4m)

        # 3. 0.09% Renko
        grp_renko = QGroupBox("3. 0.09% 微观砖形图 (极速扳机 / 每砖0.02)")
        gr_lay = QGridLayout(grp_renko)
        gr_lay.addWidget(QLabel("0.09%微观超跌门槛 (必须 <=):"), 0, 0)
        self.spin_buy_renko = QDoubleSpinBox()
        self.spin_buy_renko.setRange(0, 100)
        self.spin_buy_renko.setValue(20.0)
        gr_lay.addWidget(self.spin_buy_renko, 0, 1)

        self.chk_renko_turn_up = QCheckBox("要求跌透后第一拐向上 (启动前开枪)")
        self.chk_renko_turn_up.setChecked(True)
        gr_lay.addWidget(self.chk_renko_turn_up, 1, 0, 1, 2)

        gr_lay.addWidget(QLabel("0.09%微观逃顶门槛 (必须 >=):"), 2, 0)
        self.spin_sell_renko = QDoubleSpinBox()
        self.spin_sell_renko.setRange(0, 100)
        self.spin_sell_renko.setValue(80.0)
        gr_lay.addWidget(self.spin_sell_renko, 2, 1)

        self.chk_renko_turn_dn = QCheckBox("要求冲顶后第一拐向下 (最高点卖出)")
        self.chk_renko_turn_dn.setChecked(True)
        gr_lay.addWidget(self.chk_renko_turn_dn, 3, 0, 1, 2)
        param_layout.addWidget(grp_renko)

        # 4. Range
        grp_range = QGroupBox("4. Range 价格结构图 (纯空间确认)")
        g4r_lay = QGridLayout(grp_range)
        g4r_lay.addWidget(QLabel("抄底超跌门槛 (必须 <=):"), 0, 0)
        self.spin_buy_range = QDoubleSpinBox()
        self.spin_buy_range.setRange(0, 100)
        self.spin_buy_range.setValue(30.0)
        g4r_lay.addWidget(self.spin_buy_range, 0, 1)

        g4r_lay.addWidget(QLabel("逃顶超买门槛 (必须 >=):"), 1, 0)
        self.spin_sell_range = QDoubleSpinBox()
        self.spin_sell_range.setRange(0, 100)
        self.spin_sell_range.setValue(78.0)
        g4r_lay.addWidget(self.spin_sell_range, 1, 1)
        param_layout.addWidget(grp_range)

        # 5. 风控
        grp_risk = QGroupBox("5. 风险控制与出场")
        risk_lay = QGridLayout(grp_risk)
        risk_lay.addWidget(QLabel("最大硬止损比例 (%):"), 0, 0)
        self.spin_stop_loss = QDoubleSpinBox()
        self.spin_stop_loss.setRange(0.1, 20.0)
        self.spin_stop_loss.setSingleStep(0.1)
        self.spin_stop_loss.setValue(1.2)
        risk_lay.addWidget(self.spin_stop_loss, 0, 1)
        param_layout.addWidget(grp_risk)

        # 快捷按钮
        btn_apply_real = QPushButton("🎯 根据真实顶底归因一键填充参数")
        btn_apply_real.setStyleSheet("background-color: #388e3c; color: white; font-weight: bold; padding: 6px;")
        btn_apply_real.clicked.connect(self.apply_real_attribution_recommendation)
        param_layout.addWidget(btn_apply_real)

        btn_reset_defaults = QPushButton("↺ 恢复理论黄金默认参数")
        btn_reset_defaults.clicked.connect(self.reset_to_golden_defaults)
        param_layout.addWidget(btn_reset_defaults)

        param_layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def create_right_tab_panel(self) -> QWidget:
        self.tabs = QTabWidget()

        # Tab 1: 数据全景分布与真实顶底归因 (版本3核心亮点)
        self.tab_attribution = self.create_attribution_tab()
        self.tabs.addTab(self.tab_attribution, "🎯 真实数据范围与顶底归因 (版本3核心)")

        # Tab 2: 绩效概览与交易明细
        self.tab_performance = self.create_performance_tab()
        self.tabs.addTab(self.tab_performance, "📊 回测绩效概览与交易明细")

        # Tab 3: 四图时序与共振信号
        self.tab_signals = self.create_signals_tab()
        self.tabs.addTab(self.tab_signals, "📈 四图时序对齐与信号流")

        # Tab 4: 自定义与自适应网格寻优榜
        self.tab_optimizer = self.create_optimizer_tab()
        self.tabs.addTab(self.tab_optimizer, "⚡ 精细化网格寻优榜")

        # Tab 5: 运行日志终端
        self.tab_log = self.create_log_tab()
        self.tabs.addTab(self.tab_log, "📝 运行日志终端 (Log)")

        return self.tabs

    def create_attribution_tab(self) -> QWidget:
        """创建真实数据分布与真实波段归因标签页。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # 上部：四图指标全景分位数统计表
        grp_dist = QGroupBox("1. 四图指标真实全景统计与数据范围分布 (Data Range & Percentiles)")
        dist_lay = QVBoxLayout(grp_dist)
        self.tbl_distribution = QTableWidget()
        self.tbl_distribution.setColumnCount(9)
        self.tbl_distribution.setHorizontalHeaderLabels([
            "图表指标", "最小值 (Min)", "5%极度超跌", "25%超跌区", "中位数 (50%)", "75%超买区", "95%极度冲顶", "最大值 (Max)", "当前设定建议"
        ])
        self.tbl_distribution.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_distribution.verticalHeader().setVisible(False)
        dist_lay.addWidget(self.tbl_distribution)
        layout.addWidget(grp_dist, stretch=2)

        # 下部：真实历史大底（波谷）与真实大顶（波峰）画像表
        grp_swings = QGroupBox("2. 历史真实波谷(起爆点)与波峰(逃顶点)精确数值画像 (Reverse Swing Attribution)")
        swings_lay = QVBoxLayout(grp_swings)

        sub_splitter = QSplitter(Qt.Horizontal)

        # 波谷起爆点明细
        box_lows = QGroupBox("▼ 真实波谷起爆点 (价格大底时四图具体数值)")
        lay_l = QVBoxLayout(box_lows)
        self.tbl_swings_low = QTableWidget()
        self.tbl_swings_low.setColumnCount(8)
        self.tbl_swings_low.setHorizontalHeaderLabels(["时间", "最低价", "后涨幅%", "10m值", "4m值", "Renko值", "Range值", "WT1值"])
        self.tbl_swings_low.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        lay_l.addWidget(self.tbl_swings_low)
        sub_splitter.addWidget(box_lows)

        # 波峰逃顶点明细
        box_highs = QGroupBox("▲ 真实波峰逃顶点 (价格见顶时四图具体数值)")
        lay_h = QVBoxLayout(box_highs)
        self.tbl_swings_high = QTableWidget()
        self.tbl_swings_high.setColumnCount(8)
        self.tbl_swings_high.setHorizontalHeaderLabels(["时间", "最高价", "后跌幅%", "10m值", "4m值", "Renko值", "Range值", "WT1值"])
        self.tbl_swings_high.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        lay_h.addWidget(self.tbl_swings_high)
        sub_splitter.addWidget(box_highs)

        swings_lay.addWidget(sub_splitter)
        layout.addWidget(grp_swings, stretch=3)

        return widget

    def create_performance_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # KPI 卡片
        kpi_frame = QFrame()
        kpi_frame.setStyleSheet("background-color: #1e222d; border-radius: 6px; padding: 6px;")
        kpi_lay = QGridLayout(kpi_frame)

        self.kpi_labels = {}
        items = [
            ("累计收益率 (PnL)", "cum_pnl", "#00e676"),
            ("策略胜率 (Win Rate)", "win_rate", "#4fc3f7"),
            ("盈亏风险比 (Profit/Risk)", "profit_risk", "#ffd600"),
            ("平均最大浮亏 (MAE)", "mae", "#ff5252"),
            ("平均潜在涨幅 (MFE)", "mfe", "#69f0ae"),
            ("总交易笔数 (Trades)", "trades", "#ffffff")
        ]

        for i, (title, key, color) in enumerate(items):
            r, c = i // 3, (i % 3) * 2
            lbl_t = QLabel(f"{title}:")
            lbl_t.setStyleSheet("color: #90a4ae; font-size: 12px;")
            lbl_v = QLabel("--")
            lbl_v.setFont(QFont("Segoe UI", 13, QFont.Bold))
            lbl_v.setStyleSheet(f"color: {color};")
            kpi_lay.addWidget(lbl_t, r, c)
            kpi_lay.addWidget(lbl_v, r, c + 1)
            self.kpi_labels[key] = lbl_v

        layout.addWidget(kpi_frame)

        # 交易明细表
        layout.addWidget(QLabel("交易执行记录 (Trade History):"))
        self.tbl_trades = QTableWidget()
        self.tbl_trades.setColumnCount(11)
        self.tbl_trades.setHorizontalHeaderLabels([
            "序号", "入场时间", "出场时间", "持仓柱数", "买入价", "卖出价", "收益率 %", "潜在涨幅 MFE%", "最大回撤 MAE%", "离场类型", "盈亏状态"
        ])
        self.tbl_trades.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.tbl_trades)

        return widget

    def create_signals_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.tbl_signals = QTableWidget()
        self.tbl_signals.setColumnCount(10)
        self.tbl_signals.setHorizontalHeaderLabels([
            "时间 (Asia/Shanghai)", "收盘价", "10m 红绿值", "4m 红绿值", "4m WT1快线", "Renko 0.09%", "Range 4R", "抄底触发", "逃顶触发", "持仓状态"
        ])
        self.tbl_signals.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.tbl_signals)
        return widget

    def create_optimizer_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 寻优范围与步长自定义工具条
        opt_cfg_box = QGroupBox("自适应 / 自定义寻优空间设定 (Optimization Search Bounds & Step)")
        opt_cfg_lay = QHBoxLayout(opt_cfg_box)

        btn_auto_space = QPushButton("⚡ 根据当前数据范围自适应生成网格")
        btn_auto_space.setStyleSheet("background-color: #00897b; color: white; font-weight: bold; padding: 5px;")
        btn_auto_space.clicked.connect(self.run_optimization)
        opt_cfg_lay.addWidget(btn_auto_space)

        btn_apply_opt = QPushButton("✔ 将所选行参数应用至控制面板")
        btn_apply_opt.setStyleSheet("background-color: #3949ab; color: white; font-weight: bold; padding: 5px;")
        btn_apply_opt.clicked.connect(self.apply_selected_opt_params)
        opt_cfg_lay.addWidget(btn_apply_opt)

        layout.addWidget(opt_cfg_box)

        # 寻优榜
        self.tbl_opt = QTableWidget()
        self.tbl_opt.setColumnCount(11)
        self.tbl_opt.setHorizontalHeaderLabels([
            "10m上限", "4m上限", "Renko超跌", "Renko逃顶", "Range逃顶", "交易次数", "胜率 %", "累计收益 %", "平均MFE %", "平均MAE %", "盈亏风险比"
        ])
        self.tbl_opt.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_opt.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.tbl_opt)

        return widget

    def create_log_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 10))
        layout.addWidget(self.txt_log)

        btn_clear_log = QPushButton("清空日志")
        btn_clear_log.clicked.connect(lambda: self.txt_log.clear())
        layout.addWidget(btn_clear_log)
        return widget

    # ---------------- 业务槽函数 ----------------
    def log(self, message: str, level: str = "INFO"):
        color_map = {
            "INFO": "#81d4fa",
            "SUCCESS": "#00e676",
            "WARN": "#ffb74d",
            "ERROR": "#ff5252"
        }
        color = color_map.get(level, "#ffffff")
        from datetime import datetime
        time_str = datetime.now().strftime("%H:%M:%S")
        html_msg = f'<span style="color: #78909c;">[{time_str}]</span> <span style="color: {color}; font-weight: bold;">[{level}]</span> {message}'
        self.txt_log.appendHtml(html_msg)

    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据所在文件夹", self.txt_data_dir.text())
        if dir_path:
            self.txt_data_dir.setText(dir_path)
            self.run_data_analysis()

    def detect_files(self):
        try:
            m = MultiChartDataManager(self.txt_data_dir.text().strip())
            fmap = m.auto_detect_files()
            if not fmap:
                QMessageBox.warning(self, "检测提示", "未找到符合格式的四图表 CSV 文件！")
                return
            msg = "成功检测到以下图表文件：\n\n" + "\n".join([f"• [{k}]: {os.path.basename(v)}" for k, v in fmap.items()])
            QMessageBox.information(self, "检测成功", msg)
        except Exception as e:
            QMessageBox.critical(self, "检测失败", str(e))

    def reset_to_golden_defaults(self):
        defaults = MultiChartAlgorithm.get_default_parameters()
        self.spin_buy_10m.setValue(defaults["buy_10m_max"])
        self.spin_buy_4m.setValue(defaults["buy_4m_max"])
        self.spin_buy_wt1.setValue(defaults["buy_wt1_max"])
        self.spin_buy_renko.setValue(defaults["buy_renko_max"])
        self.spin_buy_range.setValue(defaults["buy_range_max"])
        self.spin_sell_renko.setValue(defaults["sell_renko_min"])
        self.spin_sell_range.setValue(defaults["sell_range_min"])
        self.spin_stop_loss.setValue(defaults["stop_loss_pct"])
        self.log("已恢复理论黄金默认参数。", "INFO")

    def apply_real_attribution_recommendation(self):
        """根据真实数据分布与真实波段极值归因，自动填充推荐阈值。"""
        if not self.latest_distributions:
            self.run_data_analysis()
            return

        m = MultiChartDataManager(self.txt_data_dir.text().strip())
        m.indicator_distributions = self.latest_distributions
        # 使用当前界面缓存的波段画像
        if hasattr(self, "latest_swings_low") and hasattr(self, "latest_swings_high"):
            m.swings_low = self.latest_swings_low
            m.swings_high = self.latest_swings_high

        recs = m.get_data_adaptive_recommendations()
        self.spin_buy_10m.setValue(recs["buy_10m_max"])
        self.spin_buy_4m.setValue(recs["buy_4m_max"])
        self.spin_buy_wt1.setValue(recs["buy_wt1_max"])
        self.spin_buy_renko.setValue(recs["buy_renko_max"])
        self.spin_buy_range.setValue(recs["buy_range_max"])
        self.spin_sell_renko.setValue(recs["sell_renko_min"])
        self.spin_sell_range.setValue(recs["sell_range_min"])

        self.log(
            f"🎯 成功根据真实极值归因填充参数！10m<={recs['buy_10m_max']}, 4m<={recs['buy_4m_max']}, "
            f"Renko底<={recs['buy_renko_max']}, Renko顶>={recs['sell_renko_min']}, Range顶>={recs['sell_range_min']}",
            "SUCCESS"
        )
        QMessageBox.information(self, "自适应参数应用成功", "已根据当前数据集真实波段起爆点与见顶点的画像，自动为您填充最佳门槛！")

    def run_data_analysis(self):
        """仅做数据全景分布与真实波段归因分析。"""
        self.start_worker(mode="analyze")

    def run_backtest(self):
        params = {
            "buy_10m_max": self.spin_buy_10m.value(),
            "buy_4m_max": self.spin_buy_4m.value(),
            "buy_wt1_max": self.spin_buy_wt1.value(),
            "buy_renko_max": self.spin_buy_renko.value(),
            "buy_renko_turn_up": self.chk_renko_turn_up.isChecked(),
            "buy_range_max": self.spin_buy_range.value(),
            "sell_renko_min": self.spin_sell_renko.value(),
            "sell_renko_turn_dn": self.chk_renko_turn_dn.isChecked(),
            "sell_range_min": self.spin_sell_range.value(),
            "stop_loss_pct": self.spin_stop_loss.value()
        }
        self.start_worker(mode="backtest", params=params)

    def run_optimization(self):
        params = {
            "buy_wt1_max": self.spin_buy_wt1.value(),
            "buy_range_max": self.spin_buy_range.value(),
            "buy_renko_turn_up": self.chk_renko_turn_up.isChecked(),
            "sell_renko_turn_dn": self.chk_renko_turn_dn.isChecked(),
            "stop_loss_pct": self.spin_stop_loss.value()
        }
        self.start_worker(mode="optimize", params=params)

    def start_worker(self, mode: str, params: Optional[Dict[str, Any]] = None):
        if self.worker and self.worker.isRunning():
            self.log("后台任务正在运行中，请稍候...", "WARN")
            return

        data_dir = self.txt_data_dir.text().strip()
        self.worker = StrategyWorkerThread(data_dir=data_dir, mode=mode, params=params)
        self.worker.sig_progress.connect(self.on_progress)
        self.worker.sig_log.connect(self.log)
        self.worker.sig_data_analyzed.connect(self.on_data_analyzed)
        self.worker.sig_backtest_done.connect(self.on_backtest_done)
        self.worker.sig_optimize_done.connect(self.on_optimize_done)
        self.worker.sig_error.connect(lambda err: QMessageBox.critical(self, "运行出错", err))
        self.worker.start()

    def on_progress(self, val: int, txt: str):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(txt)

    def on_data_analyzed(self, dist: Dict[str, Dict[str, float]], df_lows: pd.DataFrame, df_highs: pd.DataFrame):
        self.latest_distributions = dist
        self.latest_swings_low = df_lows
        self.latest_swings_high = df_highs

        # 刷新全景分布表
        self.tbl_distribution.setRowCount(len(dist))
        for r, (col, d) in enumerate(dist.items()):
            label = d.get("label", col)
            self.tbl_distribution.setItem(r, 0, QTableWidgetItem(label))
            self.tbl_distribution.setItem(r, 1, QTableWidgetItem(f"{d['min']:.1f}"))
            self.tbl_distribution.setItem(r, 2, QTableWidgetItem(f"{d['q05']:.1f}"))
            self.tbl_distribution.setItem(r, 3, QTableWidgetItem(f"{d['q25']:.1f}"))
            self.tbl_distribution.setItem(r, 4, QTableWidgetItem(f"{d['median']:.1f}"))
            self.tbl_distribution.setItem(r, 5, QTableWidgetItem(f"{d['q75']:.1f}"))
            self.tbl_distribution.setItem(r, 6, QTableWidgetItem(f"{d['q95']:.1f}"))
            self.tbl_distribution.setItem(r, 7, QTableWidgetItem(f"{d['max']:.1f}"))

            # 建议操作区间
            if "wt1" in col.lower():
                sugg = f"超卖 <= {d['q25']:.1f} | 超买 >= {d['q75']:.1f}"
            elif "renko" in col:
                sugg = f"抄底 <= {d['q25']:.1f} | 逃顶 >= {d['q75']:.1f}"
            elif "10m" in col:
                sugg = f"吸筹 <= {d['q25']:.1f} | 派发 >= {d['q75']:.1f}"
            else:
                sugg = f"超跌 <= {d['q25']:.1f} | 超买 >= {d['q75']:.1f}"
            self.tbl_distribution.setItem(r, 8, QTableWidgetItem(sugg))

        # 刷新真实大底表
        self.tbl_swings_low.setRowCount(len(df_lows))
        for r, row in df_lows.iterrows():
            self.tbl_swings_low.setItem(r, 0, QTableWidgetItem(str(row["dt"])))
            self.tbl_swings_low.setItem(r, 1, QTableWidgetItem(f"{row['price']:.2f}"))
            item_c = QTableWidgetItem(f"+{row['change_pct']:.2f}%")
            item_c.setForeground(QColor("#00e676"))
            self.tbl_swings_low.setItem(r, 2, item_c)
            self.tbl_swings_low.setItem(r, 3, QTableWidgetItem(f"{row['10m']}"))
            self.tbl_swings_low.setItem(r, 4, QTableWidgetItem(f"{row['4m']}"))
            self.tbl_swings_low.setItem(r, 5, QTableWidgetItem(f"{row['renko']}"))
            self.tbl_swings_low.setItem(r, 6, QTableWidgetItem(f"{row['range']}"))
            self.tbl_swings_low.setItem(r, 7, QTableWidgetItem(f"{row['wt1']}"))

        # 刷新真实大顶表
        self.tbl_swings_high.setRowCount(len(df_highs))
        for r, row in df_highs.iterrows():
            self.tbl_swings_high.setItem(r, 0, QTableWidgetItem(str(row["dt"])))
            self.tbl_swings_high.setItem(r, 1, QTableWidgetItem(f"{row['price']:.2f}"))
            item_c = QTableWidgetItem(f"{row['change_pct']:.2f}%")
            item_c.setForeground(QColor("#ff5252"))
            self.tbl_swings_high.setItem(r, 2, item_c)
            self.tbl_swings_high.setItem(r, 3, QTableWidgetItem(f"{row['10m']}"))
            self.tbl_swings_high.setItem(r, 4, QTableWidgetItem(f"{row['4m']}"))
            self.tbl_swings_high.setItem(r, 5, QTableWidgetItem(f"{row['renko']}"))
            self.tbl_swings_high.setItem(r, 6, QTableWidgetItem(f"{row['range']}"))
            self.tbl_swings_high.setItem(r, 7, QTableWidgetItem(f"{row['wt1']}"))

    def on_backtest_done(self, trades: List[Dict[str, Any]], summary: Dict[str, Any], df: pd.DataFrame):
        self.latest_trades = trades
        self.latest_aligned_df = df

        # 更新 KPI
        self.kpi_labels["cum_pnl"].setText(f"{summary['cumulative_pnl_%']:+.2f}%")
        self.kpi_labels["win_rate"].setText(f"{summary['win_rate_%']:.1f}%")
        self.kpi_labels["profit_risk"].setText(f"{summary['profit_risk_ratio']:.2f}")
        self.kpi_labels["mae"].setText(f"{summary['avg_mae_%']:.2f}%")
        self.kpi_labels["mfe"].setText(f"{summary['avg_mfe_%']:.2f}%")
        self.kpi_labels["trades"].setText(str(summary["total_trades"]))

        # 更新交易表格
        self.tbl_trades.setRowCount(len(trades))
        for r, t in enumerate(trades):
            self.tbl_trades.setItem(r, 0, QTableWidgetItem(str(t["trade_id"])))
            self.tbl_trades.setItem(r, 1, QTableWidgetItem(t["entry_time"]))
            self.tbl_trades.setItem(r, 2, QTableWidgetItem(t["exit_time"]))
            self.tbl_trades.setItem(r, 3, QTableWidgetItem(str(t["bars_held"])))
            self.tbl_trades.setItem(r, 4, QTableWidgetItem(f"{t['entry_price']:.2f}"))
            self.tbl_trades.setItem(r, 5, QTableWidgetItem(f"{t['exit_price']:.2f}"))

            pnl_item = QTableWidgetItem(f"{t['pnl_pct']:+.2f}%")
            pnl_color = "#00e676" if t["pnl_pct"] > 0 else "#ff5252"
            pnl_item.setForeground(QColor(pnl_color))
            self.tbl_trades.setItem(r, 6, pnl_item)

            self.tbl_trades.setItem(r, 7, QTableWidgetItem(f"{t['mfe_pct']:.2f}%"))
            self.tbl_trades.setItem(r, 8, QTableWidgetItem(f"{t['mae_pct']:.2f}%"))
            self.tbl_trades.setItem(r, 9, QTableWidgetItem(t["exit_type"]))

            status_item = QTableWidgetItem("盈利" if t["pnl_pct"] > 0 else "亏损")
            status_item.setForeground(QColor(pnl_color))
            self.tbl_trades.setItem(r, 10, status_item)

        # 刷新信号时序预览表
        self.populate_signals_table(df)
        self.tabs.setCurrentIndex(1)  # 切换到回测绩效页

    def populate_signals_table(self, df: pd.DataFrame):
        show_df = df.tail(300)
        self.tbl_signals.setRowCount(len(show_df))
        for r, (_, row) in enumerate(show_df.iterrows()):
            self.tbl_signals.setItem(r, 0, QTableWidgetItem(str(row["dt"])))
            self.tbl_signals.setItem(r, 1, QTableWidgetItem(f"{row['close']:.2f}"))
            self.tbl_signals.setItem(r, 2, QTableWidgetItem(f"{row.get('10m_rg_close', np.nan):.1f}"))
            self.tbl_signals.setItem(r, 3, QTableWidgetItem(f"{row.get('4m_01688_rg_close', np.nan):.1f}"))
            self.tbl_signals.setItem(r, 4, QTableWidgetItem(f"{row.get('WT1 — 快线信号', np.nan):.1f}"))
            self.tbl_signals.setItem(r, 5, QTableWidgetItem(f"{row.get('renko_009_rg_close', np.nan):.1f}"))
            self.tbl_signals.setItem(r, 6, QTableWidgetItem(f"{row.get('range_4r_rg_close', np.nan):.1f}"))

            buy_str = "★ 抄底" if row.get("raw_buy", False) else "--"
            item_b = QTableWidgetItem(buy_str)
            if row.get("raw_buy", False):
                item_b.setForeground(QColor("#00e676"))
            self.tbl_signals.setItem(r, 7, item_b)

            sell_str = "▼ 逃顶" if row.get("raw_sell", False) else "--"
            item_s = QTableWidgetItem(sell_str)
            if row.get("raw_sell", False):
                item_s.setForeground(QColor("#ff5252"))
            self.tbl_signals.setItem(r, 8, item_s)
            self.tbl_signals.setItem(r, 9, QTableWidgetItem("--"))

    def on_optimize_done(self, results: List[Dict[str, Any]]):
        self.latest_opt_results = results
        self.tbl_opt.setRowCount(min(len(results), 200))
        for r, res in enumerate(results[:200]):
            self.tbl_opt.setItem(r, 0, QTableWidgetItem(f"{res['10m']:.1f}"))
            self.tbl_opt.setItem(r, 1, QTableWidgetItem(f"{res['4m']:.1f}"))
            self.tbl_opt.setItem(r, 2, QTableWidgetItem(f"{res['renko_buy']:.1f}"))
            self.tbl_opt.setItem(r, 3, QTableWidgetItem(f"{res['renko_sell']:.1f}"))
            self.tbl_opt.setItem(r, 4, QTableWidgetItem(f"{res['range_sell']:.1f}"))
            self.tbl_opt.setItem(r, 5, QTableWidgetItem(str(res["trades"])))
            self.tbl_opt.setItem(r, 6, QTableWidgetItem(f"{res['win_rate']:.1f}%"))

            pnl_item = QTableWidgetItem(f"{res['cum_pnl']:+.2f}%")
            pnl_item.setForeground(QColor("#00e676" if res['cum_pnl'] > 0 else "#ff5252"))
            self.tbl_opt.setItem(r, 7, pnl_item)

            self.tbl_opt.setItem(r, 8, QTableWidgetItem(f"{res['mfe']:.2f}%"))
            self.tbl_opt.setItem(r, 9, QTableWidgetItem(f"{res['mae']:.2f}%"))
            self.tbl_opt.setItem(r, 10, QTableWidgetItem(f"{res['profit_risk']:.2f}"))

        self.tabs.setCurrentIndex(3)  # 切换到寻优页
        self.log(f"已展示 TOP {min(len(results), 200)} 组最优参数排行榜。", "SUCCESS")

    def apply_selected_opt_params(self):
        row = self.tbl_opt.currentRow()
        if row < 0 or row >= len(self.latest_opt_results):
            QMessageBox.warning(self, "提示", "请先在寻优表格中用鼠标点击选中一行参数！")
            return

        res = self.latest_opt_results[row]
        self.spin_buy_10m.setValue(res["10m"])
        self.spin_buy_4m.setValue(res["4m"])
        self.spin_buy_renko.setValue(res["renko_buy"])
        self.spin_sell_renko.setValue(res["renko_sell"])
        self.spin_sell_range.setValue(res["range_sell"])
        self.log(f"已将排行榜第 {row + 1} 行参数应用至控制面板！", "SUCCESS")
        QMessageBox.information(self, "参数应用成功", f"已成功将寻优榜参数应用至左侧控制面板！\n盈亏风险比: {res['profit_risk']:.2f}, 累计收益: {res['cum_pnl']:+.2f}%")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(500)
        event.accept()

    def export_signals_csv(self):
        if self.latest_aligned_df is None:
            QMessageBox.warning(self, "提示", "请先点击【启动四图共振回测】生成信号后再导出！")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "导出对齐信号文件", "aligned_signals.csv", "CSV Files (*.csv)")
        if save_path:
            self.latest_aligned_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            self.log(f"成功导出信号数据至: {save_path}", "SUCCESS")
            QMessageBox.information(self, "导出成功", f"已成功导出对齐时序信号至:\n{save_path}")

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #131722;
                color: #d1d4dc;
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #2a2e39;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #90caf9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background-color: #1e222d;
                border: 1px solid #363c4e;
                border-radius: 4px;
                color: #ffffff;
                padding: 4px 6px;
                min-height: 22px;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {
                border: 1px solid #2962ff;
            }
            QPushButton {
                background-color: #2a2e39;
                border: 1px solid #363c4e;
                border-radius: 4px;
                color: #d1d4dc;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #363c4e;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #1e222d;
            }
            QTableWidget {
                background-color: #131722;
                gridline-color: #2a2e39;
                border: 1px solid #2a2e39;
                selection-background-color: #1976d2;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                background-color: #1e222d;
                color: #90a4ae;
                padding: 6px;
                border: 1px solid #2a2e39;
                font-weight: bold;
            }
            QTabWidget::pane {
                border: 1px solid #2a2e39;
                background-color: #131722;
            }
            QTabBar::tab {
                background-color: #1e222d;
                color: #90a4ae;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #2962ff;
                color: #ffffff;
            }
            QTabBar::tab:hover:!selected {
                background-color: #2a2e39;
                color: #ffffff;
            }
            QProgressBar {
                border: 1px solid #2a2e39;
                border-radius: 4px;
                text-align: center;
                background-color: #1e222d;
                color: #ffffff;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #00e676;
                border-radius: 3px;
            }
            QPlainTextEdit {
                background-color: #0c0d12;
                border: 1px solid #2a2e39;
                color: #d1d4dc;
            }
        """)


# ==============================================================================
# 程序启动入口
# ==============================================================================
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
