"""
================================================================================
TradingView 四图表庄家资金指标 (L3 Banker Fund) 算法共振策略桌面系统 (版本2 - PyQt5版)
Multi-Chart Confluence Trading Desktop Engine - PyQt5 Edition
================================================================================
功能特性：
1. 完整的图形化交互界面 (GUI)，支持参数灵活设定与预设保存
2. 后台 QThread 多线程运行，界面完全不卡顿
3. 实时细粒度进度条 (QProgressBar) 与步骤状态反馈
4. 结构化日志系统 (Log Console)，支持级别色彩区分、清空与导出
5. 四图表跨形态对齐（10m 普通K线、4m 0.1688%砖形图、0.09%微观砖形图、Range 4R纯结构图）
6. 绩效 KPI 指标卡与交易明细表、信号时序预览表
7. 自动化超参数网格扫描寻优 (Grid Search Optimization)，支持一键将最优参数应用至面板
"""

import os
import sys
import glob
import json
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QDoubleSpinBox, QCheckBox,
    QPushButton, QProgressBar, QPlainTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QTabWidget, QFileDialog,
    QMessageBox, QSplitter, QFrame
)

# 终端编码兼容
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_timestamp_to_epoch(val) -> float:
    """将 ISO 字符串或 Unix 时间戳转换为浮点秒数。"""
    try:
        return float(val)
    except (ValueError, TypeError):
        dt = pd.to_datetime(val)
        return dt.timestamp()


# ==============================================================================
# 数据管理核心类
# ==============================================================================
class MultiChartDataManager:
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.raw_dfs: Dict[str, pd.DataFrame] = {}
        self.merged_df: Optional[pd.DataFrame] = None

    def auto_detect_files(self) -> Dict[str, str]:
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        file_map = {}

        for f in csv_files:
            basename = os.path.basename(f)
            if "10_" in basename or "10m" in basename.lower():
                file_map["10m"] = f
            elif "4R" in basename or "4r" in basename.lower():
                file_map["range_4r"] = f
            elif "2_" in basename or "0.09" in basename:
                file_map["renko_009"] = f
            elif "4_" in basename or "0.1688" in basename:
                file_map["4m_01688"] = f

        if len(file_map) < 4:
            for f in csv_files:
                if f in file_map.values():
                    continue
                try:
                    df_temp = pd.read_csv(f, nrows=10)
                    cols_str = " ".join(df_temp.columns)
                    if "WT1" in cols_str and "4m_01688" not in file_map:
                        file_map["4m_01688"] = f
                    elif "通道位置" in cols_str and "renko_009" not in file_map:
                        file_map["renko_009"] = f
                    elif "反转形状" in cols_str and "range_4r" not in file_map:
                        file_map["range_4r"] = f
                    elif "EMA" in cols_str and "10m" not in file_map:
                        file_map["10m"] = f
                except Exception:
                    pass

        return file_map

    def load_and_preprocess(self, file_map: Optional[Dict[str, str]] = None, log_fn=None, progress_fn=None) -> pd.DataFrame:
        if file_map is None:
            file_map = self.auto_detect_files()

        if log_fn:
            log_fn(f"开始加载四图表数据文件，目标目录: {os.path.abspath(self.data_dir)}", "INFO")

        dfs = {}
        step = 0
        total_steps = len(file_map) + 2

        for key, path in file_map.items():
            step += 1
            if progress_fn:
                progress_fn(int(step / total_steps * 40), f"正在读取并解析: {os.path.basename(path)}")
            if log_fn:
                log_fn(f"-> 识别到图表 [{key}]: {os.path.basename(path)}", "INFO")

            df = pd.read_csv(path)
            df["epoch"] = df["time"].apply(parse_timestamp_to_epoch)
            df["dt"] = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
            df = df.sort_values("epoch").reset_index(drop=True)

            rename_dict = {}
            for col in df.columns:
                if "red_green" in col:
                    if "关" in col or "close" in col.lower():
                        rename_dict[col] = f"{key}_rg_close"
                    elif "开" in col or "open" in col.lower():
                        rename_dict[col] = f"{key}_rg_open"
                    elif "高" in col or "high" in col.lower():
                        rename_dict[col] = f"{key}_rg_high"
                    elif "低" in col or "low" in col.lower():
                        rename_dict[col] = f"{key}_rg_low"
                elif "cyan_magenta" in col:
                    if "关" in col or "close" in col.lower():
                        rename_dict[col] = f"{key}_cm_close"
                    elif "开" in col or "open" in col.lower():
                        rename_dict[col] = f"{key}_cm_open"

            df = df.rename(columns=rename_dict)
            dfs[key] = df
            self.raw_dfs[key] = df

        if progress_fn:
            progress_fn(45, "正在进行因果无未来函数时间对齐 (merge_asof)...")
        if log_fn:
            log_fn("执行多周期时间轴因果同步...", "INFO")

        base_key = "renko_009" if "renko_009" in dfs else list(dfs.keys())[0]
        base = dfs[base_key][
            ["epoch", "dt", "open", "high", "low", "close", f"{base_key}_rg_close", f"{base_key}_cm_close"]
        ].copy()

        for extra_col in ["需求强度 %", "供给强度 %", "成交量压力 %", "趋势力度 %", "通道位置"]:
            if extra_col in dfs[base_key].columns:
                base[f"renko_{extra_col}"] = dfs[base_key][extra_col]

        for key in ["10m", "4m_01688", "range_4r"]:
            if key in dfs and key != base_key:
                other_df = dfs[key]
                merge_cols = ["epoch"]
                if f"{key}_rg_close" in other_df.columns:
                    merge_cols.append(f"{key}_rg_close")
                if f"{key}_cm_close" in other_df.columns:
                    merge_cols.append(f"{key}_cm_close")

                if key == "4m_01688":
                    for c in ["WT1 — 快线信号", "WT2 — 慢线确认", "WT 差值（动量）"]:
                        if c in other_df.columns:
                            merge_cols.append(c)
                elif key == "range_4r":
                    for c in ["买入形状", "卖出形状", "多头反转形状", "空头反转形状", "支撑", "阻力"]:
                        if c in other_df.columns:
                            merge_cols.append(c)
                elif key == "10m":
                    for c in ["Cloud Reach", "EMA 曲线", "Red %", "Blue %"]:
                        if c in other_df.columns:
                            merge_cols.append(c)

                sub_other = other_df[merge_cols].sort_values("epoch")
                base = pd.merge_asof(base, sub_other, on="epoch", direction="backward")

        if progress_fn:
            progress_fn(55, "计算指标动量与交叉导数...")

        for rg_col in ["10m_rg_close", "4m_01688_rg_close", "renko_009_rg_close", "range_4r_rg_close"]:
            if rg_col in base.columns:
                base[f"{rg_col}_d1"] = base[rg_col].diff()

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
        if log_fn:
            log_fn(f"四图表数据对齐完成！有效行数: {len(base)} 行，时间范围: {base['dt'].iloc[0]} -> {base['dt'].iloc[-1]}", "SUCCESS")
        return base


# ==============================================================================
# 策略算法与回测引擎
# ==============================================================================
class MultiChartAlgorithm:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def get_default_parameters() -> dict:
        return {
            "buy_10m_rg_max": 42.0,
            "buy_4m_rg_max": 30.0,
            "buy_renko_rg_max": 20.0,
            "buy_renko_turn_up": True,
            "buy_range_rg_max": 30.0,
            "buy_wt1_max": -20.0,
            "sell_renko_rg_min": 80.0,
            "sell_range_rg_min": 78.0,
            "sell_renko_turn_dn": True,
            "sell_4m_rg_min": 35.0,
            "sell_wt1_min": 35.0,
            "stop_loss_pct": 1.2,
        }

    def evaluate_signals(self, params: dict) -> pd.DataFrame:
        df = self.df.copy()

        df["renko_min_past5"] = df["renko_009_rg_close"].rolling(6, min_periods=1).min()

        cond_10m_buy = df["10m_rg_close"] <= params["buy_10m_rg_max"]
        cond_4m_buy = df["4m_01688_rg_close"] <= params["buy_4m_rg_max"]
        cond_renko_oversold = (df["renko_009_rg_close"] <= params["buy_renko_rg_max"]) | (
            df["renko_min_past5"] <= 15.0
        )

        if params.get("buy_renko_turn_up", True):
            cond_renko_trigger = (df["renko_009_rg_close_d1"] > 0) | df.get("renko_cross_up", False)
        else:
            cond_renko_trigger = True

        if "range_4r_rg_close" in df.columns:
            cond_range_buy = (df["range_4r_rg_close"].isna()) | (
                df["range_4r_rg_close"] <= params["buy_range_rg_max"]
            ) | (df.get("range_4r_rg_close_d1", 0) > 0)
        else:
            cond_range_buy = True

        if "WT1 — 快线信号" in df.columns:
            cond_wt_buy = (df["WT1 — 快线信号"].isna()) | (df["WT1 — 快线信号"] <= params["buy_wt1_max"])
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

        cond_renko_top = (df["renko_009_rg_close"] >= params["sell_renko_rg_min"])
        if params.get("sell_renko_turn_dn", True):
            cond_renko_top &= ((df["renko_009_rg_close_d1"] < 0) | df.get("renko_cross_dn", False))

        if "range_4r_rg_close" in df.columns:
            cond_range_top = (
                df["range_4r_rg_close"].notna()
                & (df["range_4r_rg_close"] >= params["sell_range_rg_min"])
                & (df.get("range_4r_rg_close_d1", 0) < 0)
            )
        else:
            cond_range_top = False

        cond_death_cross_top = df.get("renko_cross_dn", False) & (df["renko_009_rg_close"] >= 65.0)

        df["raw_sell"] = cond_renko_top | cond_range_top | cond_death_cross_top

        return df

    def run_backtest(self, params: dict) -> Tuple[pd.DataFrame, dict]:
        df_sig = self.evaluate_signals(params)
        stop_loss_pct = params.get("stop_loss_pct", 1.2)

        trades = []
        position = 0
        entry_price = 0.0
        entry_time = None
        entry_idx = 0
        peak_price = 0.0
        trough_price = 1e9

        for i in range(len(df_sig)):
            row = df_sig.iloc[i]

            if position == 0:
                if row["raw_buy"]:
                    position = 1
                    entry_price = row["close"]
                    entry_time = row["dt"]
                    entry_idx = i
                    peak_price = row["high"]
                    trough_price = row["low"]
            elif position == 1:
                if row["high"] > peak_price:
                    peak_price = row["high"]
                if row["low"] < trough_price:
                    trough_price = row["low"]

                current_pnl = (row["close"] - entry_price) / entry_price * 100
                is_stop_loss = current_pnl <= -stop_loss_pct

                if row["raw_sell"] or is_stop_loss:
                    exit_price = row["close"]
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    mfe_pct = (peak_price - entry_price) / entry_price * 100
                    mae_pct = (entry_price - trough_price) / entry_price * 100

                    trades.append({
                        "entry_time": str(entry_time),
                        "exit_time": str(row["dt"]),
                        "bars_held": i - entry_idx,
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price),
                        "pnl_pct": round(pnl_pct, 2),
                        "mfe_pct": round(mfe_pct, 2),
                        "mae_pct": round(mae_pct, 2),
                        "exit_type": "止损截断" if is_stop_loss else "四图逃顶"
                    })
                    position = 0

        df_trades = pd.DataFrame(trades)

        if not df_trades.empty:
            total_pnl = df_trades["pnl_pct"].sum()
            win_count = (df_trades["pnl_pct"] > 0).sum()
            loss_count = (df_trades["pnl_pct"] <= 0).sum()
            win_rate = (win_count / len(df_trades)) * 100
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


# ==============================================================================
# 后台工作线程（保证 PyQt5 界面平滑不卡顿）
# ==============================================================================
class StrategyWorkerThread(QThread):
    sig_progress = pyqtSignal(int, str)
    sig_log = pyqtSignal(str, str)
    sig_backtest_done = pyqtSignal(dict, pd.DataFrame, pd.DataFrame)
    sig_optimize_done = pyqtSignal(pd.DataFrame)
    sig_error = pyqtSignal(str)

    def __init__(self, mode: str, data_dir: str, params: dict):
        super().__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.params = params

    def run(self):
        try:
            self.sig_progress.emit(10, "正在检索数据文件...")
            manager = MultiChartDataManager(self.data_dir)
            
            def log_callback(msg, level):
                self.sig_log.emit(msg, level)
            def progress_callback(pct, msg):
                self.sig_progress.emit(pct, msg)

            merged_df = manager.load_and_preprocess(log_fn=log_callback, progress_fn=progress_callback)
            algorithm = MultiChartAlgorithm(merged_df)

            if self.mode == "backtest":
                self.sig_progress.emit(70, "正在执行四图共振信号计算与状态机回测...")
                self.sig_log.emit("开始四图共振评估与回测模拟...", "INFO")
                
                df_trades, summary = algorithm.run_backtest(self.params)
                df_sig = algorithm.evaluate_signals(self.params)

                self.sig_progress.emit(100, "回测计算完成！")
                self.sig_log.emit(f"回测完成：总交易 {summary['total_trades']} 笔，胜率 {summary['win_rate_%']}%，累计收益 {summary['cumulative_pnl_%']}%，盈亏风险比 {summary['profit_risk_ratio']}", "SUCCESS")
                self.sig_backtest_done.emit(summary, df_trades, df_sig)

            elif self.mode == "optimize":
                self.sig_progress.emit(60, "正在初始化超参数网格空间...")
                self.sig_log.emit("开始执行四图共振网格搜索寻优...", "INFO")

                grid_10m = [35.0, 42.0, 50.0]
                grid_4m = [25.0, 30.0, 35.0]
                grid_renko = [12.0, 18.0, 25.0]
                grid_sell_renko = [78.0, 82.0, 88.0]
                grid_sell_range = [75.0, 80.0, 85.0]

                total_combos = len(grid_10m) * len(grid_4m) * len(grid_renko) * len(grid_sell_renko) * len(grid_sell_range)
                done = 0
                candidates = []

                for t10 in grid_10m:
                    for t4 in grid_4m:
                        for trenko in grid_renko:
                            for s_renko in grid_sell_renko:
                                for s_range in grid_sell_range:
                                    done += 1
                                    pct = int(60 + (done / total_combos) * 38)
                                    if done % 15 == 0:
                                        self.sig_progress.emit(pct, f"网格寻优进度: {done}/{total_combos} 组参数...")

                                    p = self.params.copy()
                                    p["buy_10m_rg_max"] = t10
                                    p["buy_4m_rg_max"] = t4
                                    p["buy_renko_rg_max"] = trenko
                                    p["sell_renko_rg_min"] = s_renko
                                    p["sell_range_rg_min"] = s_range

                                    trades, smm = algorithm.run_backtest(p)
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
                    self.sig_log.emit(f"网格扫描完成，筛选出 {len(df_cand)} 组有效组合，最高盈亏风险比为: {df_cand.iloc[0]['profit_risk']}", "SUCCESS")
                else:
                    self.sig_log.emit("网格扫描未筛选出有效交易的参数组合", "WARN")

                self.sig_progress.emit(100, "超参数网格扫描寻优完成！")
                self.sig_optimize_done.emit(df_cand)

        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.sig_log.emit(f"运行发生异常: {str(e)}\n{err}", "ERROR")
            self.sig_error.emit(str(e))


# ==============================================================================
# PyQt5 主窗口界面
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TradingView 四图表庄家资金指标算法共振策略系统 (PyQt5 专业版)")
        self.resize(1380, 880)

        self.worker: Optional[StrategyWorkerThread] = None
        self.current_summary: dict = {}
        self.current_trades: pd.DataFrame = pd.DataFrame()
        self.current_signals: pd.DataFrame = pd.DataFrame()
        self.current_optimization: pd.DataFrame = pd.DataFrame()

        self.init_ui()
        self.apply_dark_style()
        self.load_default_params_to_ui()
        self.append_log("系统初始化就绪。请选择数据目录并点击【启动四图共振回测】。", "INFO")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # 1. 顶部数据源选择栏
        top_bar = QGroupBox("数据源设置与全局操作")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(8, 8, 8, 8)

        top_layout.addWidget(QLabel("数据目录:"))
        self.edit_data_dir = QLabel(os.path.abspath("."))
        self.edit_data_dir.setStyleSheet("background: #1e222d; padding: 5px 10px; border-radius: 4px; color: #5bc0be; font-weight: bold;")
        top_layout.addWidget(self.edit_data_dir, 1)

        btn_browse = QPushButton("📁 浏览目录")
        btn_browse.clicked.connect(self.browse_dir)
        top_layout.addWidget(btn_browse)

        btn_auto_detect = QPushButton("🔍 自动检测四图")
        btn_auto_detect.clicked.connect(self.detect_files)
        top_layout.addWidget(btn_auto_detect)

        self.btn_run_backtest = QPushButton("▶ 启动四图共振回测")
        self.btn_run_backtest.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_run_backtest.clicked.connect(self.start_backtest)
        top_layout.addWidget(self.btn_run_backtest)

        self.btn_run_optimize = QPushButton("⚡ 超参数网格寻优")
        self.btn_run_optimize.setStyleSheet("background-color: #f57c00; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_run_optimize.clicked.connect(self.start_optimization)
        top_layout.addWidget(self.btn_run_optimize)

        self.btn_export_csv = QPushButton("💾 导出信号CSV")
        self.btn_export_csv.clicked.connect(self.export_signals_csv)
        top_layout.addWidget(self.btn_export_csv)

        self.btn_export_trades = QPushButton("📊 导出交易明细")
        self.btn_export_trades.clicked.connect(self.export_trades_csv)
        top_layout.addWidget(self.btn_export_trades)

        root_layout.addWidget(top_bar)

        # 2. 中间左右分割主面板
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, 1)

        # 左侧：参数设定控制台
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # 参数面板 Box
        param_group = QGroupBox("四图表关键阈值参数设定 (Parameter Settings)")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(10)

        # 10m
        grp_10m = QGroupBox("1. 10分钟普通K线图 (宏观蓄势过滤)")
        g10_lay = QGridLayout(grp_10m)
        lbl_10m = QLabel("宏观吸筹门槛 (必须跌破 <=):")
        lbl_10m.setToolTip("指标范围 0~100。数值越小代表大周期越超跌。设为 42 表示 10m 必须处于 42 以下的吸筹蓄势区才允许做多。")
        g10_lay.addWidget(lbl_10m, 0, 0)
        self.spin_buy_10m = QDoubleSpinBox()
        self.spin_buy_10m.setRange(0, 100)
        self.spin_buy_10m.setSingleStep(1.0)
        g10_lay.addWidget(self.spin_buy_10m, 0, 1)
        tip_10m = QLabel("💡 说明：大周期必须处于中低位(<=42)，过滤高位诱多")
        tip_10m.setStyleSheet("color: #787b86; font-size: 11px;")
        g10_lay.addWidget(tip_10m, 1, 0, 1, 2)
        param_layout.addWidget(grp_10m)

        # 4m
        grp_4m = QGroupBox("2. 4分钟 0.1688% 砖形图 (中周期 / 对应 4_aa2e3.csv, 每砖0.05)")
        g4_lay = QGridLayout(grp_4m)
        lbl_4m = QLabel("4分钟波段超跌门槛 (必须 <=):")
        lbl_4m.setToolTip("对应 4分钟砖形图(4_aa2e3.csv)。历史大底均在 16~27 之间，设为 30 表示中周期必须跌到位。")
        g4_lay.addWidget(lbl_4m, 0, 0)
        self.spin_buy_4m = QDoubleSpinBox()
        self.spin_buy_4m.setRange(0, 100)
        g4_lay.addWidget(self.spin_buy_4m, 0, 1)
        g4_lay.addWidget(QLabel("4分钟 WT1超卖门槛 (必须 <=):"), 1, 0)
        self.spin_buy_wt1 = QDoubleSpinBox()
        self.spin_buy_wt1.setRange(-100, 100)
        g4_lay.addWidget(self.spin_buy_wt1, 1, 1)
        param_layout.addWidget(grp_4m)

        # 0.09% Renko
        grp_renko = QGroupBox("3. 0.09% 微观砖形图 (极速扳机 / 对应 2_5346c.csv, 每砖0.02)")
        gr_lay = QGridLayout(grp_renko)
        lbl_renko_buy = QLabel("0.09%微观超跌门槛 (必须 <=):")
        lbl_renko_buy.setToolTip("对应 2_5346c.csv (2分钟/0.09%微细砖)。极度灵敏，必须砸到 20 以下（甚至极限 1~10）才算跌透。")
        gr_lay.addWidget(lbl_renko_buy, 0, 0)
        self.spin_buy_renko = QDoubleSpinBox()
        self.spin_buy_renko.setRange(0, 100)
        gr_lay.addWidget(self.spin_buy_renko, 0, 1)
        self.chk_renko_turn_up = QCheckBox("要求跌透后第一拐向上 (启动前精准开枪买入)")
        self.chk_renko_turn_up.setChecked(True)
        gr_lay.addWidget(self.chk_renko_turn_up, 1, 0, 1, 2)
        
        lbl_renko_sell = QLabel("0.09%微观逃顶门槛 (必须 >=):")
        lbl_renko_sell.setToolTip("冲高至 80 以上（逼近天花板 96~101），一旦拐头向下即逃顶。")
        gr_lay.addWidget(lbl_renko_sell, 2, 0)
        self.spin_sell_renko = QDoubleSpinBox()
        self.spin_sell_renko.setRange(0, 100)
        gr_lay.addWidget(self.spin_sell_renko, 2, 1)
        self.chk_renko_turn_dn = QCheckBox("要求冲顶后第一拐向下 (最高点还没跌时卖)")
        self.chk_renko_turn_dn.setChecked(True)
        gr_lay.addWidget(self.chk_renko_turn_dn, 3, 0, 1, 2)
        param_layout.addWidget(grp_renko)

        # Range 4R
        grp_range = QGroupBox("4. Range 4R图 (纯价格结构反转确认)")
        g4r_lay = QGridLayout(grp_range)
        g4r_lay.addWidget(QLabel("抄底超跌门槛 (必须跌破 <=):"), 0, 0)
        self.spin_buy_range = QDoubleSpinBox()
        self.spin_buy_range.setRange(0, 100)
        g4r_lay.addWidget(self.spin_buy_range, 0, 1)
        g4r_lay.addWidget(QLabel("逃顶超买门槛 (必须涨破 >=):"), 1, 0)
        self.spin_sell_range = QDoubleSpinBox()
        self.spin_sell_range.setRange(0, 100)
        g4r_lay.addWidget(self.spin_sell_range, 1, 1)
        param_layout.addWidget(grp_range)

        # 风控止损
        grp_risk = QGroupBox("5. 风险控制设置 (Risk Control)")
        grisk_lay = QGridLayout(grp_risk)
        grisk_lay.addWidget(QLabel("最大硬止损比例 (%):"), 0, 0)
        self.spin_stop_loss = QDoubleSpinBox()
        self.spin_stop_loss.setRange(0.1, 10.0)
        self.spin_stop_loss.setSingleStep(0.1)
        grisk_lay.addWidget(self.spin_stop_loss, 0, 1)
        param_layout.addWidget(grp_risk)

        # 预设按钮
        btn_box = QHBoxLayout()
        btn_reset = QPushButton("↺ 恢复黄金推荐参数")
        btn_reset.clicked.connect(self.load_default_params_to_ui)
        btn_box.addWidget(btn_reset)
        param_layout.addLayout(btn_box)

        param_layout.addStretch(1)
        left_layout.addWidget(param_group)
        left_panel.setFixedWidth(380)
        splitter.addWidget(left_panel)

        # 右侧：标签页展示区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # Tab 1: 回测概览与交易记录
        tab_backtest = QWidget()
        tb_lay = QVBoxLayout(tab_backtest)
        
        # KPI 概览卡片区
        kpi_frame = QFrame()
        kpi_frame.setStyleSheet("background: #1e222d; border-radius: 6px; padding: 6px;")
        kpi_lay = QGridLayout(kpi_frame)

        self.kpi_pnl = QLabel("0.00%")
        self.kpi_winrate = QLabel("0.0%")
        self.kpi_ratio = QLabel("0.00")
        self.kpi_mae = QLabel("0.00%")
        self.kpi_mfe = QLabel("0.00%")
        self.kpi_trades = QLabel("0")

        self.setup_kpi_widget(kpi_lay, 0, 0, "累计收益率 (Total PnL)", self.kpi_pnl, "#26a69a")
        self.setup_kpi_widget(kpi_lay, 0, 1, "回测胜率 (Win Rate)", self.kpi_winrate, "#42a5f5")
        self.setup_kpi_widget(kpi_lay, 0, 2, "盈亏风险比 (Profit/Risk)", self.kpi_ratio, "#ffa726")
        self.setup_kpi_widget(kpi_lay, 1, 0, "最大不利回撤 (Avg MAE)", self.kpi_mae, "#ef5350")
        self.setup_kpi_widget(kpi_lay, 1, 1, "平均潜在涨幅 (Avg MFE)", self.kpi_mfe, "#66bb6a")
        self.setup_kpi_widget(kpi_lay, 1, 2, "总交易笔数 (Total Trades)", self.kpi_trades, "#ab47bc")

        tb_lay.addWidget(kpi_frame)

        # 交易明细表
        tb_lay.addWidget(QLabel("<b>交易明细记录 (Trade History Log):</b>"))
        self.table_trades = QTableWidget()
        self.table_trades.setColumnCount(8)
        self.table_trades.setHorizontalHeaderLabels([
            "买入时间 (Entry)", "卖出时间 (Exit)", "持仓柱数", "买入价", "卖出价", "盈亏 %", "最大潜在涨幅 %", "离场类型"
        ])
        self.table_trades.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tb_lay.addWidget(self.table_trades)

        self.tabs.addTab(tab_backtest, "📊 绩效概览与交易明细")

        # Tab 2: 四图对齐时序与信号列表
        tab_signals = QWidget()
        ts_lay = QVBoxLayout(tab_signals)
        self.table_signals = QTableWidget()
        self.table_signals.setColumnCount(8)
        self.table_signals.setHorizontalHeaderLabels([
            "时间戳", "收盘价", "10m红绿值", "4m红绿值", "Renko 0.09%红绿", "Range 4R红绿", "抄底买入信号", "逃顶卖出信号"
        ])
        self.table_signals.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        ts_lay.addWidget(self.table_signals)
        self.tabs.addTab(tab_signals, "📈 四图时序与共振信号")

        # Tab 3: 参数网格寻优排行榜
        tab_optimize = QWidget()
        to_lay = QVBoxLayout(tab_optimize)
        to_bar = QHBoxLayout()
        to_bar.addWidget(QLabel("网格寻优结果排序（根据盈亏风险比降序）："))
        btn_apply_param = QPushButton("✔ 将所选行参数应用至控制面板")
        btn_apply_param.clicked.connect(self.apply_selected_optimize_param)
        to_bar.addWidget(btn_apply_param)
        to_lay.addLayout(to_bar)

        self.table_optimize = QTableWidget()
        self.table_optimize.setColumnCount(11)
        self.table_optimize.setHorizontalHeaderLabels([
            "10m上限", "4m上限", "Renko上限", "Renko逃顶", "Range逃顶", "交易次数", "胜率 %", "累计收益 %", "平均MFE %", "平均MAE %", "盈亏风险比"
        ])
        self.table_optimize.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_optimize.setSelectionBehavior(QTableWidget.SelectRows)
        to_lay.addWidget(self.table_optimize)
        self.tabs.addTab(tab_optimize, "⚡ 超参数网格寻优榜")

        # Tab 4: 实时日志终端
        tab_log = QWidget()
        tl_lay = QVBoxLayout(tab_log)
        log_ctrl = QHBoxLayout()
        btn_clear_log = QPushButton("🗑 清空日志")
        btn_clear_log.clicked.connect(self.clear_log)
        log_ctrl.addWidget(btn_clear_log)
        btn_save_log = QPushButton("💾 保存日志至文件")
        btn_save_log.clicked.connect(self.save_log_file)
        log_ctrl.addWidget(btn_save_log)
        log_ctrl.addStretch(1)
        tl_lay.addLayout(log_ctrl)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("background-color: #121418; color: #d1d4dc; font-family: Consolas, monospace; font-size: 13px;")
        tl_lay.addWidget(self.log_edit)
        self.tabs.addTab(tab_log, "📝 运行日志终端 (Log)")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # 3. 底部进度条与状态显示
        bot_bar = QHBoxLayout()
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #90a4ae; font-weight: bold;")
        bot_bar.addWidget(self.status_label, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #37474f;
                border-radius: 4px;
                text-align: center;
                background-color: #1e222d;
                color: #ffffff;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #00bcd4;
                border-radius: 3px;
            }
        """)
        bot_bar.addWidget(self.progress_bar, 2)
        root_layout.addLayout(bot_bar)

    def setup_kpi_widget(self, layout, row, col, title, value_label, color):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(4, 4, 4, 4)
        t = QLabel(title)
        t.setStyleSheet("color: #787b86; font-size: 12px;")
        value_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold;")
        l.addWidget(t)
        l.addWidget(value_label)
        layout.addWidget(w, row, col)

    def apply_dark_style(self):
        qss = """
        QMainWindow, QWidget {
            background-color: #131722;
            color: #d1d4dc;
            font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            border: 1px solid #2a2e39;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 12px;
            font-weight: bold;
            color: #5bc0be;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
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
        QDoubleSpinBox, QSpinBox {
            background-color: #1e222d;
            border: 1px solid #363c4e;
            border-radius: 4px;
            padding: 3px;
            color: #ffffff;
            font-weight: bold;
        }
        QCheckBox {
            color: #b2b5be;
        }
        QTabWidget::pane {
            border: 1px solid #2a2e39;
            background: #131722;
        }
        QTabBar::tab {
            background: #1e222d;
            color: #787b86;
            padding: 8px 16px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
            font-weight: bold;
        }
        QTabBar::tab:selected {
            background: #2a2e39;
            color: #5bc0be;
            border-bottom: 2px solid #5bc0be;
        }
        QTableWidget {
            background-color: #131722;
            border: 1px solid #2a2e39;
            gridline-color: #1e222d;
            color: #d1d4dc;
        }
        QHeaderView::section {
            background-color: #1e222d;
            color: #90a4ae;
            padding: 5px;
            border: 1px solid #2a2e39;
            font-weight: bold;
        }
        """
        self.setStyleSheet(qss)

    def load_default_params_to_ui(self):
        defaults = MultiChartAlgorithm.get_default_parameters()
        self.spin_buy_10m.setValue(defaults["buy_10m_rg_max"])
        self.spin_buy_4m.setValue(defaults["buy_4m_rg_max"])
        self.spin_buy_wt1.setValue(defaults["buy_wt1_max"])
        self.spin_buy_renko.setValue(defaults["buy_renko_rg_max"])
        self.chk_renko_turn_up.setChecked(defaults["buy_renko_turn_up"])
        self.spin_sell_renko.setValue(defaults["sell_renko_rg_min"])
        self.chk_renko_turn_dn.setChecked(defaults["sell_renko_turn_dn"])
        self.spin_buy_range.setValue(defaults["buy_range_rg_max"])
        self.spin_sell_range.setValue(defaults["sell_range_rg_min"])
        self.spin_stop_loss.setValue(defaults["stop_loss_pct"])
        self.append_log("已恢复至黄金推荐参数预设。", "INFO")

    def get_ui_params(self) -> dict:
        return {
            "buy_10m_rg_max": self.spin_buy_10m.value(),
            "buy_4m_rg_max": self.spin_buy_4m.value(),
            "buy_wt1_max": self.spin_buy_wt1.value(),
            "buy_renko_rg_max": self.spin_buy_renko.value(),
            "buy_renko_turn_up": self.chk_renko_turn_up.isChecked(),
            "sell_renko_rg_min": self.spin_sell_renko.value(),
            "sell_renko_turn_dn": self.chk_renko_turn_dn.isChecked(),
            "buy_range_rg_max": self.spin_buy_range.value(),
            "sell_range_rg_min": self.spin_sell_range.value(),
            "stop_loss_pct": self.spin_stop_loss.value(),
        }

    def append_log(self, text: str, level: str = "INFO"):
        time_str = datetime.now().strftime("%H:%M:%S")
        color_map = {
            "INFO": "#90caf9",
            "SUCCESS": "#66bb6a",
            "WARN": "#ffa726",
            "ERROR": "#ef5350",
        }
        color = color_map.get(level, "#d1d4dc")
        html = f"<span style='color:#787b86;'>[{time_str}]</span> <span style='color:{color}; font-weight:bold;'>[{level}]</span> {text}"
        self.log_edit.appendHtml(html)
        self.log_edit.moveCursor(QTextCursor.End)

    def clear_log(self):
        self.log_edit.clear()

    def save_log_file(self):
        fname, _ = QFileDialog.getSaveFileName(self, "保存日志文件", "strategy_run.log", "Log Files (*.log);;Text Files (*.txt)")
        if fname:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(self.log_edit.toPlainText())
            QMessageBox.information(self, "成功", "日志已成功保存！")

    def browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择TradingView导出数据文件夹", self.edit_data_dir.text())
        if d:
            self.edit_data_dir.setText(d)
            self.detect_files()

    def detect_files(self):
        target_dir = self.edit_data_dir.text()
        manager = MultiChartDataManager(target_dir)
        fmap = manager.auto_detect_files()
        if not fmap:
            self.append_log(f"在目录 {target_dir} 中未找到相关 CSV 文件！", "WARN")
            QMessageBox.warning(self, "未找到文件", "未能自动识别出 CSV 图表数据，请确认目录下存在 TradingView 导出文件。")
        else:
            self.append_log(f"成功识别出 {len(fmap)} 个图表数据文件：", "SUCCESS")
            for k, v in fmap.items():
                self.append_log(f"  [{k}] -> {os.path.basename(v)}", "INFO")
            QMessageBox.information(self, "识别成功", f"成功匹配到 {len(fmap)} 个图表数据！\n" + "\n".join([f"{k}: {os.path.basename(v)}" for k, v in fmap.items()]))

    def start_backtest(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "当前已有任务正在运行，请稍候！")
            return

        params = self.get_ui_params()
        self.btn_run_backtest.setEnabled(False)
        self.btn_run_optimize.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在执行回测任务...")

        self.worker = StrategyWorkerThread("backtest", self.edit_data_dir.text(), params)
        self.worker.sig_progress.connect(self.on_progress)
        self.worker.sig_log.connect(self.append_log)
        self.worker.sig_backtest_done.connect(self.on_backtest_completed)
        self.worker.sig_error.connect(self.on_task_error)
        self.worker.start()

    def start_optimization(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "当前已有任务正在运行，请稍候！")
            return

        params = self.get_ui_params()
        self.btn_run_backtest.setEnabled(False)
        self.btn_run_optimize.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在执行网格扫描寻优...")

        self.worker = StrategyWorkerThread("optimize", self.edit_data_dir.text(), params)
        self.worker.sig_progress.connect(self.on_progress)
        self.worker.sig_log.connect(self.append_log)
        self.worker.sig_optimize_done.connect(self.on_optimize_completed)
        self.worker.sig_error.connect(self.on_task_error)
        self.worker.start()

    def on_progress(self, val: int, msg: str):
        self.progress_bar.setValue(val)
        self.status_label.setText(msg)

    def on_task_error(self, err: str):
        self.btn_run_backtest.setEnabled(True)
        self.btn_run_optimize.setEnabled(True)
        self.status_label.setText(f"发生错误: {err}")
        QMessageBox.critical(self, "错误", f"执行过程发生异常: {err}")

    def on_backtest_completed(self, summary: dict, df_trades: pd.DataFrame, df_sig: pd.DataFrame):
        self.btn_run_backtest.setEnabled(True)
        self.btn_run_optimize.setEnabled(True)
        self.current_summary = summary
        self.current_trades = df_trades
        self.current_signals = df_sig

        # 更新 KPI 卡片
        pnl = summary["cumulative_pnl_%"]
        self.kpi_pnl.setText(f"{pnl:+.2f}%")
        self.kpi_pnl.setStyleSheet(f"color: {'#26a69a' if pnl >= 0 else '#ef5350'}; font-size: 20px; font-weight: bold;")
        self.kpi_winrate.setText(f"{summary['win_rate_%']:.1f}%")
        self.kpi_ratio.setText(f"{summary['profit_risk_ratio']:.2f}")
        self.kpi_mae.setText(f"{summary['avg_mae_%']:.2f}%")
        self.kpi_mfe.setText(f"{summary['avg_mfe_%']:.2f}%")
        self.kpi_trades.setText(f"{summary['total_trades']} ({summary['win_count']}胜/{summary['loss_count']}负)")

        # 填充交易表
        self.table_trades.setRowCount(0)
        for i, row in df_trades.iterrows():
            self.table_trades.insertRow(i)
            self.table_trades.setItem(i, 0, QTableWidgetItem(str(row["entry_time"])))
            self.table_trades.setItem(i, 1, QTableWidgetItem(str(row["exit_time"])))
            self.table_trades.setItem(i, 2, QTableWidgetItem(str(row["bars_held"])))
            self.table_trades.setItem(i, 3, QTableWidgetItem(f"{row['entry_price']:.3f}"))
            self.table_trades.setItem(i, 4, QTableWidgetItem(f"{row['exit_price']:.3f}"))

            pnl_item = QTableWidgetItem(f"{row['pnl_pct']:+.2f}%")
            if row["pnl_pct"] > 0:
                pnl_item.setForeground(QColor("#26a69a"))
            else:
                pnl_item.setForeground(QColor("#ef5350"))
            pnl_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.table_trades.setItem(i, 5, pnl_item)

            self.table_trades.setItem(i, 6, QTableWidgetItem(f"{row['mfe_pct']:.2f}%"))
            self.table_trades.setItem(i, 7, QTableWidgetItem(str(row["exit_type"])))

        # 填充信号时序预览表（展示最近或触发信号的行）
        self.populate_signals_table(df_sig)
        self.tabs.setCurrentIndex(0)
        self.append_log("回测结果与交易记录已加载至界面展示。", "SUCCESS")

    def populate_signals_table(self, df_sig: pd.DataFrame):
        self.table_signals.setRowCount(0)
        # 筛选有买入或卖出标记的行，以及部分行展示
        highlight_rows = df_sig[df_sig["raw_buy"] | df_sig["raw_sell"]].copy()
        if len(highlight_rows) == 0:
            highlight_rows = df_sig.tail(50)

        for i, (_, row) in enumerate(highlight_rows.iterrows()):
            self.table_signals.insertRow(i)
            self.table_signals.setItem(i, 0, QTableWidgetItem(str(row["dt"])))
            self.table_signals.setItem(i, 1, QTableWidgetItem(f"{row['close']:.2f}"))
            self.table_signals.setItem(i, 2, QTableWidgetItem(f"{row.get('10m_rg_close', np.nan):.2f}"))
            self.table_signals.setItem(i, 3, QTableWidgetItem(f"{row.get('4m_01688_rg_close', np.nan):.2f}"))
            self.table_signals.setItem(i, 4, QTableWidgetItem(f"{row.get('renko_009_rg_close', np.nan):.2f}"))
            self.table_signals.setItem(i, 5, QTableWidgetItem(f"{row.get('range_4r_rg_close', np.nan):.2f}"))

            buy_item = QTableWidgetItem("★ 抄底买入" if row.get("raw_buy", False) else "")
            if row.get("raw_buy", False):
                buy_item.setForeground(QColor("#00e676"))
                buy_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.table_signals.setItem(i, 6, buy_item)

            sell_item = QTableWidgetItem("▼ 逃顶卖出" if row.get("raw_sell", False) else "")
            if row.get("raw_sell", False):
                sell_item.setForeground(QColor("#ff5252"))
                sell_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.table_signals.setItem(i, 7, sell_item)

    def on_optimize_completed(self, df_cand: pd.DataFrame):
        self.btn_run_backtest.setEnabled(True)
        self.btn_run_optimize.setEnabled(True)
        self.current_optimization = df_cand

        self.table_optimize.setRowCount(0)
        if df_cand.empty:
            QMessageBox.information(self, "寻优完成", "未找到符合条件的参数组合。")
            return

        for i, row in df_cand.iterrows():
            self.table_optimize.insertRow(i)
            self.table_optimize.setItem(i, 0, QTableWidgetItem(f"{row['10m_max']:.1f}"))
            self.table_optimize.setItem(i, 1, QTableWidgetItem(f"{row['4m_max']:.1f}"))
            self.table_optimize.setItem(i, 2, QTableWidgetItem(f"{row['renko_max']:.1f}"))
            self.table_optimize.setItem(i, 3, QTableWidgetItem(f"{row['sell_renko']:.1f}"))
            self.table_optimize.setItem(i, 4, QTableWidgetItem(f"{row['sell_range']:.1f}"))
            self.table_optimize.setItem(i, 5, QTableWidgetItem(str(int(row['trades']))))
            self.table_optimize.setItem(i, 6, QTableWidgetItem(f"{row['win_rate_%']:.1f}%"))

            pnl_item = QTableWidgetItem(f"{row['pnl_%']:+.2f}%")
            if row["pnl_%"] >= 0:
                pnl_item.setForeground(QColor("#26a69a"))
            else:
                pnl_item.setForeground(QColor("#ef5350"))
            self.table_optimize.setItem(i, 7, pnl_item)

            self.table_optimize.setItem(i, 8, QTableWidgetItem(f"{row['avg_mfe_%']:.2f}%"))
            self.table_optimize.setItem(i, 9, QTableWidgetItem(f"{row['avg_mae_%']:.2f}%"))

            ratio_item = QTableWidgetItem(f"{row['profit_risk']:.2f}")
            ratio_item.setForeground(QColor("#ffa726"))
            ratio_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.table_optimize.setItem(i, 10, ratio_item)

        self.tabs.setCurrentIndex(2)
        QMessageBox.information(self, "寻优完成", f"已成功完成网格寻优，共找到 {len(df_cand)} 组有效方案！可在表格中选中一行并点击【将所选行参数应用至控制面板】。")

    def apply_selected_optimize_param(self):
        row = self.table_optimize.currentRow()
        if row < 0:
            QMessageBox.warning(self, "提示", "请先在寻优结果表中选中一行参数！")
            return

        t10 = float(self.table_optimize.item(row, 0).text())
        t4 = float(self.table_optimize.item(row, 1).text())
        trenko = float(self.table_optimize.item(row, 2).text())
        s_renko = float(self.table_optimize.item(row, 3).text())
        s_range = float(self.table_optimize.item(row, 4).text())

        self.spin_buy_10m.setValue(t10)
        self.spin_buy_4m.setValue(t4)
        self.spin_buy_renko.setValue(trenko)
        self.spin_sell_renko.setValue(s_renko)
        self.spin_sell_range.setValue(s_range)

        self.append_log(f"已将寻优排行榜第 {row+1} 行参数应用至面板配置！", "SUCCESS")
        QMessageBox.information(self, "应用成功", "参数已填充至左侧控制面板，您可以直接点击【启动四图共振回测】验证！")

    def export_signals_csv(self):
        if self.current_signals.empty:
            QMessageBox.warning(self, "提示", "尚未执行回测，暂无时序信号数据！请先点击【启动四图共振回测】。")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "导出对齐时序与信号", "aligned_signals.csv", "CSV Files (*.csv)")
        if fname:
            self.current_signals.to_csv(fname, index=False, encoding="utf-8-sig")
            self.append_log(f"已成功导出带买卖信号的时序数据至: {fname}", "SUCCESS")
            QMessageBox.information(self, "成功", f"文件已保存至:\n{fname}")

    def export_trades_csv(self):
        if self.current_trades.empty:
            QMessageBox.warning(self, "提示", "当前回测无交易产生或尚未执行！")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "导出交易记录明细", "trades_log.csv", "CSV Files (*.csv)")
        if fname:
            self.current_trades.to_csv(fname, index=False, encoding="utf-8-sig")
            self.append_log(f"已成功导出交易明细数据至: {fname}", "SUCCESS")
            QMessageBox.information(self, "成功", f"交易记录已保存至:\n{fname}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
