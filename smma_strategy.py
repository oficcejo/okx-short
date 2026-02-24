"""
SMMA (Smoothed Moving Average) 策略引擎
策略：SMMA120 均线压制 + 放量阴线做空
"""
import numpy as np
import pandas as pd
from utils import logger
import config


class SMMAStrategy:
    """
    SMMA120 压制做空策略

    开仓条件：
    1. 价格在 SMMA120 下方 entry_range 范围内
    2. 当前K线为阴线（close < open）
    3. 成交量 > VOLUME_THRESHOLD

    止盈: SMMA120 * (1 - TP_PERCENT)
    止损: SMMA120 * (1 + SL_PERCENT)   → SMMA 上方 SL_PERCENT
    """

    def __init__(
        self,
        period: int = None,
        entry_range: float = None,
        tp_percent: float = None,
        sl_percent: float = None,
        volume_threshold: float = None,
    ):
        self.period = period or config.SMMA_PERIOD
        self.entry_range = entry_range or config.ENTRY_RANGE
        self.tp_percent = tp_percent or config.TP_PERCENT
        self.sl_percent = sl_percent or config.SL_PERCENT
        self.volume_threshold = volume_threshold or config.VOLUME_THRESHOLD

    @staticmethod
    def calc_smma(closes: np.ndarray, period: int) -> np.ndarray:
        """
        计算 SMMA (Smoothed Moving Average)

        公式：
          SMMA(1) = SMA(period)                           — 第一个值等于简单移动平均
          SMMA(i) = (SMMA(i-1) * (period-1) + Close(i)) / period  — 后续递推

        参数:
            closes: 收盘价数组
            period: 周期

        返回:
            smma: 与 closes 等长的 SMMA 数组 (前 period-1 个值为 NaN)
        """
        n = len(closes)
        smma = np.full(n, np.nan)

        if n < period:
            return smma

        # 第一个 SMMA 值 = 前 period 个收盘价的简单移动平均
        smma[period - 1] = np.mean(closes[:period])

        # 递推计算
        for i in range(period, n):
            smma[i] = (smma[i - 1] * (period - 1) + closes[i]) / period

        return smma

    def check_signal(self, df: pd.DataFrame) -> dict | None:
        """
        检查当前是否满足开仓信号

        参数:
            df: K线 DataFrame，需包含 [open, high, low, close, vol] 列，按时间正序排列

        返回:
            信号字典 或 None
            {
                "smma120": float,    # 当前 SMMA120 值
                "price": float,      # 当前收盘价
                "volume": float,     # 当前成交量
                "tp_price": float,   # 止盈价
                "sl_price": float,   # 止损价
                "reason": str,       # 信号描述
            }
        """
        if len(df) < self.period + 1:
            logger.debug(f"K线数量不足: {len(df)} < {self.period + 1}")
            return None

        closes = df["close"].values
        smma = self.calc_smma(closes, self.period)

        # 取最新一根已确认的K线（倒数第二根，最后一根可能未收盘）
        # 如果 df 中有 confirm 列，优先使用最新已确认的K线
        if "confirm" in df.columns:
            confirmed = df[df["confirm"] == 1]
            if len(confirmed) == 0:
                # 没有已确认的K线，取倒数第二根
                idx = len(df) - 2
            else:
                idx = confirmed.index[-1]
        else:
            idx = len(df) - 1  # 回测模式下每根K线都是已确认的

        current_close = closes[idx]
        current_open = df["open"].values[idx]
        current_vol = df["vol"].values[idx]
        current_smma = smma[idx]
        current_ts = df["ts"].iloc[idx] if "ts" in df.columns else idx

        if np.isnan(current_smma):
            logger.debug("SMMA 值尚未就绪")
            return None

        # ─────── 条件判断 ───────

        # 条件 1: 价格在 SMMA 下方 0.5% 范围内
        upper_bound = current_smma
        lower_bound = current_smma * (1 - self.entry_range)
        price_in_range = lower_bound <= current_close <= upper_bound

        # 条件 2: 阴线（收盘价 < 开盘价）
        is_bearish = current_close < current_open

        # 条件 3: 成交量(volCcy-币种数量)大于等于阈值
        # OKX图表显示的成交量是 volCcy(ETH数量)，而非 vol(合约张数)
        current_vol_ccy = df["volCcy"].values[idx] if "volCcy" in df.columns else current_vol
        vol_above = current_vol_ccy >= self.volume_threshold

        logger.debug(
            f"信号检测 | SMMA120: {current_smma:.2f} | "
            f"价格: {current_close:.2f} | "
            f"范围: [{lower_bound:.2f}, {upper_bound:.2f}] | "
            f"价格在范围内: {price_in_range} | "
            f"阴线: {is_bearish} | "
            f"成交量(volCcy): {current_vol_ccy:.2f} >= {self.volume_threshold}: {vol_above}"
        )

        if price_in_range and is_bearish and vol_above:
            tp_price, sl_price = self.calc_tp_sl(current_smma)
            signal = {
                "smma120": round(current_smma, 2),
                "price": round(current_close, 2),
                "volume": current_vol_ccy,
                "tp_price": round(tp_price, 2),
                "sl_price": round(sl_price, 2),
                "ts": current_ts,
                "reason": (
                    f"🔴 做空信号 | SMMA120={current_smma:.2f} | "
                    f"价格={current_close:.2f} ∈ [{lower_bound:.2f}, {upper_bound:.2f}] | "
                    f"阴线(O={current_open:.2f} > C={current_close:.2f}) | "
                    f"放量={current_vol_ccy:.2f} >= {self.volume_threshold}"
                ),
            }
            logger.info(signal["reason"])
            return signal

        return None

    def calc_tp_sl(self, smma_value: float) -> tuple[float, float]:
        """
        计算止盈止损价格

        止盈 = SMMA * (1 - TP_PERCENT)   → 做空盈利方向是下跌
        止损 = SMMA * (1 + SL_PERCENT)   → SMMA 上方 SL_PERCENT 作为止损缓冲区
        """
        tp_price = smma_value * (1 - self.tp_percent)
        sl_price = smma_value * (1 + self.sl_percent)
        return tp_price, sl_price
