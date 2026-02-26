"""
SMMA (Smoothed Moving Average) 压制回测策略引擎
根据 TradingView "ETH 1m SMMA 压制回测策略" 逻辑改写
"""
import numpy as np
import pandas as pd
from utils import logger
import config


class SMMAStrategy:
    """
    SMMA 压制 + 放量阴线短线做空策略
    
    开仓条件：
    1. SMMA 趋势向下: current_smma <= prev_smma
    2. 价格在 SMMA 下方 % (pctThreshold) 范围内
    3. 当前K线为阴线（close < open）
    4. 阴线实体占比 >= bodyPercent 
    5. 成交量 > avgVol * volMultiplier 且 > volMinAbs

    止损: 信号K线最高价 + stopOffset
    止盈: 固定百分比 或 风险收益比 (RR)
    """

    def __init__(self, **kwargs):
        # 对应TV策略里的过滤参数
        self.period = kwargs.get("period", config.SMMA_PERIOD)
        self.vol_min_abs = kwargs.get("vol_min_abs", config.VOL_MIN_ABS)
        self.vol_multiplier = kwargs.get("vol_multiplier", config.VOL_MULTIPLIER)
        self.body_percent = kwargs.get("body_percent", config.BODY_PERCENT)
        self.pct_threshold = kwargs.get("pct_threshold", config.PCT_THRESHOLD)

        # 对应TV策略里的离场参数
        self.tp_type = kwargs.get("tp_type", config.TP_TYPE)
        self.stop_offset = kwargs.get("stop_offset", config.STOP_OFFSET)
        self.fixed_tp = kwargs.get("fixed_tp", config.FIXED_TP)
        self.rr_ratio = kwargs.get("rr_ratio", config.RR_RATIO)

    @staticmethod
    def calc_smma(closes: np.ndarray, period: int) -> np.ndarray:
        """
        计算 SMMA (Smoothed Moving Average) / RMA
        """
        n = len(closes)
        smma = np.full(n, np.nan)

        if n < period:
            return smma

        smma[period - 1] = np.mean(closes[:period])
        for i in range(period, n):
            smma[i] = (smma[i - 1] * (period - 1) + closes[i]) / period

        return smma

    @staticmethod
    def calc_sma(values: np.ndarray, period: int) -> np.ndarray:
        """计算简单移动平均"""
        ret = np.cumsum(values, dtype=float)
        ret[period:] = ret[period:] - ret[:-period]
        res = ret[period - 1:] / period
        pad = np.full(period - 1, np.nan)
        return np.concatenate((pad, res))

    def check_signal(self, df: pd.DataFrame) -> dict | None:
        """
        检查当前是否满足开仓信号
        """
        if len(df) < self.period + 1 or len(df) < 50 + 1:
            logger.debug(f"K线数量不足，当前数量: {len(df)}")
            return None

        closes = df["close"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        
        # 优先使用 volCcy，如果没有则用 vol
        # 在 OKX 图表中，volCcy 表示标的币种数量（比如 ETH 数量）
        if "volCcy" in df.columns:
            volumes = df["volCcy"].values
        else:
            volumes = df["vol"].values

        smma = self.calc_smma(closes, self.period)
        avg_vol = self.calc_sma(volumes, 50)

        # 取最新一根已确认的K线（倒数第二根，最后一根可能未收盘）
        if "confirm" in df.columns:
            confirmed = df[df["confirm"] == 1]
            if len(confirmed) == 0:
                idx = len(df) - 2
            else:
                idx = confirmed.index[-1]
        else:
            idx = len(df) - 1  # 回测模式下每根K线都是已确认的

        if idx < 1:
            return None

        current_close = closes[idx]
        current_open = opens[idx]
        current_high = highs[idx]
        current_low = lows[idx]
        current_vol = volumes[idx]
        
        current_smma = smma[idx]
        prev_smma = smma[idx - 1]
        current_avg_vol = avg_vol[idx]
        
        current_ts = df["ts"].iloc[idx] if "ts" in df.columns else idx

        if np.isnan(current_smma) or np.isnan(current_avg_vol):
            return None

        # ================= 2. 核心计算逻辑对照 TV =================
        
        # isTrendingDown = smma <= smma[1]
        is_trending_down = current_smma <= prev_smma

        # isHighVol = volume > (avgVol * volMultiplier) and volume > volMinAbs
        is_high_vol = (current_vol > (current_avg_vol * self.vol_multiplier)) and (current_vol > self.vol_min_abs)

        # 实体占比 (对付插针)
        candle_range = current_high - current_low
        body_size = abs(current_open - current_close)
        
        is_solid_body = False
        if candle_range > 0:
            if (body_size / candle_range) * 100 >= self.body_percent:
                is_solid_body = True

        # 综合入场条件
        distance_pct = ((current_smma - current_close) / current_smma) * 100
        is_within_range = (current_close < current_smma) and (distance_pct <= self.pct_threshold)

        # shortCondition = (close < open) ...
        is_bearish = current_close < current_open
        
        short_condition = is_bearish and is_high_vol and is_solid_body and is_within_range and is_trending_down

        logger.debug(
            f"信号检测 | 趋势往下:{is_trending_down} | "
            f"大体量阴线:{is_high_vol} (vol:{current_vol:.1f}, 50ma:{current_avg_vol:.1f}) | "
            f"实体占比OK:{is_solid_body} | "
            f"距离均线近:{is_within_range} (距离:{distance_pct:.2f}%) | "
            f"阴线:{is_bearish}"
        )

        if short_condition:
            # ================= 3. 执行逻辑止损止盈 =================
            # 止损价：信号K线最高价 + 偏移量
            stop_price = current_high + self.stop_offset
            
            # 计算止盈价
            risk = stop_price - current_close
            if self.tp_type == "风险收益比 (RR)":
                take_profit_price = current_close - (risk * self.rr_ratio)
            else:
                take_profit_price = current_close * (1 - self.fixed_tp / 100)

            signal = {
                "smma": round(current_smma, 2),
                "price": round(current_close, 2),
                "volume": current_vol,
                "tp_price": round(take_profit_price, 2),
                "sl_price": round(stop_price, 2),
                "ts": current_ts,
                "reason": (
                    f"🔴 做空信号 | SMMA({self.period})={current_smma:.2f} | "
                    f"价格={current_close:.2f} (最高距差:{stop_price - current_high}) | "
                    f"放量={current_vol:.2f} (大于均量{self.vol_multiplier}倍且绝量>{self.vol_min_abs})"
                ),
            }
            logger.info(signal["reason"])
            return signal

        return None
