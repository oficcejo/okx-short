"""
回测系统
加载历史 K 线 CSV 数据，模拟 SMMA 压制 + 放量阴线做空策略的执行
"""
import os
import argparse
import pandas as pd
import numpy as np
from tabulate import tabulate
from utils import logger
from smma_strategy import SMMAStrategy
import config
import time

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class Backtester:
    """回测引擎"""

    def __init__(
        self,
        data_path: str,
        initial_capital: float = 1000.0,
        order_amount: float = None,
        leverage: int = None,
        fee_rate: float = 0.0005,
        strategy_params: dict = None,
    ):
        """
        参数:
            data_path: CSV 数据文件路径
            initial_capital: 初始资金 (USDT)
            order_amount: 每次投入金额 (USDT)
            leverage: 杠杆倍数
            fee_rate: 手续费率 (taker 0.05%)
            strategy_params: 策略参数字典
        """
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.order_amount = order_amount or config.ORDER_AMOUNT_USDT
        self.leverage = leverage or config.LEVERAGE
        self.fee_rate = fee_rate
        self.strategy = SMMAStrategy(**(strategy_params or {}))
        self.trades = []  # 交易记录
        self.equity_curve = []  # 权益曲线

    def load_data(self) -> pd.DataFrame:
        """加载 CSV 数据"""
        df = pd.read_csv(self.data_path)

        # 确保列名正确
        required_cols = ["ts", "open", "high", "low", "close", "vol"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV 缺少必要列: {col}")

        # 加载时间，如果带时区信息则保留，否则假设为北京时间
        df["ts"] = pd.to_datetime(df["ts"])
        if df["ts"].dt.tz is None:
            from datetime import timedelta, timezone
            beijing_tz = timezone(timedelta(hours=8))
            df["ts"] = df["ts"].dt.tz_localize(beijing_tz)
        for col in ["open", "high", "low", "close", "vol"]:
            df[col] = df[col].astype(float)

        df = df.sort_values("ts").reset_index(drop=True)
        # logger.info(f"数据加载完成: {len(df)} 条 K 线 | {df['ts'].iloc[0]} ~ {df['ts'].iloc[-1]}")
        return df

    def run(self, silent=False):
        """执行回测"""
        # loguru 不支持 setLevel，我们依赖代码中的 if not silent 判断
        # 或者可以使用 logger.remove() 但这会影响全局。
        # 鉴于我们大部分日志都有 silent 判断，这里只需保持原样即可。

        df = self.load_data()
        closes = df["close"].values
        opens = df["open"].values
        # 使用 volCcy(币种成交量) 而非 vol(合约张数)，与OKX图表保持一致
        vols = df["volCcy"].values if "volCcy" in df.columns else df["vol"].values
        highs = df["high"].values
        lows = df["low"].values

        # 预计算完整 SMMA 和均量线
        smma = SMMAStrategy.calc_smma(closes, self.strategy.period)
        avg_vol = SMMAStrategy.calc_sma(vols, 50)

        in_position = False
        entry_price = 0
        entry_time = None
        tp_price = 0
        sl_price = 0
        position_sz = 0  # 合约张数
        ct_val = 0.01  # ETH 合约面值

        if not silent:
            logger.info("=" * 60)
            logger.info("开始回测")
            logger.info(f"初始资金: {self.initial_capital} USDT")
            logger.info(f"每次投入: {self.order_amount} USDT | 杠杆: {self.leverage}x")
            logger.info(f"手续费率: {self.fee_rate * 100}%")
            logger.info("=" * 60)

        for i in range(self.strategy.period, len(df)):
            current_smma = smma[i]
            if np.isnan(current_smma):
                continue

            current_close = closes[i]
            current_high = highs[i]
            current_low = lows[i]
            current_time = df["ts"].iloc[i]

            # 记录权益
            if in_position:
                unrealized_pnl = (entry_price - current_close) * position_sz * ct_val
                current_equity = self.capital + unrealized_pnl
            else:
                current_equity = self.capital
            self.equity_curve.append({"ts": current_time, "equity": current_equity})

            if in_position:
                exit_price = None
                exit_reason = ""

                if current_low <= tp_price:
                    exit_price = tp_price
                    exit_reason = "止盈"
                elif current_high >= sl_price:
                    exit_price = sl_price
                    exit_reason = "止损"

                if exit_price is not None:
                    pnl_per_contract = (entry_price - exit_price) * ct_val
                    gross_pnl = pnl_per_contract * position_sz
                    fee = abs(exit_price * position_sz * ct_val) * self.fee_rate
                    net_pnl = gross_pnl - fee
                    self.capital += net_pnl

                    trade_record = {
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(exit_price, 2),
                        "smma_val": round(smma[i], 2),
                        "tp_price": round(tp_price, 2),
                        "sl_price": round(sl_price, 2),
                        "sz": position_sz,
                        "gross_pnl": round(gross_pnl, 4),
                        "fee": round(fee, 4),
                        "net_pnl": round(net_pnl, 4),
                        "capital": round(self.capital, 4),
                        "reason": exit_reason,
                    }
                    self.trades.append(trade_record)
                    if not silent:
                        emoji = "✅" if net_pnl > 0 else "❌"
                        logger.info(
                            f"{emoji} {exit_reason} | 入场: {entry_price:.2f} → 出场: {exit_price:.2f} | "
                            f"盈亏: {net_pnl:+.4f} USDT | 资金: {self.capital:.2f}"
                        )
                    in_position = False
                    continue

            else:
                is_trending_down = (i > 0 and current_smma <= smma[i - 1])
                is_high_vol = (vols[i] > (avg_vol[i] * self.strategy.vol_multiplier)) and (vols[i] > self.strategy.vol_min_abs)
                
                candle_range = current_high - current_low
                body_size = abs(opens[i] - current_close)
                is_solid_body = False
                if candle_range > 0 and (body_size / candle_range) * 100 >= self.strategy.body_percent:
                    is_solid_body = True
                
                distance_pct = ((current_smma - current_close) / current_smma) * 100
                is_within_range = (current_close < current_smma) and (distance_pct <= self.strategy.pct_threshold)
                is_bearish = current_close < opens[i]
                
                short_condition = is_bearish and is_high_vol and is_solid_body and is_within_range and is_trending_down

                if short_condition:
                    if self.capital < self.order_amount:
                        if not silent:
                            logger.warning(f"资金不足，跳过信号: {self.capital:.2f} < {self.order_amount}")
                        continue

                    entry_price = current_close
                    entry_time = current_time
                    
                    sl_price = current_high + self.strategy.stop_offset
                    risk = sl_price - current_close
                    if self.strategy.tp_type == "风险收益比 (RR)":
                        tp_price = current_close - (risk * self.strategy.rr_ratio)
                    else:
                        tp_price = current_close * (1 - self.strategy.fixed_tp / 100)

                    position_sz = int((self.order_amount * self.leverage) / (entry_price * ct_val))
                    position_sz = max(position_sz, 1)

                    open_fee = abs(entry_price * position_sz * ct_val) * self.fee_rate
                    self.capital -= open_fee

                    in_position = True
                    if not silent:
                        logger.info(
                            f"🔴 做空开仓 | {current_time} | 价格: {entry_price:.2f} | "
                            f"SMMA: {current_smma:.2f} | 成交量: {vols[i]:.2f} | "
                            f"张数: {position_sz}"
                        )

        if in_position:
            exit_price = closes[-1]
            pnl_per_contract = (entry_price - exit_price) * ct_val
            gross_pnl = pnl_per_contract * position_sz
            fee = abs(exit_price * position_sz * ct_val) * self.fee_rate
            net_pnl = gross_pnl - fee
            self.capital += net_pnl
            self.trades.append({
                "entry_time": entry_time,
                "exit_time": df["ts"].iloc[-1],
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "smma_val": round(smma[-1], 2),
                "tp_price": round(tp_price, 2),
                "sl_price": round(sl_price, 2),
                "sz": position_sz,
                "gross_pnl": round(gross_pnl, 4),
                "fee": round(fee, 4),
                "net_pnl": round(net_pnl, 4),
                "capital": round(self.capital, 4),
                "reason": "回测结束平仓",
            })

        if silent:
            return self.get_summary()
        else:
            self.print_report()

    def get_summary(self) -> dict:
        """获取回测简要统计"""
        if not self.trades:
            return {"roi": -100.0, "win_rate": 0.0, "total_trades": 0, "profit_factor": 0.0, "max_drawdown": 0.0}

        trades_df = pd.DataFrame(self.trades)
        total_trades = len(trades_df)
        wins = trades_df[trades_df["net_pnl"] > 0]
        losses = trades_df[trades_df["net_pnl"] <= 0]
        win_rate = len(wins) / total_trades * 100
        roi = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        loss_sum = abs(losses["net_pnl"].sum())
        profit_factor = wins["net_pnl"].sum() / loss_sum if loss_sum > 0 else float("inf")

        equity_df = pd.DataFrame(self.equity_curve)
        peak = equity_df["equity"].expanding().max()
        drawdown = (equity_df["equity"] - peak) / peak * 100
        max_drawdown = drawdown.min()

        return {
            "roi": round(roi, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": total_trades,
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_drawdown, 2),
            "final_capital": round(self.capital, 2)
        }

    def print_report(self):
        """打印回测报告"""
        print("\n" + "=" * 70)
        print("回测报告")
        print("=" * 70)

        if not self.trades:
            print("⚠️  回测期间没有产生任何交易信号")
            return

        trades_df = pd.DataFrame(self.trades)

        # ─── 基本统计 ───
        total_trades = len(trades_df)
        wins = trades_df[trades_df["net_pnl"] > 0]
        losses = trades_df[trades_df["net_pnl"] <= 0]
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

        total_pnl = trades_df["net_pnl"].sum()
        total_fees = trades_df["fee"].sum()
        final_capital = self.capital
        roi = (final_capital - self.initial_capital) / self.initial_capital * 100

        avg_win = wins["net_pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["net_pnl"].mean() if len(losses) > 0 else 0
        p_factor = abs(wins["net_pnl"].sum() / losses["net_pnl"].sum()) if len(losses) > 0 and losses["net_pnl"].sum() != 0 else float("inf")

        # ─── 最大回撤 ───
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            peak = equity_df["equity"].expanding().max()
            drawdown = (equity_df["equity"] - peak) / peak * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        # 持仓时间统计
        if "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
            trades_df["duration"] = pd.to_datetime(trades_df["exit_time"]) - pd.to_datetime(trades_df["entry_time"])
            avg_duration = trades_df["duration"].mean()
        else:
            avg_duration = "N/A"

        # ─── 打印统计表格 ───
        stats = [
            ["初始资金", f"{self.initial_capital:.2f} USDT"],
            ["最终资金", f"{final_capital:.2f} USDT"],
            ["总收益", f"{total_pnl:+.4f} USDT"],
            ["收益率", f"{roi:+.2f}%"],
            ["总手续费", f"{total_fees:.4f} USDT"],
            ["─" * 20, "─" * 20],
            ["总交易次数", total_trades],
            ["盈利次数", len(wins)],
            ["亏损次数", len(losses)],
            ["胜率", f"{win_rate:.1f}%"],
            ["─" * 20, "─" * 20],
            ["平均盈利", f"{avg_win:+.4f} USDT"],
            ["平均亏损", f"{avg_loss:+.4f} USDT"],
            ["盈亏比", f"{p_factor:.2f}"],
            ["最大回撤", f"{max_drawdown:.2f}%"],
            ["平均持仓时间", str(avg_duration)],
        ]
        print(tabulate(stats, headers=["指标", "值"], tablefmt="simple_outline"))

        # ─── 打印交易明细 ───
        print(f"\n交易明细 (共 {total_trades} 笔)")
        print("-" * 70)
        detail_cols = ["entry_time", "exit_time", "entry_price", "exit_price", "net_pnl", "reason"]
        detail_df = trades_df[detail_cols].copy()
        detail_df.columns = ["开仓时间", "平仓时间", "开仓价", "平仓价", "净盈亏", "原因"]
        print(tabulate(detail_df, headers="keys", tablefmt="simple_outline", showindex=True, floatfmt=".4f"))

        # ─── 生成图表 ───
        if HAS_MATPLOTLIB:
            self._plot_results(trades_df)

    def _plot_results(self, trades_df: pd.DataFrame):
        """生成回测结果图表"""
        output_dir = os.path.dirname(self.data_path) or "data"
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle("SMMA 压制 + 放量阴线做空 回测结果", fontsize=14, fontweight="bold")

        # 权益曲线
        eq_df = pd.DataFrame(self.equity_curve)
        if not eq_df.empty:
            axes[0].plot(eq_df["ts"], eq_df["equity"], color="#2196F3", linewidth=1)
            axes[0].axhline(y=self.initial_capital, color="gray", linestyle="--", alpha=0.5)
            axes[0].set_ylabel("权益 (USDT)")
            axes[0].set_title("权益曲线")
            axes[0].grid(alpha=0.3)

        # 每笔交易盈亏
        colors = ["#4CAF50" if pnl > 0 else "#F44336" for pnl in trades_df["net_pnl"]]
        axes[1].bar(range(len(trades_df)), trades_df["net_pnl"], color=colors)
        axes[1].axhline(y=0, color="gray", linewidth=0.5)
        axes[1].set_ylabel("盈亏 (USDT)")
        axes[1].set_xlabel("交易序号")
        axes[1].set_title("逐笔盈亏")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(output_dir, f"backtest_result_{int(time.time())}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        # logger.info(f"📈 回测图表已保存: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="SMMA 压制做空策略回测")
    parser.add_argument("--data", type=str, required=True, help="历史K线 CSV 文件路径")
    parser.add_argument("--capital", type=float, default=1000.0, help="初始资金 (USDT, 默认: 1000)")
    parser.add_argument("--amount", type=float, default=None, help=f"每次投入金额 (默认: {config.ORDER_AMOUNT_USDT})")
    parser.add_argument("--leverage", type=int, default=None, help=f"杠杆倍数 (默认: {config.LEVERAGE})")
    args = parser.parse_args()

    backtester = Backtester(
        data_path=args.data,
        initial_capital=args.capital,
        order_amount=args.amount,
        leverage=args.leverage,
    )
    backtester.run()


if __name__ == "__main__":
    main()
import time
