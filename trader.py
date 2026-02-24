"""
实时交易机器人
主循环监控 1 分钟 K 线，检测 SMMA120 压制 + 放量阴线信号后自动做空
"""
import time
import argparse
from utils import logger
from okx_client import OKXClient
from smma_strategy import SMMAStrategy
import config


class Trader:
    """超短线做空交易机器人"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.client = OKXClient()
        self.strategy = SMMAStrategy()
        self.in_position = False

        if dry_run:
            logger.info("⚠️  DRY-RUN 模式：仅监控信号，不执行交易")

    def setup(self):
        """初始化：设置杠杆、检查账户"""
        logger.info("=" * 60)
        logger.info("OKX ETH 永续合约超短线做空机器人")
        logger.info(f"交易对: {config.INST_ID}")
        logger.info(f"杠杆: {config.LEVERAGE}x")
        logger.info(f"策略: SMMA{config.SMMA_PERIOD} 压制 + 放量阴线做空")
        logger.info(f"入场范围: SMMA 下方 {config.ENTRY_RANGE * 100}%")
        logger.info(f"止盈: SMMA 下方 {config.TP_PERCENT * 100}%")
        logger.info(f"止损: SMMA 上方 {config.SL_PERCENT * 100}%")
        logger.info(f"成交量阈值: {config.VOLUME_THRESHOLD}")
        logger.info(f"投入金额: {config.ORDER_AMOUNT_USDT} USDT")
        logger.info(f"Broker Tag: {config.BROKER_TAG}")
        logger.info("=" * 60)

        if not self.dry_run:
            # 设置杠杆
            self.client.set_leverage()
            # 检查余额
            balance = self.client.get_balance()
            logger.info(f"账户 USDT 余额: {balance:.2f}")
            if balance < config.ORDER_AMOUNT_USDT:
                logger.warning(f"余额不足！需要至少 {config.ORDER_AMOUNT_USDT} USDT")

            # 检查是否已有持仓
            positions = self.client.get_positions()
            if positions:
                self.in_position = True
                logger.info(f"检测到现有持仓: {len(positions)} 个")
                for p in positions:
                    logger.info(
                        f"  持仓方向: {p.get('posSide')} | "
                        f"数量: {p.get('pos')} | "
                        f"未实现盈亏: {p.get('upl', '0')}"
                    )

    def check_and_trade(self):
        """一个检查周期：获取K线 → 判断信号 → 执行交易"""
        # 获取K线数据
        df = self.client.get_candles()
        if df.empty:
            logger.warning("获取K线数据失败，跳过本轮")
            return

        # 检测信号
        signal = self.strategy.check_signal(df)

        if signal is None:
            # 无信号，输出当前状态
            closes = df["close"].values
            smma = SMMAStrategy.calc_smma(closes, config.SMMA_PERIOD)
            latest_smma = smma[-1] if not all(map(lambda x: x != x, smma)) else 0
            latest_price = closes[-1]
            if latest_smma > 0:
                diff_pct = (latest_price - latest_smma) / latest_smma * 100
                logger.info(
                    f"📊 监控中 | 价格: {latest_price:.2f} | "
                    f"SMMA120: {latest_smma:.2f} | "
                    f"偏离: {diff_pct:+.3f}%"
                )
            return

        # 有信号！
        if self.in_position:
            logger.info("⚠️  已有持仓，跳过本次信号")
            return

        if self.dry_run:
            logger.info(f"🔔 [DRY-RUN] 检测到信号但不执行交易")
            logger.info(f"   SMMA120: {signal['smma120']}")
            logger.info(f"   价格: {signal['price']}")
            logger.info(f"   成交量: {signal['volume']:.0f}")
            logger.info(f"   止盈: {signal['tp_price']}")
            logger.info(f"   止损: {signal['sl_price']}")
            return

        # 执行交易
        self._execute_trade(signal)

    def _execute_trade(self, signal: dict):
        """执行做空交易"""
        price = signal["price"]
        tp_price = signal["tp_price"]
        sl_price = signal["sl_price"]

        # 计算合约张数
        sz = self.client.calc_contract_size(config.ORDER_AMOUNT_USDT, price)
        logger.info(f"📐 计算合约张数: {sz} 张 (投入 {config.ORDER_AMOUNT_USDT} USDT, 杠杆 {config.LEVERAGE}x)")

        # 市价做空
        order = self.client.place_market_short(sz=str(sz))
        if order is None:
            logger.error("❌ 开仓失败")
            return

        # 设置止盈止损
        tp_sl = self.client.place_tp_sl(
            tp_price=tp_price,
            sl_price=sl_price,
            sz=str(sz),
        )
        if tp_sl is None:
            logger.warning("⚠️  止盈止损设置失败，请手动设置！")

        self.in_position = True
        logger.info(f"🎯 交易已执行 | 张数: {sz} | TP: {tp_price} | SL: {sl_price}")

    def run(self):
        """主循环"""
        self.setup()
        logger.info(f"🚀 机器人启动，每 {config.POLL_INTERVAL} 秒检查一次...")

        try:
            while True:
                try:
                    # 检查仓位状态
                    if self.in_position and not self.dry_run:
                        positions = self.client.get_positions()
                        if not positions:
                            self.in_position = False
                            logger.info("📭 仓位已平 (止盈/止损触发)")

                    self.check_and_trade()
                except Exception as e:
                    logger.error(f"运行异常: {e}")

                time.sleep(config.POLL_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n⏹️  机器人已手动停止")


def main():
    parser = argparse.ArgumentParser(description="OKX ETH 永续合约超短线做空机器人")
    parser.add_argument("--dry-run", action="store_true", help="试运行模式：仅监控信号，不执行交易")
    args = parser.parse_args()

    trader = Trader(dry_run=args.dry_run)
    trader.run()


if __name__ == "__main__":
    main()
