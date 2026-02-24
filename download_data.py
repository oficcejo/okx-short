"""
历史 K 线数据下载工具
从 OKX API 批量下载 1 分钟 K 线数据，保存为 CSV 格式
"""
import os
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone
from utils import logger
from okx_client import OKXClient
import config

# 北京时区 UTC+8
BEIJING_TZ = timezone(timedelta(hours=8))


def download_candles(days: int = 7, inst_id: str = None, bar: str = None, output_dir: str = "data"):
    """
    下载历史 K 线数据

    参数:
        days: 下载天数
        inst_id: 交易对
        bar: K线周期
        output_dir: 输出目录
    """
    inst_id = inst_id or config.INST_ID
    bar = bar or config.BAR
    client = OKXClient()

    os.makedirs(output_dir, exist_ok=True)

    # 计算时间范围（使用北京时间）
    end_time = datetime.now(BEIJING_TZ)
    start_time = end_time - timedelta(days=days)

    logger.info(f"开始下载 {inst_id} {bar} K线数据")
    logger.info(f"时间范围: {start_time.strftime('%Y-%m-%d %H:%M')} ~ {end_time.strftime('%Y-%m-%d %H:%M')} 北京时间")

    all_data = []
    after = str(int(end_time.timestamp() * 1000))  # 从最新开始往前翻页
    batch = 0

    while True:
        batch += 1
        data = client.get_history_candles(inst_id=inst_id, bar=bar, after=after, limit=100)

        if not data:
            logger.info("没有更多数据了")
            break

        all_data.extend(data)

        # 获取最早的时间戳作为下一页的 after
        oldest_ts = int(data[-1][0])
        oldest_time = datetime.fromtimestamp(oldest_ts / 1000, BEIJING_TZ)

        logger.info(f"  批次 {batch}: 获取 {len(data)} 条 | 最早: {oldest_time.strftime('%Y-%m-%d %H:%M')} 北京时间")

        # 检查是否已到达开始时间
        if oldest_time <= start_time:
            logger.info("已到达目标起始时间")
            break

        after = str(oldest_ts)
        time.sleep(0.2)  # API 限速

    if not all_data:
        logger.warning("未获取到任何数据")
        return None

    # 转换为 DataFrame
    df = pd.DataFrame(
        all_data,
        columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"],
    )

    # 数据类型转换
    for col in ["open", "high", "low", "close", "vol", "volCcy", "volCcyQuote"]:
        df[col] = df[col].astype(float)
    # OKX API 返回 UTC 毫秒时间戳，转换为北京时间（UTC+8）
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True).dt.tz_convert('Asia/Shanghai')
    df["confirm"] = df["confirm"].astype(int)

    # 去重 + 排序
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # 过滤时间范围
    df = df[df["ts"] >= pd.Timestamp(start_time)].reset_index(drop=True)

    # 保存
    filename = f"{inst_id.replace('-', '_')}_{bar}_{days}d.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)

    logger.info(f"✅ 数据已保存: {filepath}")
    logger.info(f"   总记录数: {len(df)}")
    logger.info(f"   时间范围: {df['ts'].iloc[0]} ~ {df['ts'].iloc[-1]}")

    return filepath


def main():
    parser = argparse.ArgumentParser(description="OKX 历史 K 线数据下载工具")
    parser.add_argument("--days", type=int, default=7, help="下载天数 (默认: 7)")
    parser.add_argument("--inst-id", type=str, default=None, help=f"交易对 (默认: {config.INST_ID})")
    parser.add_argument("--bar", type=str, default=None, help=f"K线周期 (默认: {config.BAR})")
    parser.add_argument("--output", type=str, default="data", help="输出目录 (默认: data)")
    args = parser.parse_args()

    download_candles(
        days=args.days,
        inst_id=args.inst_id,
        bar=args.bar,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
