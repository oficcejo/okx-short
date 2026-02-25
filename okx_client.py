"""
OKX API 封装层
统一封装行情、账户、交易等接口调用
"""
import time
import pandas as pd
from datetime import timedelta, timezone
from okx import MarketData, Account, Trade
from utils import logger
import config

# 北京时区 UTC+8
BEIJING_TZ = timezone(timedelta(hours=8))


class OKXClient:
    """OKX 交易所 API 客户端"""

    def __init__(self):
        flag = "1" if config.DEMO else "0"  # 0=实盘, 1=模拟盘
        proxy = config.HTTPS_PROXY or config.HTTP_PROXY or None
        kwargs = dict(
            api_key=config.API_KEY,
            api_secret_key=config.SECRET_KEY,
            passphrase=config.PASSPHRASE,
            flag=flag,
            debug=False,
        )
        if proxy:
            kwargs["proxy"] = proxy

        # 行情接口不需要鉴权，用公共接口
        self.market = MarketData.MarketAPI(flag=flag, debug=False)
        self.account = Account.AccountAPI(**kwargs)
        self.trade = Trade.TradeAPI(**kwargs)

        logger.info(f"OKX 客户端初始化完成 | 模式: {'模拟盘' if config.DEMO else '实盘'}")

    # ═══════════════════════════════════════════
    # 行情数据
    # ═══════════════════════════════════════════

    def get_candles(self, inst_id: str = None, bar: str = None, limit: int = None) -> pd.DataFrame:
        """
        获取最新 K 线数据
        返回 DataFrame: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
        """
        inst_id = inst_id or config.INST_ID
        bar = bar or config.BAR
        limit = limit or config.CANDLE_FETCH_LIMIT

        result = self.market.get_candlesticks(instId=inst_id, bar=bar, limit=str(limit))
        if result["code"] != "0":
            logger.error(f"获取K线失败: {result['msg']}")
            return pd.DataFrame()

        df = pd.DataFrame(
            result["data"],
            columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"],
        )
        # 转换数据类型
        for col in ["open", "high", "low", "close", "vol", "volCcy", "volCcyQuote"]:
            df[col] = df[col].astype(float)
        # OKX API 返回 UTC 毫秒时间戳，转换为北京时间（UTC+8）
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True).dt.tz_convert('Asia/Shanghai')
        df["confirm"] = df["confirm"].astype(int)
        # 按时间正序排列
        df = df.sort_values("ts").reset_index(drop=True)
        return df

    def get_history_candles(
        self, inst_id: str = None, bar: str = None, after: str = None, before: str = None, limit: int = 100
    ) -> list:
        """
        获取历史 K 线数据（单次请求）
        after/before 为毫秒时间戳字符串
        """
        inst_id = inst_id or config.INST_ID
        bar = bar or config.BAR

        params = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        if after:
            params["after"] = after
        if before:
            params["before"] = before

        result = self.market.get_history_candlesticks(**params)
        if result["code"] != "0":
            logger.error(f"获取历史K线失败: {result['msg']}")
            return []
        return result["data"]

    # ═══════════════════════════════════════════
    # 账户操作
    # ═══════════════════════════════════════════

    def get_balance(self, ccy: str = "USDT") -> float:
        """获取指定币种余额"""
        result = self.account.get_account_balance(ccy=ccy)
        if result["code"] != "0":
            logger.error(f"获取余额失败: {result['msg']}")
            return 0.0
        try:
            details = result["data"][0]["details"]
            for d in details:
                if d["ccy"] == ccy:
                    return float(d["availBal"])
        except (IndexError, KeyError):
            pass
        return 0.0

    def set_leverage(self, inst_id: str = None, lever: int = None, mgn_mode: str = None):
        """设置杠杆倍数"""
        inst_id = inst_id or config.INST_ID
        lever = lever or config.LEVERAGE
        mgn_mode = mgn_mode or config.MGN_MODE

        result = self.account.set_leverage(
            instId=inst_id, lever=str(lever), mgnMode=mgn_mode, posSide="short"
        )
        if result["code"] != "0":
            logger.error(f"设置杠杆失败: {result['msg']}")
            return False
        logger.info(f"杠杆已设置: {inst_id} {lever}x ({mgn_mode})")
        return True

    # ═══════════════════════════════════════════
    # 交易操作
    # ═══════════════════════════════════════════

    def place_market_short(self, inst_id: str = None, sz: str = "1", td_mode: str = None) -> dict | None:
        """
        市价做空开仓
        sz: 合约张数
        td_mode: 保证金模式 (cross/isolated)
        """
        inst_id = inst_id or config.INST_ID
        td_mode = td_mode or config.MGN_MODE

        # 构建下单参数
        # 注意：OKX账户可能设置为单向持仓模式，此时不应传posSide
        params = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": "sell",
            "ordType": "market",
            "sz": sz,
            "tag": "c314b0aecb5bBCDE",
        }
        logger.debug(f"下单参数: {params}")

        result = self.trade.place_order(**params)
        if result["code"] != "0":
            logger.error(f"做空开仓失败: {result['msg']}")
            logger.error(f"错误码: {result.get('code', 'unknown')}")
            logger.error(f"完整响应: {result}")
            return None

        order_id = result["data"][0]["ordId"]
        logger.info(f"✅ 做空开仓成功 | 订单ID: {order_id} | 张数: {sz}")
        return result["data"][0]

    def place_tp_sl(
        self, inst_id: str = None, tp_price: float = 0, sl_price: float = 0, sz: str = "1", td_mode: str = None
    ) -> dict | None:
        """
        设置止盈止损 (TP/SL)
        做空情况下：tp_price < 当前价, sl_price > 当前价
        """
        inst_id = inst_id or config.INST_ID
        td_mode = td_mode or config.MGN_MODE

        result = self.trade.place_algo_order(
            instId=inst_id,
            tdMode=td_mode,
            side="buy",         # 做空的平仓方向是 buy
            ordType="oco",      # OCO = One-Cancels-the-Other (止盈止损)
            sz=sz,
            tpTriggerPx=str(round(tp_price, 2)),
            tpOrdPx="-1",       # -1 表示市价触发
            slTriggerPx=str(round(sl_price, 2)),
            slOrdPx="-1",       # -1 表示市价触发
            tag="c314b0aecb5bBCDE",
        )
        if result["code"] != "0":
            logger.error(f"设置止盈止损失败: {result['msg']}")
            return None

        algo_id = result["data"][0].get("algoId", "unknown")
        logger.info(f"✅ 止盈止损已设置 | TP: {tp_price} | SL: {sl_price} | AlgoID: {algo_id}")
        return result["data"][0]

    def get_positions(self, inst_id: str = None) -> list:
        """获取当前持仓"""
        inst_id = inst_id or config.INST_ID
        result = self.account.get_positions(instId=inst_id)
        if result["code"] != "0":
            logger.error(f"获取持仓失败: {result['msg']}")
            return []
        # 过滤出有仓位的记录
        positions = [p for p in result["data"] if float(p.get("pos", "0")) != 0]
        return positions

    def close_position(self, inst_id: str = None, td_mode: str = None) -> bool:
        """市价平仓"""
        inst_id = inst_id or config.INST_ID
        td_mode = td_mode or config.MGN_MODE
        result = self.trade.close_positions(
            instId=inst_id,
            mgnMode=td_mode,
            tag="c314b0aecb5bBCDE",
        )
        if result["code"] != "0":
            logger.error(f"平仓失败: {result['msg']}")
            return False
        logger.info("✅ 已市价平仓")
        return True

    def calc_contract_size(self, usdt_amount: float, price: float) -> int:
        """
        计算合约张数
        ETH-USDT-SWAP 合约面值 = 0.1 ETH/张 (根据API验证)
        张数 = (USDT金额 * 杠杆) / (价格 * 面值)
        """
        ct_val = 0.1  # ETH合约面值 = 0.1 ETH
        sz = int((usdt_amount * config.LEVERAGE) / (price * ct_val))
        return max(sz, 1)  # 最少1张
