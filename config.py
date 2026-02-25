"""
配置管理模块
从 .env 文件和环境变量中加载所有配置参数
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════
# OKX API 凭证
# ═══════════════════════════════════════════
API_KEY = os.getenv("OKX_API_KEY", "")
SECRET_KEY = os.getenv("OKX_SECRET_KEY", "")
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
DEMO = os.getenv("OKX_DEMO", "false").lower() == "true"

# ═══════════════════════════════════════════
# 代理设置
# ═══════════════════════════════════════════
HTTP_PROXY = os.getenv("HTTP_PROXY", "")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")

# ═══════════════════════════════════════════
# 交易参数
# ═══════════════════════════════════════════
INST_ID = "ETH-USDT-SWAP"          # 交易对：ETH永续合约
LEVERAGE = 5                       # 杠杆倍数
BAR = "1m"                          # K线周期
MGN_MODE = "cross"                  # 保证金模式：全仓

# ═══════════════════════════════════════════
# 策略参数
# ═══════════════════════════════════════════
SMMA_PERIOD = 120                   # SMMA 均线周期
ENTRY_RANGE = 0.005                 # 入场范围：SMMA 下方 0.2%
TP_PERCENT = 0.05                   # 止盈：SMMA 下方 2%
SL_PERCENT = 0.001                  # 止损：SMMA 上方 0.2%
VOLUME_THRESHOLD = 12000             # 成交量阈值 (volCcy-币种数量，单位ETH，与OKX图表一致)
ORDER_AMOUNT_USDT = 20              # 每次开仓投入 USDT

# ═══════════════════════════════════════════
# 运行参数
# ═══════════════════════════════════════════
POLL_INTERVAL = 5                   # 轮询间隔（秒）
CANDLE_FETCH_LIMIT = 150            # 获取K线数量（需 > SMMA_PERIOD）
