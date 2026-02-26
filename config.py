"""
配置管理模块 默认已是回测最优配置 新手无需改动 或只改动投入资金 默认50usdt20倍杠杆
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
LEVERAGE = 20                       # 杠杆倍数 
BAR = "1m"                          # K线周期
MGN_MODE = "cross"                  # 保证金模式：全仓

# ═══════════════════════════════════════════
# 策略参数
# ═══════════════════════════════════════════
SMMA_PERIOD = 170                   # SMMA 周期 
VOL_MIN_ABS = 1000                  # 最小绝对量能
VOL_MULTIPLIER = 9                  # 均量倍数 (RV)
BODY_PERCENT = 60.0                 # 阴线实体占比 (%)
PCT_THRESHOLD = 1.0                 # 距离 SMMA 的最大百分比距离 (%)

# 离场参数
TP_TYPE = "固定百分比"                # 止盈模式 ("固定百分比" 或 "风险收益比 (RR)")
STOP_OFFSET = 0.5                   # 止损离最高价偏移 (USDT)
FIXED_TP = 1.2                      # 固定止盈百分比 (%)
RR_RATIO = 2.0                      # 风险收益比 (1:N)
ORDER_AMOUNT_USDT = 50              # 每次开仓投入 USDT

# ═══════════════════════════════════════════
# 运行参数
# ═══════════════════════════════════════════
POLL_INTERVAL = 5                   # 轮询间隔（秒）
CANDLE_FETCH_LIMIT = 500            # 获取K线数量（需足够长以保证均线计算准确）
