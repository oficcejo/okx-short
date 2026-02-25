import time
from utils import logger
import config
from okx_client import OKXClient

def test_live_trade():
    logger.info("="*60)
    logger.info("开始实盘开仓平仓测试 (20 USDT 本金, 5x 杠杆)")
    logger.info("="*60)
    
    # 临时覆盖 config 中的杠杆为 5 倍，确保 calc_contract_size 计算正确
    config.LEVERAGE = 5
    
    client = OKXClient()
    inst_id = "ETH-USDT-SWAP"
    
    # 1. 检查账户余额
    balance = client.get_balance("USDT")
    logger.info(f"账户当前 USDT 余额: {balance:.2f}")
    if balance < 20:
        logger.error("余额不足 20 USDT，为确保测试安全，终止操作。")
        return
        
    # 2. 设置杠杆
    logger.info(">>> 正在设置杠杆为 5x...")
    success = client.set_leverage(inst_id=inst_id, lever=5, mgn_mode="cross")
    if not success:
        logger.error("设置杠杆失败，终止测试。")
        return
        
    # 3. 获取当前价格并计算所需张数
    df = client.get_candles(inst_id=inst_id, limit=1)
    if df.empty:
        logger.error("获取当前价格失败。")
        return
    current_price = df.iloc[-1]['close']
    logger.info(f"当前 ETH 价格: {current_price:.2f}")
    
    sz = client.calc_contract_size(20, current_price)
    logger.info(f"计划做空合约张数: {sz} (基于 20 USDT 和 5x 杠杆)")
    
    # 4. 开仓 (市价做空)
    logger.info(">>> 执行市价做空开仓...")
    order = client.place_market_short(inst_id=inst_id, sz=str(sz), td_mode="cross")
    if not order:
        logger.error("❌ 开仓失败！")
        return
        
    logger.info("✅ 开仓可能已提交，等待 3 秒后查询持仓...")
    time.sleep(3)
    
    # 5. 查询当前持仓状态
    logger.info(">>> 查询当前持仓...")
    positions = client.get_positions(inst_id=inst_id)
    if not positions:
        logger.warning("未查到持仓，订单可能未成交或已被其他服务平仓。")
    else:
        for p in positions:
            logger.info(f"当前持仓 >> 方向: {p.get('posSide', 'net')} | 张数: {p.get('pos')} | 强平价: {p.get('liqPx')} | 盈亏: {p.get('upl')}")
            
    # 6. 市价平仓
    logger.info(">>> 准备执行市价平仓...")
    close_success = client.close_position(inst_id=inst_id, td_mode="cross")
    if close_success:
        logger.info("✅ 平仓指令发送成功！")
    else:
        logger.error("❌ 平仓失败，请立即使登录 App 手动平仓！")
        return
        
    # 7. 结算显示
    time.sleep(2)  # 给服务器一点时间结算
    logger.info(">>> 检查最终持仓和余额...")
    final_positions = client.get_positions(inst_id=inst_id)
    if final_positions:
        logger.warning(f"⚠️ 警告！仍然存在持仓: {final_positions}")
    else:
        logger.info("目前无任何持仓，已安全退出。")
        
    final_balance = client.get_balance("USDT")
    logger.info(f"测试完毕 | 初始余额: {balance:.4f} USDT | 最终余额: {final_balance:.4f} USDT")
    logger.info(f"测试总盈亏变动（含手续费）: {final_balance - balance:.4f} USDT")
    logger.info("="*60)

if __name__ == "__main__":
    test_live_trade()
