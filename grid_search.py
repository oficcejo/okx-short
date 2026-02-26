
import pandas as pd
import numpy as np
from itertools import product
from backtester import Backtester
from utils import logger
import argparse
import time
from tabulate import tabulate

def run_grid_search(data_path, period_range, multi_range, threshold_range):
    results = []
    
    # 构建所有组合
    combinations = list(product(period_range, multi_range, threshold_range))
    total = len(combinations)
    
    logger.info(f"开始网格搜索，总组合数: {total}")
    start_time = time.time()
    
    # 预加载数据以提高效率
    # 注意: Backtester 目前在构造函数里并不加载数据，而是在 run() 里加载
    # 为了避免重复读取硬盘，我们可以先读取一次 DataFrame (虽然 Backtester 是每次读取)
    # 但由于 CSV 很小，重复读取损耗可忽略。如果追求极致，可以重构 Backtester。
    
    for i, (period, multi, threshold) in enumerate(combinations):
        params = {
            "period": period,
            "vol_multiplier": multi,
            "pct_threshold": threshold
        }
        
        tester = Backtester(data_path=data_path, strategy_params=params)
        summary = tester.run(silent=True)
        
        res = {
            "period": period,
            "multiplier": multi,
            "threshold": threshold,
            **summary
        }
        results.append(res)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            msg = f"进度: {i+1}/{total} | 已耗时: {elapsed:.1f}s"
            logger.info(msg)
            print(msg, flush=True)
            
    df_results = pd.DataFrame(results)
    # 按 ROI 降序排列
    df_results = df_results.sort_values(by="roi", ascending=False)
    
    end_time = time.time()
    logger.info(f"网格搜索完成! 总耗时: {end_time - start_time:.1f}s")
    
    return df_results

def main():
    parser = argparse.ArgumentParser(description="网格搜索参数最优化")
    parser.add_argument("--data", type=str, required=True, help="历史K线数据路径")
    parser.add_argument("--top", type=int, default=10, help="显示前N个最优解")
    args = parser.parse_args()
    
    # 定义搜索范围
    period_range = range(120, 201, 10)         # 120, 130, ..., 200 (9个)
    multi_range = np.arange(3.0, 10.1, 1.0)    # 3, 4, ..., 10 (8个)
    threshold_range = np.arange(0.1, 1.1, 0.1)  # 0.1, 0.2, ..., 1.0 (10个)
    
    # 总计 9 * 8 * 10 = 720 个组合
    
    df = run_grid_search(args.data, period_range, multi_range, threshold_range)
    
    # 打印前 N 个
    print("\n" + "="*80)
    print(f"Top {args.top} 最佳参数组合 (基于 ROI)")
    print("="*80)
    print(tabulate(df.head(args.top), headers="keys", tablefmt="simple_outline", showindex=False))
    
    # 保存结果
    output_file = "grid_search_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n所有结果已保存至: {output_file}")

if __name__ == "__main__":
    main()
