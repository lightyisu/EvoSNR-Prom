
import argparse # 新增：用于处理命令行参数
# 1. 命令行参数解析 (Parse Command Line Arguments)
# 使用argparse让脚本更灵活
def parse_args():
    parser = argparse.ArgumentParser(description="EVOSNR Model Training and Evaluation Script.")

    # --- NEW: Add --seed argument ---
    parser.add_argument('--seed', type=int, default=1,
                        help="Random seed for reproducibility. Passed from Bash script.")
    # --- END NEW ---
    parser.add_argument('--transfer_mode', type=str, default='mixed',
                        choices=['mixed', 'source_only', 'target_only'],
                        help='Training mode: mixed, source_only, or target_only.')


    args = parser.parse_args()
    return args