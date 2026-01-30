# logger_utils.py
import logging
import os
import time

_logger_instance = None

def get_rank_for_logger():
    return int(os.environ.get("RANK", 0))

def setup_logger():
    global _logger_instance
    if _logger_instance is not None:
        return _logger_instance

    rank = get_rank_for_logger()
    logger = logging.getLogger(f"train_logger_rank{rank}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # 创建目录
        log_dir = "traininglogs"
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"train_log_rank{rank}.log")
        fh = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            f"[%(asctime)s][%(levelname)s][RANK {rank}] %(message)s",
            "%H:%M:%S"
        )
        formatter.converter = time.localtime
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Stream Handler（终端只打印 INFO 及以上）
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)  # ✅ 只显示 INFO 以上的日志
        sh_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        sh.setFormatter(sh_formatter)
        logger.addHandler(sh)

    _logger_instance = logger
    return logger