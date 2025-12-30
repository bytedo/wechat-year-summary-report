"""
logger.py - 统一日志配置模块

提供项目级别的日志配置，支持控制台和文件输出。
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = None,
    level: int = logging.INFO,
    log_file: str = None,
    log_dir: str = None
) -> logging.Logger:
    """
    配置并返回日志记录器。
    
    参数:
        name: 日志记录器名称，默认为 root
        level: 日志级别
        log_file: 日志文件名（可选）
        log_dir: 日志目录（可选）
        
    返回:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复配置
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 控制台输出（带颜色）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(formatter))
    logger.addHandler(console_handler)
    
    # 文件输出（可选）
    if log_file or log_dir:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / 'output' / 'logs'
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            log_file = f"wechat_analyze_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(
            log_dir / log_file,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ColoredFormatter(logging.Formatter):
    """
    带颜色的日志格式化器（仅控制台）。
    """
    
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
    }
    RESET = '\033[0m'
    
    def __init__(self, base_formatter: logging.Formatter):
        self.base_formatter = base_formatter
    
    def format(self, record: logging.LogRecord) -> str:
        # 获取基础格式化结果
        message = self.base_formatter.format(record)
        
        # 添加颜色
        color = self.COLORS.get(record.levelname, '')
        if color:
            # 只给级别名称加颜色
            message = message.replace(
                f'| {record.levelname}',
                f'| {color}{record.levelname}{self.RESET}'
            )
        
        return message


# 默认 logger 实例
default_logger = None


def get_logger(name: str = 'wechat_analyze') -> logging.Logger:
    """
    获取默认的日志记录器。
    
    参数:
        name: 日志记录器名称
        
    返回:
        Logger 实例
    """
    global default_logger
    if default_logger is None:
        default_logger = setup_logger(name)
    return default_logger
