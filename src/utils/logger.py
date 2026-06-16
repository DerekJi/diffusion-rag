"""日志工厂模块。"""

import logging


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。

    Args:
        name: 模块名称，通常传入 __name__。

    Returns:
        配置好的 Logger 实例。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
