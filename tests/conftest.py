"""测试用占位端点。

公开版 eval/config.py 的端点默认全空；部分单测需要一个"已配置"的平台地址才能走到
协议逻辑。必须在任何测试模块 import config/adapters 之前设置，所以放在 conftest 顶层。
"""
import os

for _k, _v in {
    "ASR_PLATFORM_URL": "http://platform.test:8000",
    "ASR_PLATFORM_WS_URL": "ws://platform.test:8000",
    "LIGHT_URL": "http://lite.test:8001",
    "EXT_ADV_URL": "http://ext.test:8002",
    "GEMMA_URL": "http://gemma.test:8003",
    "SENSEVOICE_URL": "http://sensevoice.test:8004",
    "QWEN3_ASR_URL": "http://qwen3.test:8005",
    "QWEN3_ASR_WS_URL": "ws://qwen3.test:8005/ws",
}.items():
    os.environ.setdefault(_k, _v)
