"""
ai/base.py - AI 分析器基类

包含 API 调用、重试机制、速率限制等核心功能。
"""

import os
import time
import logging
from functools import wraps
from typing import List, Callable, Any

from dotenv import load_dotenv

# 配置日志
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


def retry_on_failure(max_retries: int = 3, base_delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    API 调用重试装饰器，使用指数退避策略。
    
    参数:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        exceptions: 需要捕获的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)  # 指数退避: 1s, 2s, 4s
                        logger.warning(f"API 调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}，{delay:.1f}秒后重试...")
                        time.sleep(delay)
                    else:
                        logger.error(f"API 调用失败，已达最大重试次数: {e}")
            raise last_exception
        return wrapper
    return decorator


class AIAnalyzerBase:
    """
    AI 分析器基类，提供 API 调用和基础功能。
    
    通过 `LLM_REQUEST_DELAY` 环境变量控制请求间延迟（秒），默认 2 秒。
    """
    
    # 请求间延迟配置
    REQUEST_DELAY = float(os.getenv('LLM_REQUEST_DELAY', '2.0'))
    
    # 上次请求时间
    _last_request_time: float = 0
    
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        """
        初始化 AI 分析器。
        
        参数:
            base_url: API 基础地址
            api_key: API 密钥
            model: 模型名称
        """
        self.base_url = base_url or os.getenv('LLM_BASE_URL', 'https://api.deepseek.com/v1')
        self.api_key = api_key or os.getenv('LLM_API_KEY', '')
        self.model = model or os.getenv('LLM_MODEL', 'deepseek-chat')
        
        # 检查是否启用 Mock 模式
        self.mock_mode = not self.api_key or self.api_key == 'your-api-key-here'
        
        if not self.mock_mode:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
                )
            except ImportError:
                logger.warning("openai 库未安装，启用 Mock 模式")
                self.mock_mode = True
    
    def _wait_for_rate_limit(self):
        """
        等待以满足请求速率限制。
        
        确保两次 API 调用之间至少间隔 REQUEST_DELAY 秒。
        """
        current_time = time.time()
        time_since_last = current_time - AIAnalyzerBase._last_request_time
        
        if time_since_last < self.REQUEST_DELAY:
            wait_time = self.REQUEST_DELAY - time_since_last
            time.sleep(wait_time)
        
        AIAnalyzerBase._last_request_time = time.time()
    
    def _call_api(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_retries: int = 3
    ) -> str:
        """
        带速率限制和重试机制的 API 调用。
        
        参数:
            messages: 对话消息列表
            temperature: 生成温度
            max_tokens: 最大 token 数
            max_retries: 最大重试次数
            
        返回:
            API 响应内容
        """
        if self.mock_mode:
            return "[Mock 模式] 未配置 API Key"
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # 速率限制等待
                self._wait_for_rate_limit()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return self._extract_content(response)
                
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # 是否是可重试的错误
                retryable = any(x in error_msg for x in ['timeout', '504', '502', '503', 'rate limit'])
                
                if retryable and attempt < max_retries - 1:
                    delay = 2 * (attempt + 1)  # 2s, 4s, 6s
                    logger.warning(f"API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}，{delay}秒后重试...")
                    time.sleep(delay)
                else:
                    break
        
        # 所有重试都失败
        logger.error(f"API 调用失败: {last_exception}")
        raise last_exception
    
    def _extract_content(self, response) -> str:
        """
        从不同格式的 API 响应中提取内容。
        """
        # OpenAI 标准格式
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content or ""
            if hasattr(choice, 'text'):
                return choice.text or ""
        
        # 字符串
        if isinstance(response, str):
            return response
        
        # 字典
        if isinstance(response, dict):
            if 'choices' in response and response['choices']:
                choice = response['choices'][0]
                if 'message' in choice:
                    return choice['message'].get('content', '')
                return choice.get('text', '')
            return response.get('content', str(response))
        
        return str(response)


# 为兼容性导出（原 ai_analyzer.py 的接口）
# 实际 AIAnalyzer 类将在原文件中保持，逐步迁移时使用
class AIAnalyzer(AIAnalyzerBase):
    """
    完整的 AI 分析器（暂时保持兼容，后续逐步迁移各子模块）。
    
    注意：此类仅作为过渡，实际实现仍在 ai_analyzer.py 中。
    """
    pass
