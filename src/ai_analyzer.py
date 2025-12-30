"""
ai_analyzer.py - AI 分析代理模块

使用 LLM 对聊天数据进行深度分析，包括话题总结、用户画像等。
支持 Mock 模式，当没有 API Key 时返回模拟数据。
"""

import os
import random
import re
import time
import logging
from functools import wraps
from typing import List, Optional, Callable, Any

import pandas as pd
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


class AIAnalyzer:
    """
    AI 分析器，支持 OpenAI 兼容接口（DeepSeek/Moonshot 等）。
    
    通过 `LLM_REQUEST_DELAY` 环境变量控制请求间延迟（秒），默认 2 秒。
    增加该值可有效降低 504 超时错误发生的概率。
    """
    
    # 请求间延迟配置（从环境变量读取，默认 2 秒）
    REQUEST_DELAY = float(os.getenv('LLM_REQUEST_DELAY', '2.0'))
    
    # 上次请求时间（用于计算需要等待的时间）
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
        
        确保两次 API 调用之间至少间隔 REQUEST_DELAY 秒，
        防止因请求过于密集导致 504 网关超时。
        """
        if AIAnalyzer._last_request_time > 0:
            elapsed = time.time() - AIAnalyzer._last_request_time
            if elapsed < self.REQUEST_DELAY:
                wait_time = self.REQUEST_DELAY - elapsed
                logger.debug(f"速率限制：等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
    
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
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # 请求前等待，满足速率限制
                self._wait_for_rate_limit()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # 更新上次请求时间
                AIAnalyzer._last_request_time = time.time()
                
                content = self._extract_content(response)
                if content:
                    return content
                raise ValueError("API 返回空内容")
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # 检测 504 网关超时错误，采用更长的退避时间
                is_504_error = '504' in error_str or 'gateway' in error_str or 'timeout' in error_str
                
                if attempt < max_retries:
                    if is_504_error:
                        # 504 错误使用更长的退避时间 (5s, 10s, 15s)
                        delay = 5.0 * (attempt + 1)
                        logger.warning(f"⚠️ API 504 超时 (尝试 {attempt + 1}/{max_retries + 1}): 服务端繁忙，{delay:.0f} 秒后重试...")
                    else:
                        # 普通错误使用指数退避 (2s, 4s, 8s)
                        delay = 2.0 * (2 ** attempt)
                        logger.warning(f"API 调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}，{delay:.1f} 秒后重试...")
                    
                    time.sleep(delay)
                    
                    # 504 错误后还需额外重置速率限制计时器
                    if is_504_error:
                        AIAnalyzer._last_request_time = time.time()
                else:
                    logger.error(f"API 调用失败，已达最大重试次数: {e}")
        
        raise last_exception
    
    def analyze(self, df: pd.DataFrame, top_users: List[dict]) -> dict:
        """
        执行 AI 分析。
        
        参数:
            df: 消息数据 DataFrame
            top_users: 活跃用户列表
            
        返回:
            AI 分析结果字典
        """
        if self.mock_mode:
            return self._mock_analyze(top_users)
        
        # 采样消息用于分析
        sampled_messages = self._sample_messages(df)
        
        # 构建分析提示
        prompt = self._build_prompt(sampled_messages, top_users)
        
        try:
            content = self._call_api(
                messages=[
                    {"role": "system", "content": "你是一位充满诗意与温情的群聊回忆官，像老朋友一样记录着每一个珍贵瞬间。你相信每段对话背后都有故事，每位群友都是独一无二的存在。你的文字温暖如冬日暖阳，让读者感受到友谊的力量与时光的珍贵。用心发现那些看似平凡却闪闪发光的日常，让每个人都能在你的报告中找到属于自己的温馨记忆。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return self._parse_response(content, top_users)
            
        except Exception as e:
            logger.warning(f"AI 分析失败: {e}，使用 Mock 数据")
            return self._mock_analyze(top_users)
    
    def _sample_messages(self, df: pd.DataFrame, per_month: int = 8, min_length: int = 5) -> List[dict]:
        """
        按月均匀采样消息，确保时间分布均衡。
        
        参数:
            df: 消息 DataFrame
            per_month: 每月采样消息数（确保时间均匀分布）
            min_length: 最小消息长度
        """
        sampled = []
        
        # 添加月份列
        df = df.copy()
        df['month'] = df['date'].str[:7]  # YYYY-MM
        
        # 按月份分组采样
        for month, group in df.groupby('month'):
            # 过滤较长的消息
            long_messages = group[group['content'].str.len() > min_length]
            
            if long_messages.empty:
                continue
            
            # 每月均匀采样
            sample_size = min(per_month, len(long_messages))
            month_sample = long_messages.sample(n=sample_size)
            
            for _, row in month_sample.iterrows():
                # 截断消息长度，避免过长内容触发审核
                content = self._sanitize_content(row['content'])
                if len(content) > 80:
                    content = content[:80] + '...'
                sampled.append({
                    'user': row['user'],
                    'content': content,
                    'date': row['date']
                })
        
        return sampled
    
    def _sanitize_content(self, text: str) -> str:
        """
        隐私脱敏：替换手机号等敏感信息。
        """
        # 替换手机号
        text = re.sub(r'1[3-9]\d{9}', '138****0000', text)
        # 替换邮箱
        text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '***@**.com', text)
        # 替换身份证号
        text = re.sub(r'\d{17}[\dXx]', '****', text)
        
        return text
    
    def _build_prompt(self, messages: List[dict], top_users: List[dict]) -> str:
        """
        构建分析提示词。
        """
        # 按月份整理消息
        from collections import defaultdict
        monthly_msgs = defaultdict(list)
        for m in messages:
            month = m['date'][:7]  # YYYY-MM
            monthly_msgs[month].append(m)
        
        # 格式化按月消息（每月最多10条）
        msg_sections = []
        for month in sorted(monthly_msgs.keys()):
            month_num = int(month.split('-')[1])
            month_name = f"{month_num}月"
            month_messages = monthly_msgs[month][:10]
            msg_text = "\n".join([
                f"  - {m['user']}: {m['content']}"
                for m in month_messages
            ])
            msg_sections.append(f"### {month_name}\n{msg_text}")
        
        all_msg_text = "\n\n".join(msg_sections)
        
        # 格式化所有用户（用于人物画像）
        all_user_names = [u['user'] for u in top_users]
        
        # 生成月份列表
        months_list = [f"{int(m.split('-')[1])}月" for m in sorted(monthly_msgs.keys())]
        
        prompt = f"""请分析以下微信群聊记录，生成一份充满温情的年度群聊报告。

想象你是这个群的老朋友，在年末翻看过去一年的聊天记录，想要为大家写一封温暖的年度回忆信。

## 群聊消息样本（按月整理）:
{all_msg_text}

## 群成员列表: {', '.join(all_user_names)}
## 包含月份: {', '.join(months_list)}

---

## 分析任务

请按以下格式输出 Markdown 报告：

### 🎯 群聊年度关键词
用 3 个能触动人心的关键词总结这个群的氛围（如：陪伴、温暖、成长、欢笑等）。
这些关键词应该让群友们看到时会心一笑，想起那些美好的日子。

### 👥 群友画像墙
为**每一位**群成员生成一句温暖的画像（包括：{', '.join(all_user_names)}）。
- 用温柔、欣赏的语气，像是在介绍自己珍视的朋友
- 发现并放大每个人身上的闪光点，让 TA 感受到被看见
- 给每人一个带有温度的"称号"（如：深夜陪聊家、群聊小太阳、永远在线的倾听者等）
- 让每个人读到自己的画像时，都能感受到群友们的喜爱

### 📅 月度话题回顾
请为**每一个月**（{', '.join(months_list)}）都生成详细的话题回顾，格式如下：

**📌 1月**
- 🔹 话题1：简短描述（1-2句话说明大家聊了什么，要有故事感）
- 🔹 话题2：简短描述
- 🔹 话题3：简短描述

**📌 2月**
...（以此类推，每个月都要有 2-4 个话题）

**⚠️ 每个月都必须有内容，不能跳过任何月份！**

### ✨ 年度温馨时刻
挑选 2-3 个最能体现群友情谊的**温馨瞬间**，可以是：
- 有人遇到困难时，大家齐心帮忙的场景
- 深夜还有人陪聊的温暖
- 让大家笑出声的有趣对话
- 节日里互相祝福的温情

---

## ⚠️ 重要规则
1. **用心写**：想象你在给最好的朋友们写年终信，字里行间都是真挚的情感
2. **有温度**：让每句话都能让读者感受到群聊的温暖和归属感
3. **讲故事**：不是干巴巴地列清单，而是用故事串联起这一年的回忆
4. **每个人都重要**：确保每个人都有画像，让大家都感受到被珍视
5. **避开敏感话题**：用"日常趣事"等概括性描述替代不适合公开的内容
6. **正向积极**：即使是吐槽，也要转化为轻松有趣的回忆
"""
        return prompt
    
    def _extract_content(self, response) -> str:
        """
        从不同格式的 API 响应中提取内容。
        支持：OpenAI 标准格式、字符串、字典等多种格式。
        """
        # 如果是字符串，直接返回
        if isinstance(response, str):
            return response
        
        # 标准 OpenAI 格式
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content
            if hasattr(choice, 'text'):
                return choice.text
        
        # 字典格式
        if isinstance(response, dict):
            # OpenAI 格式的字典
            if 'choices' in response and response['choices']:
                choice = response['choices'][0]
                if isinstance(choice, dict):
                    if 'message' in choice and 'content' in choice['message']:
                        return choice['message']['content']
                    if 'text' in choice:
                        return choice['text']
            # 直接有 content 字段
            if 'content' in response:
                return response['content']
            # 直接有 text 字段
            if 'text' in response:
                return response['text']
        
        # 尝试转换为字符串
        return str(response) if response else None
    
    def _parse_response(self, content: str, top_users: List[dict]) -> dict:
        """
        解析 AI 响应。
        """
        return {
            'raw_content': content,
            'keywords': self._extract_keywords(content),
            'user_profiles': self._extract_user_profiles(content, top_users),
            'topics': self._extract_topics(content),
            'highlights': self._extract_highlights(content),
        }
    
    def _extract_keywords(self, content: str) -> List[str]:
        """尝试从内容中提取关键词"""
        # 简单实现，实际可以用更精确的匹配
        keywords = ['欢乐', '吐槽', '互助']
        return keywords
    
    def _extract_user_profiles(self, content: str, top_users: List[dict]) -> List[dict]:
        """尝试提取用户画像"""
        profiles = []
        for user in top_users[:3]:
            profiles.append({
                'user': user['user'],
                'description': '群内活跃分子，话题发起者'
            })
        return profiles
    
    def _extract_topics(self, content: str) -> List[str]:
        """尝试提取话题"""
        return ['日常闲聊', '工作吐槽', '生活分享']
    
    def _extract_highlights(self, content: str) -> str:
        """提取精彩片段"""
        return '群友们的日常欢乐时光~'
    
    def _mock_analyze(self, top_users: List[dict]) -> dict:
        """
        Mock 模式：返回模拟数据。
        """
        user_names = [u['user'] for u in top_users[:3]]
        
        mock_content = f"""## 🎯 群聊年度关键词

**陪伴** | **温暖** | **一起笑**

这一年，我们用文字搭建起一座温暖的小屋——在这里，有人分享喜悦，有人倾诉烦恼，更多的是互相陪伴、一起成长的日常。

---

## 👥 我们的宝藏群友

{"" if not user_names else f'''
### 🥇 {user_names[0] if len(user_names) > 0 else "神秘人"}
群里的小太阳，每天准时给大家带来元气。有 TA 的地方，就有欢声笑语。谢谢你，总是第一个冒泡，让群里永远不冷清！

### 🥈 {user_names[1] if len(user_names) > 1 else "隐藏大佬"}  
深夜的暖心陪伴者，总在大家需要的时候出现。虽然话不多，但每一句都恰到好处，是我们的定心丸。

### 🥉 {user_names[2] if len(user_names) > 2 else "潜水达人"}
快乐的传播者，表情包选手。总能在恰当的时刻用一个表情包化解尴尬、点燃气氛，群里的开心果！
'''}

---

## � 我们的温馨日常

1. **早安晚安** - 每一天，都有人在群里说"早"，这份坚持本身就很温暖
2. **深夜陪聊** - 不管多晚，总有人愿意听你说话，这就是友情
3. **互相打气** - 工作累了、生活烦了，群里总有人给你加油打气
4. **一起笑** - 那些让我们笑到肚子疼的瞬间，是这一年最珍贵的记忆
5. **默默关心** - 有人请假没冒泡，总会有人问一句"最近还好吗"

---

## ✨ 年度温馨时刻

> "最让人感动的，是某个深夜有人说'睡不着'，马上就有人回'我也是，聊聊？'

> 隔着屏幕，我们也能感受到彼此的温度。"

这一年，感谢有你们。不管未来怎样，这份友情，我们会一直记得。❤️

---

*（这份报告或许简单，但每一个字都承载着我们共同的回忆）*
"""
        
        return {
            'raw_content': mock_content,
            'keywords': ['欢乐', '吐槽', '互帮互助'],
            'user_profiles': [
                {'user': user_names[0] if user_names else '用户1', 'description': '群内话痨担当'},
                {'user': user_names[1] if len(user_names) > 1 else '用户2', 'description': '深夜冲浪选手'},
                {'user': user_names[2] if len(user_names) > 2 else '用户3', 'description': '表情包大师'},
            ],
            'topics': ['日常打卡', '美食分享', '工作吐槽', '游戏开黑', '生活琐事'],
            'highlights': '今年最难忘的一刻...',
            'is_mock': True
        }
    
    def summarize_clusters(self, cluster_representatives: dict) -> dict:
        """
        使用 LLM 为聚类生成有意义的话题名称。
        
        参数:
            cluster_representatives: 每个聚类的代表性消息
                {0: [{'content': '...', 'user': '...'}, ...], 1: [...], ...}
                
        返回:
            话题名称字典 {0: '话题名', 1: '话题名', ...}
        """
        if self.mock_mode:
            return self._mock_summarize_clusters(cluster_representatives)
        
        # 构建 Prompt
        prompt = self._build_cluster_prompt(cluster_representatives)
        
        try:
            content = self._call_api(
                messages=[
                    {
                        "role": "system",
                        "content": "你是群聊星系的命名大师，擅长用电影片名或小说章节的风格为话题组起名。你的名字要有故事感、画面感，能让群友一看就想起那些美好时光。严禁使用功能性、事务性、负面的命名。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return self._parse_cluster_names(content, cluster_representatives)
            
        except Exception as e:
            logger.warning(f"话题命名失败: {e}，使用默认名称")
            return self._mock_summarize_clusters(cluster_representatives)
    
    def _build_cluster_prompt(self, cluster_representatives: dict) -> str:
        """构建聚类命名的提示词。"""
        lines = ["以下是通过向量算法自动聚类的群聊消息组，请为每组起一个充满故事感的名字。\n"]
        
        for cluster_id, messages in cluster_representatives.items():
            if not messages:
                continue
            
            lines.append(f"## 分组 {cluster_id}")
            for msg in messages[:8]:  # 每组展示8条增加理解
                content = msg['content'][:80]  # 截断长内容
                lines.append(f"- {msg['user']}: {content}")
            lines.append("")
        
        lines.append("""
请返回 JSON 格式的结果，例如：
{"0": "深夜食堂", "1": "午后摸鱼时光", "2": "周末奇遇记"}

## 命名风格要求：
- **像电影片名**：有画面感、故事感（如"深夜食堂"、"那些年我们一起追的剧"）
- **像小说章节**：温馨有趣（如"午后摸鱼时光"、"打工人的日常"）
- **简短有力**：2-6个字，朗朗上口
- **勾起回忆**：让群友一看就能想起那些对话

## ⛔ 禁止使用：
- 功能性命名（如"需求招募"、"问题解答"、"信息咨询"）
- 抽象命名（如"难以启齿"、"深度交流"、"综合讨论"）
- 负面命名（如"分手"、"吐槽"、"抱怨"、"冲突"）""")
        
        return "\n".join(lines)
    
    def _parse_cluster_names(self, content: str, cluster_representatives: dict) -> dict:
        """解析 LLM 返回的话题名称。"""
        import json
        
        # 尝试提取 JSON
        try:
            # 尝试直接解析
            names = json.loads(content)
            return {int(k): v for k, v in names.items()}
        except:
            pass
        
        # 尝试从文本中提取 JSON 块
        import re
        json_match = re.search(r'\{[^{}]+\}', content)
        if json_match:
            try:
                names = json.loads(json_match.group())
                return {int(k): v for k, v in names.items()}
            except:
                pass
        
        # 失败则返回默认名称
        return self._mock_summarize_clusters(cluster_representatives)
    
    def _mock_summarize_clusters(self, cluster_representatives: dict) -> dict:
        """Mock 模式：返回默认话题名称。"""
        default_names = [
            "日常闲聊", "技术交流", "午餐拼单",
            "吐槽大会", "表情包互动", "深夜emo",
            "摸鱼时间", "周末计划", "生活分享"
        ]
        
        result = {}
        for i, cluster_id in enumerate(cluster_representatives.keys()):
            name = default_names[i % len(default_names)]
            result[cluster_id] = name
        
        return result
    
    def refine_keywords(self, raw_keywords: list, sample_messages: list = None) -> list:
        """
        使用 AI 筛选并优化年度关键词。
        
        参数:
            raw_keywords: jieba 分词后的高频词列表 [{'word': '...', 'count': N}, ...]
            sample_messages: 可选，消息样本用于上下文理解
            
        返回:
            [{'word': '关键词'}, ...]
        """
        if not raw_keywords:
            return []
        
        if self.mock_mode:
            return self._mock_refine_keywords(raw_keywords)
        
        # 提取词语列表
        words_text = "、".join([f"{kw['word']}({kw['count']}次)" for kw in raw_keywords[:50]])
        
        prompt = f"""请从以下高频词中筛选出 **8-12 个** 最能代表群聊年度特色的关键词。

## 候选高频词（按出现次数）：
{words_text}

## 筛选标准：
1. **有意义的词语**：名词、动词、形容词，能让人联想到具体场景
2. **群聊特色词**：能体现群友们共同话题、习惯或记忆的词
3. **排除无意义词**：如"一个"、"然后"、"这个"、"什么"等口水词
4. **排除过于通用的词**：如"知道"、"可以"、"没有"等

## 输出格式：
直接返回一个 JSON 字符串数组，例如：
["加班", "奶茶", "摸鱼", "开会", "周末"]

请筛选出最能唤起群友回忆的词语："""
        
        try:
            content = self._call_api(
                messages=[
                    {"role": "system", "content": "你是群聊年度回忆编辑，擅长从高频词中发现能唤起群友共同回忆的关键词。只输出JSON字符串数组。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            # 解析 JSON
            import json
            import re
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                words = json.loads(json_match.group())
                # 转换为统一格式
                return [{'word': w} for w in words[:12] if isinstance(w, str)]
            
        except Exception as e:
            logger.warning(f"关键词筛选失败: {e}")
        
        return self._mock_refine_keywords(raw_keywords)
    
    def _mock_refine_keywords(self, raw_keywords: list) -> list:
        """Mock 模式：简单过滤返回"""
        # 简单过滤，去除太短或太通用的词
        stopwords = {'一个', '这个', '那个', '什么', '怎么', '可以', '没有', '知道', '然后', '现在', '时候', '因为', '所以', '但是', '还是', '已经', '就是', '不是', '真的', '觉得'}
        filtered = [
            {'word': kw['word']}
            for kw in raw_keywords[:15]
            if kw['word'] not in stopwords and len(kw['word']) >= 2
        ]
        return filtered[:10]
    
    def select_golden_quotes(self, hot_messages: list) -> list:
        """
        使用 AI 从热门消息中甄选金句。
        
        参数:
            hot_messages: 热门消息列表（来自 stats_engine）
            
        返回:
            [{'user': '用户', 'content': '金句', 'reason': '入选理由'}, ...]
        """
        if not hot_messages:
            return []
        
        if self.mock_mode:
            return self._mock_golden_quotes(hot_messages)
        
        # 扩大候选范围，选取更多消息供 AI 筛选
        candidates = hot_messages[:30]
        msg_text = "\n".join([
            f"{i+1}. [{m['user']}]: {m['content']}"
            for i, m in enumerate(candidates)
        ])
        
        prompt = f"""你是这个群聊的"年度回忆官"，正在为群友们整理一份最珍贵的"年度金句集"。

## 候选消息：
{msg_text}

## 任务要求：
请从这些消息中，精选出 **8-12 条** 最值得被记住的话语。

这些金句应该让群友们看到时：
- 忍不住笑出声
- 或者心头一暖
- 或者感叹"说得真好"
- 或者想起当时热闹的场景

### 类别说明：
- 😂 **笑到流泪**：让大家笑出腹肌的神句
- 💡 **醍醐灌顶**：群友的人生智慧
- 🔥 **经典名场面**：引发全员接龙的高光时刻
- 💖 **心里暖暖的**：那些被温暖到的瞬间
- 🎭 **神仙回复**：教科书级别的神回

## 输出格式（JSON 数组）：
[
  {{"user": "用户名", "content": "完整金句内容", "reason": "入选理由（10字内，要温馨有趣）", "category": "类别标签"}}
]

## 写作要求：
- 入选理由要像在夸奖好朋友一样自然温暖
- 优先选那些能唤起美好回忆的话
- 让每一条金句都承载着群友间的情谊
- 金句内容保持原样"""
        
        try:
            content = self._call_api(
                messages=[
                    {"role": "system", "content": "你是群友们的年度回忆官，用温暖的视角发现每一句值得被珍藏的话语。你相信平凡对话中藏着最真挚的情谊，善于发现那些让人会心一笑或心头一暖的瞬间。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return self._parse_golden_quotes(content, candidates)
            
        except Exception as e:
            logger.warning(f"金句甄选失败: {e}，使用默认")
            return self._mock_golden_quotes(hot_messages)
    
    def _parse_golden_quotes(self, content: str, candidates: list) -> list:
        """解析 AI 返回的金句。"""
        import json
        
        try:
            # 尝试提取 JSON 数组
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                quotes = json.loads(json_match.group())
                return quotes[:5]
        except:
            pass
        
        return self._mock_golden_quotes(candidates)
    
    def _mock_golden_quotes(self, hot_messages: list) -> list:
        """Mock 模式：返回模拟金句。"""
        quotes = []
        for msg in hot_messages[:3]:
            quotes.append({
                'user': msg['user'],
                'content': msg['content'][:50],
                'reason': '引发热烈讨论'
            })
        return quotes
    
    def summarize_peak_day(self, peak_day_data: dict) -> str:
        """
        为巅峰日生成 50 字摘要。
        
        参数:
            peak_day_data: {'date': '日期', 'count': 数量, 'sample_messages': [...]}
            
        返回:
            50 字左右的摘要
        """
        if not peak_day_data or not peak_day_data.get('date'):
            return "这一天，群里格外热闹，大家畅所欲言..."
        
        if self.mock_mode:
            return f"那一天（{peak_day_data['date']}），群里共产生了 {peak_day_data['count']} 条消息，大家聊得格外开心，话题一个接一个停不下来！"
        
        # 构建样本消息文本
        samples = peak_day_data.get('sample_messages', [])
        msg_text = "\n".join([
            f"- {m['user']}: {m['content'][:30]}"
            for m in samples[:5]
        ])
        
        prompt = f"""这是群里最热闹的一天（{peak_day_data['date']}，{peak_day_data['count']} 条消息刷屏！）是什么让大家如此热情呢？

当天消息片段：
{msg_text}

请用 50 字左右，像在给老朋友讲故事一样，温馨地描述那天的热闹场景。
开头用"那一天，"作为引子。
让读到这段话的人，仿佛能感受到当时群里欢腾的气氛。"""
        
        try:
            content = self._call_api(
                messages=[
                    {"role": "system", "content": "你是群友们的专属回忆官，用故事的方式记录那些珍贵的日子。你的文字总是带着温度，让人读完后嘴角上扬。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            return content.strip() if content else self._mock_peak_day_summary(peak_day_data)
            
        except Exception as e:
            logger.warning(f"巅峰日摘要失败: {e}")
            return self._mock_peak_day_summary(peak_day_data)
    
    def _mock_peak_day_summary(self, peak_day_data: dict) -> str:
        """Mock 模式：返回默认巅峰日摘要。"""
        return f"那一天（{peak_day_data.get('date', '某天')}），群里产生了 {peak_day_data.get('count', 0)} 条消息，欢声笑语不断，友谊在这里升温！"
    
    def generate_topic_memories(self, monthly_data: list) -> list:
        """
        为每个月生成话题回忆描述。
        
        参数:
            monthly_data: 月度分析数据列表（来自 monthly_analyzer）
            
        返回:
            [{'month': '1月', 'topics': [...], 'memory': '回忆描述'}, ...]
        """
        if not monthly_data:
            return []
        
        results = []
        
        for month_info in monthly_data:
            # 获取本月样本消息
            samples = month_info.get('sample_messages', [])
            old_topics = month_info.get('topics', [])
            
            if self.mock_mode or not samples:
                ai_result = {
                    'topics': [],
                    'memory': self._mock_topic_memory(month_info)
                }
            else:
                ai_result = self._generate_month_memory(month_info, samples, old_topics)
            
            results.append({
                'month': month_info.get('month_name', ''),
                'month_key': month_info.get('month', ''),
                'topics': ai_result.get('topics', []),  # AI 提取的具体话题
                'memory': ai_result.get('memory', ''),
                'stats': month_info.get('stats', {})
            })
        
        return results
    
    def _generate_month_memory(self, month_info: dict, samples: list, topics: list) -> dict:
        """使用 AI 生成单月话题回忆和具体话题列表。"""
        month_name = month_info.get('month_name', '本月')
        
        # 构建消息样本（增加数量）
        msg_text = "\n".join([
            f"- {m['user']}: {m['content'][:60]}"
            for m in samples[:15]
        ])
        
        prompt = f"""请为群聊的 {month_name} 写一份温暖的月度回忆录。

## 本月消息样本：
{msg_text}

## 任务：
1. 提取 3-5 个本月**具体发生的温馨故事/话题**（是能让群友们回忆起来的事情）
2. 写一段 80-100 字的月度回忆，像是在给老朋友寄去的明信片

## 写作指引：
- 用"这个月"开头，像讲故事一样娓娓道来
- 让群友读到时能想起那些快乐时光
- 文字要温暖，像冬日里的热可可

## 输出格式（严格JSON）：
{{
  "topics": [
    {{"title": "话题标题（4-8字，要有故事感）", "desc": "一句话描述，温馨有趣"}},
    ...
  ],
  "memory": "这个月，..."
}}

## 示例：
{{
  "topics": [
    {{"title": "小明的惊喜生日", "desc": "凌晨准时送上祝福，暖到心坎"}},
    {{"title": "打工人互助联盟", "desc": "加班吐槽里藏着互相打气"}},
    {{"title": "跨年约定", "desc": "期待着一起迎接新年"}}
  ],
  "memory": "这个月，群里充满了温馨的惊喜——大家一起给小明庆生，虽然隔着屏幕，祝福却暖暖的..."
}}

## 要求：
- 话题要**具体且温暖**，能唤起美好回忆
- 描述要**有温度**，让人读了嘴角上扬
- 避免敏感话题，用正向方式描述"""
        
        try:
            content = self._call_api(
                messages=[
                    {"role": "system", "content": "你是一位用心记录友情故事的回忆官，擅长从日常对话中发现那些闪闪发光的温馨时刻。只输出JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # 解析 JSON
            import json
            import re
            
            # 尝试提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    'topics': result.get('topics', []),
                    'memory': result.get('memory', self._mock_topic_memory(month_info))
                }
            else:
                return {
                    'topics': [],
                    'memory': content.strip() if content else self._mock_topic_memory(month_info)
                }
            
        except Exception as e:
            logger.warning(f"{month_name}话题提取失败: {e}")
            return {
                'topics': [],
                'memory': self._mock_topic_memory(month_info)
            }
    
    def analyze_weekly_batches(self, weekly_samples: list, use_cache: bool = True) -> tuple[str, dict]:
        """
        按周批次分析消息，生成年度深度总结。
        
        参数:
            weekly_samples: 每周消息样本列表 [{'week': '...', 'messages': [...]}, ...]
            use_cache: 是否使用缓存（默认开启）
            
        返回:
            (年度总结文本, 每周总结字典 {'2024-W01': '总结内容...'})
        """
        if not weekly_samples:
            return "", {}
        
        if self.mock_mode:
            return "（Mock模式跳过深度周度分析）", {}
        
        # === 缓存处理 ===
        import hashlib
        import json
        from pathlib import Path
        
        # 缓存放在项目的 tmp 目录
        cache_dir = Path(__file__).parent.parent / "tmp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算数据哈希（用于判断数据是否变化）
        cache_key_data = json.dumps([
            {'week': w['week'], 'count': len(w['messages']), 
             'sample': w['messages'][0]['content'][:50] if w['messages'] else ''}
            for w in weekly_samples
        ], ensure_ascii=False, sort_keys=True)
        cache_hash = hashlib.md5(cache_key_data.encode()).hexdigest()[:12]
        cache_file = cache_dir / f"weekly_analysis_{cache_hash}.json"
        
        # 尝试读取缓存
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                    print(f"   💾 已加载周度分析缓存 ({len(cached.get('weekly_summaries', {}))} 周)")
                    return cached.get('yearly_summary', ''), cached.get('weekly_summaries', {})
            except Exception as e:
                logger.warning(f"缓存读取失败: {e}")
        
        print(f"   🧠 正在进行深度周度分析 (共 {len(weekly_samples)} 周)...")
        
        weekly_summaries_dict = {}
        weekly_summaries_text_list = []
        
        # 尝试使用 tqdm 进度条
        try:
            from tqdm import tqdm
            week_iter = tqdm(weekly_samples, desc="   周度分析", unit="周", ncols=60)
        except ImportError:
            week_iter = weekly_samples
        
        # 每周单独分析 (带跨周上下文)
        previous_summary = ""  # 用于连贯剧情
        for i, week_data in enumerate(week_iter):
            week_label = week_data['week']
            msgs = week_data['messages']
            if not msgs:
                continue
                
            # 构建消息文本 (只取 user: content) - 扩大到 15000 字符
            msg_text = "\n".join([f"{m['user']}: {m['content']}" for m in msgs])
            
            # 构建跨周上下文
            context_section = ""
            if previous_summary:
                context_section = f"""
## 上周回顾（用于剧情连贯）：
{previous_summary[:500]}
---
"""
            
            prompt = f"""你是群友们的年度回忆官，正在为大家撰写一份能勾起美好回忆的周报。

{context_section}## {week_label} 本周消息记录：
{msg_text[:15000]}

## 写作任务：
请用温暖、怀旧的笔触，总结这周群里的温馨瞬间和有趣故事。

## 写作要求：
1. **具体事件**：列出 2-3 个让人印象深刻的话题或故事，要有细节（谁说了什么、发生了什么）
2. **金句提取**：挑选 1-2 个让人会心一笑的梗或金句
3. **剧情连贯**：如果有上周延续的话题，自然地串联起来
4. **文风要求**：
   - 像老朋友在回忆往事，娓娓道来
   - 让群友读到时能想起当时的情景
   - 语气温暖幽默，让人读完嘴角上扬

## ⛔ 禁止事项：
- 严禁提及分手、离婚、冲突、吵架、抱怨等负面话题
- 如果消息中有负面内容，请忽略或转化为轻松的吐槽风格
- 保持整体积极、温馨的基调"""

            try:
                summary = self._call_api(
                    messages=[
                        {"role": "system", "content": "你是一位温暖的回忆录作家，擅长从日常对话中发现闪光时刻，用充满故事感的文字让读者重温美好。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6,
                    max_tokens=500
                )
                weekly_summaries_dict[week_label] = summary
                weekly_summaries_text_list.append(f"### {week_label}\n{summary}")
                previous_summary = summary  # 更新跨周上下文
                print(f"     ✓ {week_label} 分析完成")
            except Exception as e:
                logger.warning(f"{week_label} 分析失败: {e}")
        
        # 汇总所有周报生成年度总结
        all_summaries = "\n\n".join(weekly_summaries_text_list)
        
        final_prompt = f"""这一年，我们在群里留下了无数温暖的回忆。现在，请为大家写一份让人想搬出小板凳细细品读的年度回忆录。

## 每周摘要（满满的回忆）：
{all_summaries}

## 任务：
请生成一篇让人读完会心头一暖的年度回忆文章（Markdown格式）：

### 🎬 如果这一年是一部电影
给它起个温暖的名字，写一段让人充满期待的"剧情简介"（50字左右）。

### 🌟 年度高光时刻
列出 5 个最值得纪念的温馨场景或欢乐事件：
- 要结合具体周的内容细节
- 让群友们能第一时间想起当时的快乐
- 用画面感强的语言描述

### 📜 友谊编年史
用时间线串起这一年的故事：
- 像给未来的自己写信一样
- 记录大家的变化和成长
- 突出那些"只有我们懂"的瞬间

### 💬 我们的专属记忆
那些只有我们才懂的梗和口头禅：
- 是这一年友谊的印记
- 让每个读到的人会心一笑

## 写作风格要求：
1. **勾起回忆**：像老朋友翻着相册聊往事，每个细节都能让人想起当时的情景
2. **温情脉脉**：字里行间都是对群友的珍视，让读者感受到"我们是一伙的"
3. **幽默暖心**：幽默中带着暖意，让人笑着笑着就觉得很幸福
4. **细节为王**：引用具体的人名、事件、金句，让群友能第一时间对号入座

## ⛔ 禁止事项：
- 严禁提及分手、离婚、冲突、吵架、抱怨等负面话题
- 保持整体积极、温馨、怀旧的基调"""

        try:
            print("   📝 正在生成年度深度总结...")
            content = self._call_api(
                messages=[
                    {"role": "system", "content": "你是一位文笔细腻的回忆录作家，擅长用温暖怀旧的笔触把平凡日常写成让人动容的故事。你的文字能勾起读者心底最柔软的回忆。"},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.75,
                max_tokens=2500
            )
            
            # 保存缓存
            if use_cache:
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'yearly_summary': content,
                            'weekly_summaries': weekly_summaries_dict
                        }, f, ensure_ascii=False, indent=2)
                    print(f"   💾 周度分析已缓存")
                except Exception as e:
                    logger.warning(f"缓存保存失败: {e}")
            
            return content, weekly_summaries_dict
        except Exception as e:
            logger.warning(f"年度总结生成失败: {e}")
            return "年度总结生成失败", weekly_summaries_dict

    def generate_monthly_summary_from_weekly(
        self, 
        monthly_data: list, 
        weekly_summaries: dict
    ) -> list:
        """
        基于周度总结生成月度话题回忆（更精准）。
        """
        results = []
        import datetime
        
        for month_info in monthly_data:
            month_key = month_info.get('month', '') # 'YYYY-MM'
            month_name = month_info.get('month_name', '')
            
            # 找到属于该月的所有周总结
            relevant_weeks = []
            for week_key, summary in weekly_summaries.items():
                try:
                    # 简单判断：如果周的字符串里包含年月... 兼容 YYYY-MM
                    # 或者解析 ISO 周
                    if month_key in week_key:
                         relevant_weeks.append(summary)
                    else:
                        # 尝试通过日期计算
                        y, w = week_key.split('-W')
                        week_start = datetime.datetime.strptime(f'{y}-W{w}-1', "%Y-W%W-%w")
                        if week_start.strftime('%Y-%m') == month_key:
                            relevant_weeks.append(summary)
                except:
                    pass
            
            if not relevant_weeks:
                # 尝试普通生成或Mock
                if self.mock_mode:
                    memory = self._mock_topic_memory(month_info)
                    topics = []
                else:
                    # 如果没有周数据，还是调用原来的方法，或者返回空
                    # 这里为了健壮性，调用原来的采样方法
                    samples = month_info.get('sample_messages', [])
                    if samples:
                        res = self._generate_month_memory(month_info, samples, [])
                        memory = res.get('memory', '')
                        topics = res.get('topics', [])
                    else:
                        memory = self._mock_topic_memory(month_info)
                        topics = []
            else:
                # 使用周报汇总生成月报
                combined_weekly = "\n".join(relevant_weeks)
                prompt = f"""以下是 {month_name} 里那些值得珍藏的群聊时光：
{combined_weekly}

请基于这些回忆，写一段 80-100 字的**月度温馨回忆**。
同时提取 3 个最能触动人心的话题标签。

写作指南：
- 用"这个月"开头，让全文像是在给老朋友写信
- 让群友读到时能想起那些快乐时光
- 文字要温暖，像冬日里的一杯热茶

输出格式（JSON）：
{{
  "topics": [{{"title": "...", "desc": "..."}}],
  "memory": "..."
}}"""
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "你是一个精炼的群聊记录员。输出JSON。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7
                    )
                    content = self._extract_content(response)
                    import json, re
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        res = json.loads(json_match.group())
                        memory = res.get('memory', '')
                        topics = res.get('topics', [])
                    else:
                        memory = content
                        topics = []
                except Exception as e:
                    print(f"   ⚠️ {month_name} 汇总失败: {e}")
                    memory = self._mock_topic_memory(month_info)
                    topics = []

            results.append({
                'month': month_name,
                'month_key': month_key,
                'topics': topics,
                'memory': memory,
                'stats': month_info.get('stats', {})
            })
            
        return results

    def generate_user_profiles_with_mbti(self, df: pd.DataFrame, top_users: List[str]) -> List[dict]:
        """
        生成用户画像及 MBTI 预测。
        
        参数:
            df: 完整消息 DataFrame
            top_users: 需要分析的用户列表 (用户名)
            
        返回:
            [{'user': '...', 'persona': '...', 'description': '...', 'mbti': '...'}, ...]
        """
        profiles = []
        if self.mock_mode:
            return self._mock_user_profiles_mbti(top_users)
        
        print(f"   👥 正在生成用户画像及 MBTI (分析前 {len(top_users)} 位活跃用户)...")
        
        # 尝试使用 tqdm 进度条
        try:
            from tqdm import tqdm
            user_iter = tqdm(top_users, desc="   用户画像", unit="人", ncols=60)
        except ImportError:
            user_iter = top_users
        
        for user in user_iter:
            # 提取该用户的发言样本 - 扩大到 1000 条以获得更准确的画像
            user_df = df[df['user'] == user]
            sample_size = min(1000, len(user_df))
            user_msgs = user_df['content'].sample(n=sample_size).tolist()
            msg_text = "\n".join(user_msgs)[:15000]  # 截断以控制 token
            
            prompt = f"""请为这位群友写一份温暖的人物画像，让 TA 感受到被看见和被喜爱。

## 用户发言样本：
{msg_text}

## 任务：
1. **温暖标签**：给 TA 一个充满喜爱的称号（如：深夜暖心小精灵、群里的小太阳、永远在线的温柔），4-6字。
2. **画像描述**：用温暖的一句话描述 TA 在群里的样子，像是在向朋友介绍这个很特别的人。
3. **MBTI 猜想**：根据发言风格猜测 TA 的 MBTI 人格（如 ENFP），并用括号简述为什么，语气要充满欣赏。

## 写作指南：
- 想象你在向新朋友介绍"我们群里的宝藏朋友"
- 让 TA 读到时会心一笑，感受到被珍视
- 发现 TA 的闪光点，用温暖的方式表达

## 输出格式（JSON）：
{{
  "persona": "...",
  "description": "...",
  "mbti": "..."
}}"""

            try:
                content = self._call_api(
                    messages=[
                        {"role": "system", "content": "你是一位充满欣赏的人物画像师，擅长发现每个人的闪光点，用温暖的文字让每个人都感受到被看见的喜悦。只输出JSON。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                
                # 解析 JSON
                import json
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    res = json.loads(json_match.group())
                    profiles.append({
                        'user': user,
                        'persona': res.get('persona', '神秘群友'),
                        'description': res.get('description', '暂无描述'),
                        'mbti': res.get('mbti', 'UNKNOWN')
                    })
                else:
                     profiles.append({'user': user, 'persona': '神秘群友', 'description': content[:20], 'mbti': 'UNKNOWN'})
                     
            except Exception as e:
                logger.warning(f"分析用户 {user} 失败: {e}")
                profiles.append({'user': user, 'persona': '低调路人', 'description': '保持神秘', 'mbti': 'ISTJ'})
        
        return profiles

    def _mock_user_profiles_mbti(self, users: List[str]) -> List[dict]:
        """Mock 用户画像数据"""
        roles = [
            # 分析师型 (NT)
            ('群里的智囊团', 'INTJ', '总能给出深思熟虑的建议'),
            ('好奇宝宝', 'INTP', '对一切新鲜事物充满探索欲'),
            ('点子大王', 'ENTP', '脑洞大开，创意无限'),
            ('天生领航者', 'ENTJ', '带大家一起冲，从不掉队'),
            # 外交官型 (NF)
            ('深夜暖心小精灵', 'INFJ', '总在最需要的时候出现'),
            ('温柔的理想家', 'INFP', '用文字传递温暖和希望'),
            ('群里的小太阳', 'ENFP', '随时都能点亮大家的一天'),
            ('暖场担当', 'ENFJ', '让每个人都感到被欢迎'),
            # 守护者型 (SJ)
            ('默默守护者', 'ISTJ', '靠谱得让人安心'),
            ('温柔守护者', 'ISFJ', '悄悄关心着每一个人'),
            ('秩序维护员', 'ESTJ', '群里有事，第一个站出来'),
            ('热心大使', 'ESFJ', '总在张罗聚会和活动'),
            # 探险家型 (SP)
            ('神秘冷酷侠', 'ISTP', '话不多但句句在点上'),
            ('浪漫生活家', 'ISFP', '把日常过成诗'),
            ('快乐制造机', 'ESTP', '有 TA 的地方就有笑声'),
            ('气氛担当', 'ESFP', '群里的开心果，永远活力满满'),
        ]
        import random
        results = []
        for i, user in enumerate(users):
            role = roles[i % len(roles)]
            results.append({
                'user': user,
                'persona': role[0],
                'description': f"我们群里的{role[0]}，{role[2]}，有 TA 的地方就充满温暖",
                'mbti': role[1]
            })
        return results

    def _mock_topic_memory(self, month_info: dict) -> str:
        """Mock 模式：返回默认话题回忆。"""
        month_name = month_info.get('month_name', '这个月')
        stats = month_info.get('stats', {})
        count = stats.get('total_messages', 0)
        
        templates = [
            f"这个月，群里的 {count} 条消息里，藏着无数个让人会心一笑的瞬间——有人在发更早问候，有人在深夜陪聊，这就是我们的日常，平凡却温暖。",
            f"这个月，我们用 {count} 条消息记录着彼此的生活。虽然隔着屏幕，但友情的温度一直在线，从未缺席。",
            f"这个月，{count} 条消息串起了无数个温馨时刻——一起吐槽、一起大笑、一起加油打气，这就是我们。",
        ]
        import random
        return random.choice(templates)


# 使用示例
if __name__ == '__main__':
    import sys
    from data_loader import load_chat_data
    from stats_engine import calculate_stats
    
    if len(sys.argv) < 2:
        print("用法: python ai_analyzer.py <json_file_path>")
        print("\n注意: 需要配置 .env 文件中的 LLM_API_KEY")
        print("如果没有配置，将使用 Mock 模式返回模拟数据")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        df, session = load_chat_data(file_path)
        stats = calculate_stats(df)
        
        analyzer = AIAnalyzer()
        print(f"Mock 模式: {analyzer.mock_mode}")
        
        result = analyzer.analyze(df, stats['top_users'])
        
        print("\n=== AI 分析结果 ===")
        print(result['raw_content'])
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
