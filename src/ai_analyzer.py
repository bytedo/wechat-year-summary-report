"""
ai_analyzer.py - AI 分析代理模块

使用 LLM 对聊天数据进行深度分析，包括话题总结、用户画像等。
支持 Mock 模式，当没有 API Key 时返回模拟数据。
"""

import os
import random
import re
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class AIAnalyzer:
    """
    AI 分析器，支持 OpenAI 兼容接口（DeepSeek/Moonshot 等）。
    """
    
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
                print("警告: openai 库未安装，启用 Mock 模式")
                self.mock_mode = True
    
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个温暖有趣的群聊回忆官，擅长发现朋友间的温馨时刻。你的报告总是充满正能量，让每个人都感到被重视和喜爱。绝对不要输出任何负面、敏感或可能引起争议的内容。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # 兼容不同的响应格式
            content = self._extract_content(response)
            if not content:
                raise ValueError("无法从响应中提取内容")
            
            return self._parse_response(content, top_users)
            
        except Exception as e:
            print(f"AI 分析失败: {e}，使用 Mock 数据")
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
        
        prompt = f"""请分析以下微信群聊记录，生成一份温馨有趣的年度群聊报告。

## 群聊消息样本（按月整理）:
{all_msg_text}

## 群成员列表: {', '.join(all_user_names)}
## 包含月份: {', '.join(months_list)}

---

## 分析任务

请按以下格式输出 Markdown 报告：

### 🎯 群聊年度关键词
用 3 个**积极正向**的关键词总结这个群的氛围（如：欢乐、互助、温暖等）。

### 👥 群友画像墙
为**每一位**群成员生成一句话正向画像（包括：{', '.join(all_user_names)}）。
- 用可爱/温暖/幽默的语气
- 突出每个人的闪光点和贡献
- 给每人一个有趣的"称号"（如：表情包大师、暖心担当、气氛组组长等）

### 📅 月度话题回顾
请为**每一个月**（{', '.join(months_list)}）都生成详细的话题回顾，格式如下：

**📌 1月**
- 🔹 话题1：简短描述（1-2句话说明大家聊了什么）
- 🔹 话题2：简短描述
- 🔹 话题3：简短描述

**📌 2月**
...（以此类推，每个月都要有 2-4 个话题）

**⚠️ 每个月都必须有内容，不能跳过任何月份！**

### ✨ 年度温馨时刻
挑选 2-3 个**温馨有趣**的群聊片段或互动场景。

---

## ⚠️ 重要规则
1. **只输出积极正向的内容**，展现群友间的友谊和欢乐
2. **避开任何敏感话题**（如感情问题、个人隐私、争吵等）
3. **语气要温暖有趣**，像是老朋友之间的年终暖心回顾
4. **每个人都要有画像**，不能遗漏任何群成员
5. **每个月都要有话题回顾**，不能跳过任何月份
6. 如果消息中有不适合公开的内容，用"日常趣事"等概括性描述替代
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

**欢乐** | **吐槽** | **互帮互助**

这是一个充满欢声笑语的群聊，群友们在这里分享生活、吐槽工作、互相帮助。

---

## 👥 活跃成员画像

{"" if not user_names else f'''
### 🥇 {user_names[0] if len(user_names) > 0 else "神秘人"}
群内话痨担当，每天准时报到，是群里的气氛组组长。发言风格幽默风趣，经常能把大家逗乐。

### 🥈 {user_names[1] if len(user_names) > 1 else "隐藏大佬"}
深夜冲浪选手，擅长在凌晨发表人生感悟。偶尔冒泡，句句经典。

### 🥉 {user_names[2] if len(user_names) > 2 else "潜水达人"}
表情包大师，总能在关键时刻甩出完美的表情包救场。
'''}

---

## 🔥 热门话题

1. **日常打卡** - 早安晚安问候从未断过
2. **美食分享** - 深夜放毒，减肥路上的绊脚石
3. **工作吐槽** - 打工人的心酸，只有群友懂
4. **游戏开黑** - 一起上分，一起掉分
5. **生活琐事** - 家长里短，温暖日常

---

## ✨ 群聊名场面

> "今年最难忘的一刻，大概是某位群友凌晨三点还在群里发消息，结果第二天上班迟到被老板骂了..."

群聊虽然话不多，但每一句都是感情。这一年，感谢有你们的陪伴！🎉

---

*（以上分析由 AI 自动生成，如有雷同，纯属巧合）*
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个群聊话题分类专家。请根据提供的消息样本，为每个话题组起一个简短有趣的名字（2-6个字）。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # 兼容不同的响应格式
            content = self._extract_content(response)
            if not content:
                raise ValueError("无法从响应中提取内容")
            
            return self._parse_cluster_names(content, cluster_representatives)
            
        except Exception as e:
            print(f"   ⚠️ 话题命名失败: {e}，使用默认名称")
            return self._mock_summarize_clusters(cluster_representatives)
    
    def _build_cluster_prompt(self, cluster_representatives: dict) -> str:
        """构建聚类命名的提示词。"""
        lines = ["以下是通过向量算法自动聚类的群聊消息组，请为每组起一个简短有趣的名字。\n"]
        
        for cluster_id, messages in cluster_representatives.items():
            if not messages:
                continue
            
            lines.append(f"## 分组 {cluster_id}")
            for msg in messages[:5]:  # 每组最多展示5条
                content = msg['content'][:50]  # 截断长内容
                lines.append(f"- {msg['user']}: {content}")
            lines.append("")
        
        lines.append("""
请返回 JSON 格式的结果，例如：
{"0": "午餐拼单", "1": "技术交流", "2": "摸鱼时间"}

注意：
- 名字要简短（2-6个字）
- 可以幽默有趣
- 要能反映该组消息的主题""")
        
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
        
        prompt = f"""你是一个专业的群聊"金句挖掘师"。请从以下群聊消息中，精选出最值得记住的"年度金句"。

## 候选消息：
{msg_text}

## 任务要求：
请挑选 **8-12 条** 最精彩的金句，分为以下类别：

### 类别说明：
- 😂 **搞笑担当**：最让人捧腹的话
- 💡 **金玉良言**：有道理、有深度的话
- 🔥 **名场面**：引发热烈讨论的话
- 💖 **暖心时刻**：温暖人心的话
- 🎭 **神回复**：神级回复、反转、吐槽

## 输出格式（JSON 数组）：
[
  {{"user": "用户名", "content": "完整金句内容", "reason": "入选理由（10字内）", "category": "类别标签"}}
]

## 注意：
- 优先选择有趣、正向、有创意的内容
- 每个类别至少选 1 条
- 避免任何敏感或负面内容
- 金句内容保持原样，不要修改"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的群聊金句挖掘师，擅长发现群里最精彩、最有趣、最温暖的话语。你的审美很好，善于抓住重点。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            content = self._extract_content(response)
            return self._parse_golden_quotes(content, candidates)
            
        except Exception as e:
            print(f"   ⚠️ 金句甄选失败: {e}，使用默认")
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
        
        prompt = f"""这是群里消息最多的一天（{peak_day_data['date']}，共 {peak_day_data['count']} 条消息）的部分消息：

{msg_text}

请用 50 字左右，温馨有趣地描述"那一天大家都在聊什么"。
开头用"那一天，"作为引子。不要提及任何敏感内容。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个温馨的群聊回忆官。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            content = self._extract_content(response)
            return content.strip() if content else self._mock_peak_day_summary(peak_day_data)
            
        except Exception as e:
            print(f"   ⚠️ 巅峰日摘要失败: {e}")
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
        
        prompt = f"""请分析群聊的 {month_name} 消息，提取具体话题。

## 本月消息样本：
{msg_text}

## 任务：
1. 提取 3-5 个本月**具体发生的话题/事件**（不是单词，而是具体的事情）
2. 写一段 80-100 字的月度回忆

## 输出格式（严格JSON）：
{{
  "topics": [
    {{"title": "话题标题（4-8字）", "desc": "一句话描述"}},
    ...
  ],
  "memory": "这个月，..."
}}

## 示例：
{{
  "topics": [
    {{"title": "小明生日聚会", "desc": "大家给小明庆生，热闹非凡"}},
    {{"title": "年底加班吐槽", "desc": "集体吐槽加班，互相打气"}},
    {{"title": "跨年计划讨论", "desc": "商量去哪跨年"}}
  ],
  "memory": "这个月，大家一起给小明庆祝了生日，还集体吐槽了年底加班的辛苦..."
}}

## 要求：
- 话题要**具体**，不要是"日常闲聊"这样笼统的
- 从消息内容中推断具体事件
- 内容正向温馨，避免敏感话题"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个群聊分析师，擅长从聊天记录中提取具体的话题事件。只输出JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            content = self._extract_content(response)
            
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
            print(f"   ⚠️ {month_name}话题提取失败: {e}")
            return {
                'topics': [],
                'memory': self._mock_topic_memory(month_info)
            }
    
    def _mock_topic_memory(self, month_info: dict) -> str:
        """Mock 模式：返回默认话题回忆。"""
        month_name = month_info.get('month_name', '这个月')
        stats = month_info.get('stats', {})
        count = stats.get('total_messages', 0)
        
        templates = [
            f"这个月，群里产生了 {count} 条消息，大家聊得热火朝天，欢声笑语不断！",
            f"这个月，{count} 条消息记录着我们的日常，每一条都是友谊的见证。",
            f"这个月，我们用 {count} 条消息填满了这个小天地，快乐一直在线！",
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
