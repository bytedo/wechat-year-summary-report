"""
ai/golden_quotes.py - 金句选择模块

包含金句筛选和巅峰日摘要功能。
"""

import json
import logging
import re
from typing import List

logger = logging.getLogger(__name__)


class GoldenQuotesMixin:
    """
    金句混入类，提供金句筛选和巅峰日摘要方法。
    
    需要与 AIAnalyzerBase 一起使用。
    """
    
    def select_golden_quotes(self, candidates: List[dict], max_quotes: int = 5) -> List[dict]:
        """
        从候选消息中选出最具代表性的"金句"。
        
        参数:
            candidates: 候选消息列表 [{'user': '...', 'content': '...', 'reply_count': N}, ...]
            max_quotes: 最大返回数量
            
        返回:
            [{'user': '...', 'quote': '...', 'reason': '...', 'category': '...'}, ...]
        """
        if not candidates:
            return []
        
        if self.mock_mode:
            return self._mock_golden_quotes(candidates[:max_quotes])
        
        # 构建候选列表
        candidate_text = "\n".join([
            f"- {c['user']}: \"{c['content'][:80]}\" (引发 {c.get('reply_count', 0)} 条回复)"
            for c in candidates[:30]
        ])
        
        prompt = f"""你是群聊年度回忆的策划人，请从以下引发热烈讨论的消息中，选出 {max_quotes} 条最值得纪念的"金句"。

## 候选消息：
{candidate_text}

## 选择标准：
1. **能唤起回忆**：读到后群友们会想起那个场景
2. **有温度或有趣**：温馨、搞笑、或充满智慧
3. **有故事感**：背后可能有一段有趣的故事
4. **正能量**：传递友情、快乐、互助的信息

## ⛔ 排除标准：
- 任何负面、敏感、不雅内容
- 纯粹的口水话或无意义内容

## ❗❗ 输出规则：
1. **只输出纯 JSON**，不要任何额外文字
2. **不要使用 markdown 代码块**

## 输出格式（JSON 数组）：
[
  {{
    "user": "发言者",
    "quote": "金句内容（可适当精炼）",
    "reason": "为什么选这句（20字内）",
    "category": "类别"
  }},
  ...
]

类别可选：【暖心时刻】【爆笑名场面】【智慧金句】【群内梗】【年度名言】"""

        try:
            content = self._call_api(
                messages=[
                    {"role": "system", "content": "你是一位擅长发现群聊亮点的策划人。只输出JSON。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=600
            )
            
            # 解析 JSON
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                result = json.loads(json_match.group())
                return result[:max_quotes]
            else:
                return self._mock_golden_quotes(candidates[:max_quotes])
                
        except Exception as e:
            logger.warning(f"金句选择失败: {e}")
            return self._mock_golden_quotes(candidates[:max_quotes])
    
    def _mock_golden_quotes(self, candidates: List[dict]) -> List[dict]:
        """Mock 金句数据"""
        results = []
        categories = ['暖心时刻', '爆笑名场面', '智慧金句', '群内梗', '年度名言']
        for i, c in enumerate(candidates):
            results.append({
                'user': c.get('user', '群友'),
                'quote': c.get('content', '...')[:50],
                'reason': '引发热烈讨论',
                'category': categories[i % len(categories)]
            })
        return results
    
    def summarize_peak_day(self, peak_day_data: dict) -> str:
        """
        为巅峰日生成文字摘要。
        
        参数:
            peak_day_data: {'date': '...', 'count': N, 'sample_messages': [...]}
            
        返回:
            温馨的巅峰日描述文字
        """
        if not peak_day_data or not peak_day_data.get('date'):
            return ""
        
        if self.mock_mode:
            return self._mock_peak_day_summary(peak_day_data)
        
        date = peak_day_data.get('date', '那天')
        count = peak_day_data.get('count', 0)
        samples = peak_day_data.get('sample_messages', [])
        
        sample_text = "\n".join([
            f"- {m['user']}: {m['content'][:50]}"
            for m in samples[:10]
        ])
        
        prompt = f"""这是群聊历史上最热闹的一天：{date}，创下了 {count} 条消息的纪录！

## 当天消息样本：
{sample_text}

请用 50-80 字，用温暖怀旧的笔触描述这一天发生了什么，让群友们读到时能回忆起那天的快乐。

## 写作要求：
- 像是在给老朋友讲故事
- 语气温暖、充满回忆感
- 不要出现任何负面内容

直接输出描述文字，不要任何其他内容。"""

        try:
            content = self._call_api(
                messages=[
                    {"role": "system", "content": "你是一位温暖的回忆录作家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return content.strip()
        except Exception as e:
            logger.warning(f"巅峰日摘要生成失败: {e}")
            return self._mock_peak_day_summary(peak_day_data)
    
    def _mock_peak_day_summary(self, peak_day_data: dict) -> str:
        """Mock 模式：返回默认巅峰日摘要。"""
        return f"那一天（{peak_day_data.get('date', '某天')}），群里产生了 {peak_day_data.get('count', 0)} 条消息，欢声笑语不断，友谊在这里升温！"
