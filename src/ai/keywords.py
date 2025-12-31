"""
ai/keywords.py - 关键词优化模块

包含关键词优化和筛选功能。
"""

import json
import logging
import re
from typing import List

logger = logging.getLogger(__name__)


class KeywordsMixin:
    """
    关键词混入类，提供关键词优化方法。
    
    需要与 AIAnalyzerBase 一起使用。
    """
    
    def refine_keywords(self, keywords: List[dict], max_keywords: int = 15) -> List[dict]:
        """
        使用 AI 优化关键词列表：
        1. 过滤无意义/敏感词汇
        2. 合并同类/近义词
        3. 选出最能代表群聊特色的关键词
        
        返回:
            [{'keyword': '...', 'category': '...', 'description': '...'}, ...]
        """
        if not keywords:
            return []
        
        if self.mock_mode:
            return self._mock_keywords(keywords[:max_keywords])
        
        # 提取原始关键词文本
        raw_keywords = [k['word'] if isinstance(k, dict) else k for k in keywords]
        
        prompt = f"""你是一位群聊年报制作师，以下是群聊词频统计前 {len(raw_keywords)} 的关键词：
{', '.join(raw_keywords[:50])}

请帮我从中筛选并优化出最能代表这个群聊温馨氛围的 {max_keywords} 个关键词。

## 筛选原则：
1. **保留温暖的**：体现群友之间友情、陪伴、互助的词
2. **保留有趣的**：群里的梗、口头禅、专属用语
3. **保留有记忆的**：能让群友一看就想起某些美好时刻的词
4. **合并同类词**：如"吃饭吃啥吃什么"合并为"吃什么"

## 过滤原则：
1. **删除敏感词**：任何负面、不雅、政治相关的词
2. **删除无意义词**：太泛化的词（如"东西""事情""问题"）
3. **删除日常口头语**：太普通的词（如"哈哈""好的""嗯嗯"）

## ❗❗ 输出规则：
1. **只输出纯 JSON**，不要任何额外文字
2. **不要使用 markdown 代码块**

## 输出格式（JSON 数组）：
[
  {{"word": "关键词", "category": "类别", "reason": "为什么保留"}},
  ...
]

类别可选：【日常互动】【美食时刻】【工作吐槽】【游戏电竞】【情感交流】【群内梗】【其他】"""

        try:
            content = self._call_api(
                messages=[
                    {"role": "system", "content": "你是一位擅长发现群聊温情的编辑，帮助筛选最有意义的关键词。只输出JSON。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )
            
            # 解析 JSON
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                result = json.loads(json_match.group())
                return result[:max_keywords]
            else:
                return self._mock_keywords(raw_keywords[:max_keywords])
                
        except Exception as e:
            logger.warning(f"关键词优化失败: {e}")
            return self._mock_keywords(raw_keywords[:max_keywords])
    
    def _mock_keywords(self, keywords: List) -> List[dict]:
        """Mock 关键词数据"""
        results = []
        for kw in keywords:
            word = kw['word'] if isinstance(kw, dict) else kw
            results.append({
                'word': word,
                'category': '日常互动',
                'reason': '群里常用词'
            })
        return results
