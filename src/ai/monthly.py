"""
ai/monthly.py - 月度分析模块

包含月度话题回忆生成功能。
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


class MonthlyAnalysisMixin:
    """
    月度分析混入类，提供月度话题回忆相关方法。
    
    需要与 AIAnalyzerBase 一起使用。
    """
    
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

## ❗❗ 输出规则（必须严格遵守）：
1. **只输出纯 JSON**，不要任何额外文字、注释或解释
2. **不要使用 markdown 代码块**（不要用 ```json ```）
3. **不要输出多个 JSON 对象**，只输出一个完整的 JSON

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
    
    def _mock_topic_memory(self, month_info: dict) -> str:
        """Mock 模式：返回默认话题回忆。"""
        import random
        month_name = month_info.get('month_name', '这个月')
        stats = month_info.get('stats', {})
        count = stats.get('total_messages', 0)
        
        templates = [
            f"这个月，群里的 {count} 条消息里，藏着无数个让人会心一笑的瞬间——有人在发更早问候，有人在深夜陪聊，这就是我们的日常，平凡却温暖。",
            f"这个月，我们用 {count} 条消息记录着彼此的生活。虽然隔着屏幕，但友情的温度一直在线，从未缺席。",
            f"这个月，{count} 条消息串起了无数个温馨时刻——一起吐槽、一起大笑、一起加油打气，这就是我们。",
        ]
        return random.choice(templates)
    
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
                    if month_key in week_key:
                         relevant_weeks.append(summary)
                    else:
                        y, w = week_key.split('-W')
                        week_start = datetime.datetime.strptime(f'{y}-W{w}-1', "%Y-W%W-%w")
                        if week_start.strftime('%Y-%m') == month_key:
                            relevant_weeks.append(summary)
                except:
                    pass
            
            if not relevant_weeks:
                if self.mock_mode:
                    memory = self._mock_topic_memory(month_info)
                    topics = []
                else:
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

请基于这些回忆，写一段 80-120 字的**月度温馨回忆**。
同时提取 3 个最能触动人心的话题标签。

写作指南：
- 用"这个月"开头，让全文像是在给老朋友写信
- 让群友读到时能想起那些快乐时光
- 文字要温暖，像冬日里的一杯热茶

## ❗❗ 输出规则（必须严格遵守）：
1. **只输出纯 JSON**，不要任何额外文字、注释或解释
2. **不要使用 markdown 代码块**（不要用 ```json ```）
3. **不要输出多个 JSON 对象**，只输出一个完整的 JSON

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
                    # 使用非贪婪匹配并逐个验证，只取第一个有效 JSON
                    json_candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
                    memory = ''
                    topics = []
                    parsed_ok = False
                    for candidate in json_candidates:
                        try:
                            res = json.loads(candidate)
                            if 'memory' in res or 'topics' in res:
                                memory = res.get('memory', '')
                                topics = res.get('topics', [])
                                parsed_ok = True
                                break
                        except:
                            continue
                    # 如果没有解析到有效 JSON，使用原始内容作为回忆
                    if not parsed_ok and not memory:
                        memory = content.strip() if content else ''
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
