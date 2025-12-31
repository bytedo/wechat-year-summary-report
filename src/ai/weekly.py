"""
ai/weekly.py - 周度分析模块

包含周度批次分析和年度总结生成功能。
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class WeeklyAnalysisMixin:
    """
    周度分析混入类，提供周度分析相关方法。
    
    需要与 AIAnalyzerBase 一起使用。
    """
    
    def analyze_weekly_batches(
        self, 
        weekly_samples: list, 
        use_cache: bool = True
    ) -> Tuple[str, dict]:
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
        cache_dir = Path(__file__).parent.parent.parent / ".cache"
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
