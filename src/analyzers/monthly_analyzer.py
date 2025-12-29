"""
monthly_analyzer.py - 月度分析模块

提供按月聚合的群聊数据分析功能，支持详细的月度话题和人物分析。
"""

from typing import List, Dict, Any
from collections import Counter
from datetime import datetime

import pandas as pd


class MonthlyAnalyzer:
    """月度分析器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化月度分析器。
        
        参数:
            df: 包含 timestamp, user, content, date 列的 DataFrame
        """
        self.df = df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """预处理数据，添加月份信息"""
        self.df['year_month'] = self.df['date'].str[:7]  # YYYY-MM
        self.df['month_num'] = self.df['timestamp'].dt.month
        self.df['month_name'] = self.df['month_num'].apply(lambda x: f"{x}月")
    
    def analyze(self) -> List[Dict[str, Any]]:
        """
        执行月度分析。
        
        返回:
            [{'month': '2024-01', 'month_name': '1月', 'stats': {...}, 'topics': [...], ...}, ...]
        """
        results = []
        
        for year_month, group in self.df.groupby('year_month'):
            month_num = int(year_month.split('-')[1])
            
            month_data = {
                'month': year_month,
                'month_num': month_num,
                'month_name': f"{month_num}月",
                'year': int(year_month.split('-')[0]),
                'stats': self._get_month_stats(group),
                'topics': self._extract_topics(group),
                'highlights': self._get_month_highlights(group),
                'active_users': self._get_active_users(group),
                'sample_messages': self._get_sample_messages(group),
                'emoji_stats': self._get_emoji_stats(group),
            }
            results.append(month_data)
        
        # 按月份排序
        results.sort(key=lambda x: x['month'])
        return results
    
    def _get_month_stats(self, group: pd.DataFrame) -> Dict[str, Any]:
        """获取月统计数据"""
        days_in_month = group['date'].nunique()
        
        return {
            'total_messages': len(group),
            'active_users': group['user'].nunique(),
            'active_days': days_in_month,
            'avg_per_day': round(len(group) / max(days_in_month, 1), 1),
            'peak_day': self._get_peak_day(group),
            'most_active_hour': int(group['hour'].mode().iloc[0]) if not group['hour'].mode().empty else 0,
            'total_chars': group['content'].str.len().sum(),
        }
    
    def _get_peak_day(self, group: pd.DataFrame) -> Dict[str, Any]:
        """获取月内最活跃的一天"""
        daily_counts = group.groupby('date').size()
        if daily_counts.empty:
            return {'date': None, 'count': 0}
        
        peak_date = daily_counts.idxmax()
        return {
            'date': peak_date,
            'count': int(daily_counts.max())
        }
    
    def _extract_topics(self, group: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        提取本月热门话题。
        
        返回:
            [{'keyword': '关键词', 'count': 频次, 'sample': '示例消息'}, ...]
        """
        import jieba
        
        # 合并所有消息
        all_text = ' '.join(group['content'].astype(str).tolist())
        
        # 分词并过滤
        words = jieba.cut(all_text, cut_all=False)
        filtered = [w.strip() for w in words if len(w.strip()) >= 2 and not w.strip().isdigit()]
        
        # 统计频次
        counter = Counter(filtered)
        top_words = counter.most_common(top_n)
        
        # 为每个关键词找一条示例消息
        topics = []
        for word, count in top_words:
            sample_msg = group[group['content'].str.contains(word, na=False)].head(1)
            sample = sample_msg['content'].iloc[0][:50] if not sample_msg.empty else ''
            
            topics.append({
                'keyword': word,
                'count': count,
                'sample': sample
            })
        
        return topics
    
    def _get_month_highlights(self, group: pd.DataFrame) -> Dict[str, Any]:
        """获取月度高光时刻"""
        if group.empty:
            return {}
        
        # 月度之星（消息最多的用户）
        top_talker = group['user'].value_counts().head(1)
        
        # 最长消息
        longest_idx = group['content'].str.len().idxmax()
        longest_msg = group.loc[longest_idx]
        
        # 最活跃的一天
        daily_counts = group.groupby('date').size()
        busiest_day = daily_counts.idxmax() if not daily_counts.empty else None
        
        return {
            'monthly_star': {
                'user': top_talker.index[0] if not top_talker.empty else None,
                'count': int(top_talker.iloc[0]) if not top_talker.empty else 0,
                'percentage': round(top_talker.iloc[0] / len(group) * 100, 1) if not top_talker.empty else 0
            },
            'longest_message': {
                'user': longest_msg['user'],
                'content': longest_msg['content'][:80] + '...' if len(longest_msg['content']) > 80 else longest_msg['content'],
                'date': longest_msg['date']
            },
            'busiest_day': {
                'date': busiest_day,
                'count': int(daily_counts.max()) if not daily_counts.empty else 0
            }
        }
    
    def _get_active_users(self, group: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
        """获取本月活跃用户排行"""
        user_counts = group['user'].value_counts().head(top_n)
        total = len(group)
        
        return [
            {
                'rank': i + 1,
                'user': user,
                'count': int(count),
                'percentage': round(count / total * 100, 1)
            }
            for i, (user, count) in enumerate(user_counts.items())
        ]
    
    def _get_sample_messages(self, group: pd.DataFrame, n: int = 20) -> List[Dict[str, Any]]:
        """
        获取本月样本消息（用于 AI 分析）。
        
        优化策略：优先选取消息密集的时间段（热点日），更容易捕捉到具体话题。
        """
        if len(group) <= n:
            sample = group
        else:
            # 策略：选取消息最密集的几天
            daily_counts = group.groupby('date').size().sort_values(ascending=False)
            
            # 选取消息最多的 3-5 天
            top_days = daily_counts.head(5).index.tolist()
            
            # 从这些热点日获取消息
            hot_messages = group[group['date'].isin(top_days)]
            
            if len(hot_messages) >= n:
                # 从热点日中均匀采样
                messages_per_day = max(n // len(top_days), 3)
                sampled = []
                
                for day in top_days:
                    day_msgs = hot_messages[hot_messages['date'] == day]
                    # 选取每天中间时段的消息（更可能是话题高潮）
                    if len(day_msgs) > messages_per_day:
                        # 按时间排序，选取中间段
                        day_msgs_sorted = day_msgs.sort_values('timestamp')
                        mid_start = len(day_msgs_sorted) // 4
                        mid_end = mid_start + messages_per_day
                        sampled.append(day_msgs_sorted.iloc[mid_start:mid_end])
                    else:
                        sampled.append(day_msgs)
                
                sample = pd.concat(sampled).head(n)
            else:
                # 热点日消息不够，补充随机采样
                remaining = n - len(hot_messages)
                other_msgs = group[~group['date'].isin(top_days)]
                if len(other_msgs) > remaining:
                    extra = other_msgs.sample(n=remaining)
                else:
                    extra = other_msgs
                sample = pd.concat([hot_messages, extra]).head(n)
        
        # 按时间排序返回
        sample = sample.sort_values('timestamp')
        
        return [
            {
                'user': row['user'],
                'content': row['content'][:120],  # 增加内容长度
                'date': row['date'],
                'time': row['timestamp'].strftime('%H:%M') if hasattr(row['timestamp'], 'strftime') else ''
            }
            for _, row in sample.iterrows()
        ]
    
    def _get_emoji_stats(self, group: pd.DataFrame) -> Dict[str, Any]:
        """统计表情使用情况"""
        import re
        
        # 匹配 [表情名] 格式
        emoji_pattern = r'\[([^\]]+)\]'
        all_content = ' '.join(group['content'].astype(str).tolist())
        emojis = re.findall(emoji_pattern, all_content)
        
        if not emojis:
            return {'total': 0, 'top_emojis': []}
        
        counter = Counter(emojis)
        top_emojis = [
            {'emoji': f"[{emoji}]", 'count': count}
            for emoji, count in counter.most_common(5)
        ]
        
        return {
            'total': len(emojis),
            'top_emojis': top_emojis
        }


def get_monthly_analysis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    便捷函数：执行月度分析。
    
    参数:
        df: 消息 DataFrame
        
    返回:
        月度分析结果列表
    """
    analyzer = MonthlyAnalyzer(df)
    return analyzer.analyze()
