"""
weekly_analyzer.py - 周度分析模块

提供按周聚合的群聊数据分析功能。
"""

from typing import List, Dict, Any
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd


class WeeklyAnalyzer:
    """周度分析器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化周度分析器。
        
        参数:
            df: 包含 timestamp, user, content, date 列的 DataFrame
        """
        self.df = df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """预处理数据，添加周次信息"""
        # 添加年份和周次
        self.df['year_week'] = self.df['timestamp'].dt.strftime('%Y-W%W')
        self.df['week_start'] = self.df['timestamp'].dt.to_period('W').dt.start_time
    
    def analyze(self) -> List[Dict[str, Any]]:
        """
        执行周度分析。
        
        返回:
            [{'week': '2024-W01', 'week_start': '2024-01-01', 'stats': {...}, 'highlights': {...}}, ...]
        """
        results = []
        
        for year_week, group in self.df.groupby('year_week'):
            week_data = {
                'week': year_week,
                'week_start': group['week_start'].iloc[0].strftime('%Y-%m-%d'),
                'week_end': (group['week_start'].iloc[0] + timedelta(days=6)).strftime('%Y-%m-%d'),
                'stats': self._get_week_stats(group),
                'highlights': self._get_week_highlights(group),
                'hot_topics': self._get_hot_topics(group),
                'active_users': self._get_active_users(group),
            }
            results.append(week_data)
        
        # 按周次排序
        results.sort(key=lambda x: x['week'])
        return results
    
    def _get_week_stats(self, group: pd.DataFrame) -> Dict[str, Any]:
        """获取周统计数据"""
        return {
            'total_messages': len(group),
            'active_users': group['user'].nunique(),
            'avg_per_day': round(len(group) / 7, 1),
            'peak_day': group.groupby('date').size().idxmax() if not group.empty else None,
            'peak_hour': int(group['hour'].mode().iloc[0]) if not group['hour'].mode().empty else 0,
        }
    
    def _get_week_highlights(self, group: pd.DataFrame) -> Dict[str, Any]:
        """获取周高光时刻"""
        if group.empty:
            return {'top_talker': None, 'longest_message': None}
        
        # 话痨担当
        top_talker = group['user'].value_counts().head(1)
        top_talker_info = {
            'user': top_talker.index[0] if not top_talker.empty else None,
            'count': int(top_talker.iloc[0]) if not top_talker.empty else 0
        }
        
        # 最长消息
        longest_idx = group['content'].str.len().idxmax()
        longest_msg = group.loc[longest_idx]
        
        return {
            'top_talker': top_talker_info,
            'longest_message': {
                'user': longest_msg['user'],
                'content': longest_msg['content'][:100],
                'length': len(longest_msg['content'])
            }
        }
    
    def _get_hot_topics(self, group: pd.DataFrame, top_n: int = 5) -> List[str]:
        """提取本周热门话题关键词"""
        import jieba
        
        # 合并所有消息
        all_text = ' '.join(group['content'].astype(str).tolist())
        
        # 分词
        words = jieba.cut(all_text, cut_all=False)
        
        # 过滤短词和停用词
        filtered = [w for w in words if len(w) >= 2 and not w.isdigit()]
        
        # 统计频次
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(top_n)]
    
    def _get_active_users(self, group: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """获取本周活跃用户"""
        user_counts = group['user'].value_counts().head(top_n)
        total = len(group)
        
        return [
            {
                'user': user,
                'count': int(count),
                'percentage': round(count / total * 100, 1)
            }
            for user, count in user_counts.items()
        ]


    def get_weekly_samples(self, max_per_week: int = 1000) -> List[Dict[str, Any]]:
        """
        获取每周的消息样本用于 AI 分析。
        
        参数:
            max_per_week: 每周最大样本数
            
        返回:
            [{'week': '2024-W01', 'messages': [...]}, ...]
        """
        samples = []
        for year_week, group in self.df.groupby('year_week'):
            # 如果消息数超过限制，优先保留"热门"消息（这里简单用长度和时间分布做均匀采样）
            # 也可以结合 stats_engine 的逻辑筛选回复多的
            if len(group) > max_per_week:
                # 简单均匀采样，保持时间分布
                step = len(group) // max_per_week
                week_msgs = group.iloc[::step].head(max_per_week)
            else:
                week_msgs = group
            
            # 格式化消息
            msgs = []
            for _, row in week_msgs.iterrows():
                msgs.append({
                    'user': row['user'],
                    'content': row['content'],
                    'date': row['date'],
                    'timestamp': row['timestamp']
                })
            
            samples.append({
                'week': year_week,
                'week_start': group['week_start'].iloc[0].strftime('%Y-%m-%d'),
                'messages': msgs
            })
            
        return samples


def get_weekly_analysis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    便捷函数：执行周度分析。
    
    参数:
        df: 消息 DataFrame
        
    返回:
        周度分析结果列表
    """
    analyzer = WeeklyAnalyzer(df)
    return analyzer.analyze()

def get_weekly_samples_for_ai(df: pd.DataFrame, max_per_week: int = 1000) -> List[Dict[str, Any]]:
    """
    便捷函数：获取周度消息样本。
    """
    analyzer = WeeklyAnalyzer(df)
    return analyzer.get_weekly_samples(max_per_week)
