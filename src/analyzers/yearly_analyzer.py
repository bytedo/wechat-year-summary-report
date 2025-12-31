"""
yearly_analyzer.py - 年度分析模块

提供年度汇总分析，生成年度报告所需的所有数据。
"""

from typing import List, Dict, Any
from collections import Counter
from datetime import datetime

import pandas as pd


class YearlyAnalyzer:
    """年度分析器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化年度分析器。
        
        参数:
            df: 包含 timestamp, user, content, date 列的 DataFrame
        """
        self.df = df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """预处理数据"""
        self.df['year'] = self.df['timestamp'].dt.year
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['weekday'] = self.df['timestamp'].dt.weekday
    
    def analyze(self) -> Dict[str, Any]:
        """
        执行年度分析。
        
        返回:
            {'overview': {...}, 'rankings': {...}, 'highlights': {...}, ...}
        """
        return {
            'overview': self._get_overview(),
            'rankings': self._get_rankings(),
            'highlights': self._get_highlights(),
            'timeline': self._get_timeline(),
            'keywords': self._get_yearly_keywords(),
            'fun_facts': self._get_fun_facts(),
            'user_profiles': self._get_user_profiles(),
            'charts': self._get_charts_data(),
            'quote_candidates': self._get_quote_candidates(),
        }

    def _get_quote_candidates(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取潜在的金句候选消息。
        策略：
        1. 长度适中 (10-100字)
        2. 排除纯数字/URL
        3. 优先包含标点符号或表情的消息
        """
        # 过滤
        mask = (
            (self.df['content'].str.len() >= 10) & 
            (self.df['content'].str.len() <= 100) &
            (~self.df['content'].str.contains('http', na=False)) &
            (~self.df['content'].str.contains('红包', na=False)) &
            (~self.df['content'].str.isnumeric())
        )
        candidates = self.df[mask]
        
        if candidates.empty:
            return []
            
        # 简单随机采样，或者后续可以优化为按月份均匀采样
        # 为了增加多样性，按月份分组采样
        results = []
        try:
            # 尝试每组采几个
            param_n = max(1, limit // 12)
            subset = candidates.groupby('month').apply(
                lambda x: x.sample(n=min(len(x), param_n)),
                include_groups=False
            )
            # Flatten
            if isinstance(subset, pd.DataFrame):
                samples = subset
            else:
                # pandas groupby apply return varies
                samples = subset.reset_index(level=0, drop=True)
                
            # 转换为 dict list
            for _, row in samples.iterrows():
                results.append({
                    'user': row['user'],
                    'content': row['content'],
                    'date': row['date']
                })
        except Exception as e:
            # Fallback random sample
            print(f"Sampling error: {e}")
            samples = candidates.sample(n=min(len(candidates), limit))
            for _, row in samples.iterrows():
                results.append({
                    'user': row['user'],
                    'content': row['content'],
                    'date': row['date']
                })
                
        return results

    def _get_charts_data(self) -> Dict[str, Any]:
        """获取图表所需数据"""
        # 1. 24小时活跃度分布
        hour_counts = self.df['hour'].value_counts().sort_index()
        hourly_activity = [int(hour_counts.get(i, 0)) for i in range(24)]
        
        # 2. 月度活跃趋势
        month_counts = self.df['month'].value_counts().sort_index()
        monthly_activity = [int(month_counts.get(i, 0)) for i in range(1, 13)]
        
        return {
            'hourly': hourly_activity,
            'monthly': monthly_activity
        }
    
    def _get_overview(self) -> Dict[str, Any]:
        """获取年度总览"""
        return {
            'total_messages': len(self.df),
            'total_users': self.df['user'].nunique(),
            'total_days': self.df['date'].nunique(),
            'total_chars': int(self.df['content'].str.len().sum()),
            'date_start': self.df['date'].min(),
            'date_end': self.df['date'].max(),
            'avg_per_day': round(len(self.df) / max(self.df['date'].nunique(), 1), 1),
            'peak_month': self._get_peak_period('month'),
            'peak_weekday': self._get_peak_period('weekday'),
            'peak_hour': self._get_peak_period('hour'),
        }
    
    def _get_peak_period(self, period: str) -> Dict[str, Any]:
        """获取高峰期"""
        counts = self.df[period].value_counts()
        if counts.empty:
            return {'value': None, 'count': 0}
        
        peak = counts.idxmax()
        
        # 格式化显示
        if period == 'month':
            display = f"{peak}月"
        elif period == 'weekday':
            weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            display = weekdays[peak]
        elif period == 'hour':
            display = f"{peak}:00"
        else:
            display = str(peak)
        
        return {
            'value': int(peak),
            'display': display,
            'count': int(counts.max())
        }
    
    def _get_rankings(self) -> Dict[str, Any]:
        """获取各类排行榜"""
        user_counts = self.df['user'].value_counts()
        total = len(self.df)
        
        # 话痨排行榜
        top_talkers = [
            {
                'rank': i + 1,
                'user': user,
                'count': int(count),
                'percentage': round(count / total * 100, 1)
            }
            for i, (user, count) in enumerate(user_counts.head(10).items())
        ]
        
        # 潜水员排行榜（消息最少的活跃用户）
        lurkers = [
            {
                'rank': i + 1,
                'user': user,
                'count': int(count)
            }
            for i, (user, count) in enumerate(user_counts.tail(5).items())
        ]
        
        return {
            'top_talkers': top_talkers,
            'lurkers': lurkers[::-1],  # 反转，最少的在前
        }
    
    def _get_highlights(self) -> Dict[str, Any]:
        """获取年度高光时刻"""
        # 最活跃的一天
        daily_counts = self.df.groupby('date').size()
        peak_day = daily_counts.idxmax() if not daily_counts.empty else None
        peak_day_count = int(daily_counts.max()) if not daily_counts.empty else 0
        
        # 最长消息
        longest_idx = self.df['content'].str.len().idxmax()
        longest_msg = self.df.loc[longest_idx]
        
        # 最早发消息的人（每天最早）
        early_birds = self.df.groupby('date').apply(
            lambda x: x.loc[x['timestamp'].idxmin()]['user'],
            include_groups=False
        ).value_counts()
        
        # 夜猫子（23:00-03:00 发消息最多）
        night_owls = self.df[
            (self.df['hour'] >= 23) | (self.df['hour'] <= 3)
        ]['user'].value_counts()
        
        return {
            'peak_day': {
                'date': peak_day,
                'count': peak_day_count
            },
            'longest_message': {
                'user': longest_msg['user'],
                'content': longest_msg['content'][:100] + '...' if len(longest_msg['content']) > 100 else longest_msg['content'],
                'length': len(longest_msg['content']),
                'date': longest_msg['date']
            },
            'early_bird': {
                'user': early_birds.index[0] if not early_birds.empty else None,
                'days': int(early_birds.iloc[0]) if not early_birds.empty else 0
            },
            'night_owl': {
                'user': night_owls.index[0] if not night_owls.empty else None,
                'count': int(night_owls.iloc[0]) if not night_owls.empty else 0
            }
        }
    
    def _get_timeline(self) -> List[Dict[str, Any]]:
        """获取年度时间轴（每月概览）"""
        timeline = []
        
        for month, group in self.df.groupby('month'):
            month_top_user = group['user'].value_counts().head(1)
            
            timeline.append({
                'month': int(month),
                'month_name': f"{month}月",
                'message_count': len(group),
                'user_count': group['user'].nunique(),
                'top_user': month_top_user.index[0] if not month_top_user.empty else None,
            })
        
        timeline.sort(key=lambda x: x['month'])
        return timeline
    
    def _get_yearly_keywords(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """获取年度关键词"""
        import jieba
        
        # 合并所有消息
        all_text = ' '.join(self.df['content'].astype(str).tolist())
        
        # 分词
        words = jieba.cut(all_text, cut_all=False)
        filtered = [w.strip() for w in words if len(w.strip()) >= 2 and not w.strip().isdigit()]
        
        # 统计
        counter = Counter(filtered)
        
        return [
            {'word': word, 'count': count}
            for word, count in counter.most_common(top_n)
        ]
    
    def _get_fun_facts(self) -> List[Dict[str, Any]]:
        """获取年度趣味数据"""
        facts = []
        
        # 总字符数换算
        total_chars = int(self.df['content'].str.len().sum())
        facts.append({
            'icon': '📝',
            'title': '年度打字量',
            'value': f"{total_chars:,}",
            'unit': '字',
            'description': f"相当于写了 {total_chars // 500} 篇作文"
        })
        
        # 消息最多的小时
        hour_counts = self.df['hour'].value_counts()
        peak_hour = hour_counts.idxmax() if not hour_counts.empty else 0
        facts.append({
            'icon': '⏰',
            'title': '黄金聊天时段',
            'value': f"{peak_hour}:00",
            'unit': '',
            'description': f"共产生 {int(hour_counts.max()):,} 条消息"
        })
        
        # 最爱用的表情
        import re
        all_content = ' '.join(self.df['content'].astype(str).tolist())
        emojis = re.findall(r'\[([^\]]+)\]', all_content)
        if emojis:
            top_emoji = Counter(emojis).most_common(1)[0]
            facts.append({
                'icon': '😊',
                'title': '年度最爱表情',
                'value': f"[{top_emoji[0]}]",
                'unit': '',
                'description': f"共使用 {top_emoji[1]} 次"
            })
        
        # 周末 vs 工作日
        weekend_count = len(self.df[self.df['weekday'].isin([5, 6])])
        weekday_count = len(self.df[~self.df['weekday'].isin([5, 6])])
        facts.append({
            'icon': '📅',
            'title': '工作日 vs 周末',
            'value': f"{weekday_count:,} : {weekend_count:,}",
            'unit': '',
            'description': '工作日更话痨！' if weekday_count > weekend_count else '周末才是聊天时间！'
        })
        
        return facts
    
    def _get_user_profiles(self) -> List[Dict[str, Any]]:
        """为每个用户生成画像数据"""
        profiles = []
        
        for user in self.df['user'].unique():
            user_df = self.df[self.df['user'] == user]
            
            # 活跃时段
            hour_mode = user_df['hour'].mode()
            active_hour = int(hour_mode.iloc[0]) if not hour_mode.empty else 0
            
            # 平均消息长度
            avg_length = round(user_df['content'].str.len().mean(), 1)
            
            # 第一条和最后一条消息
            first_msg = user_df.sort_values('timestamp').iloc[0]
            last_msg = user_df.sort_values('timestamp').iloc[-1]
            
            profiles.append({
                'user': user,
                'total_messages': len(user_df),
                'active_days': user_df['date'].nunique(),
                'active_hour': active_hour,
                'avg_message_length': avg_length,
                'first_message': {
                    'content': first_msg['content'][:50],
                    'date': first_msg['date']
                },
                'last_message': {
                    'content': last_msg['content'][:50],
                    'date': last_msg['date']
                }
            })
        
        # 按消息数排序
        profiles.sort(key=lambda x: x['total_messages'], reverse=True)
        return profiles


def get_yearly_highlights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    便捷函数：执行年度分析。
    
    参数:
        df: 消息 DataFrame
        
    返回:
        年度分析结果
    """
    analyzer = YearlyAnalyzer(df)
    return analyzer.analyze()
