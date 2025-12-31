"""
stats_engine.py - 统计分析引擎

负责对清洗后的聊天数据进行统计分析，生成各类指标和图表数据。
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import jieba
import pandas as pd


def calculate_stats(df: pd.DataFrame, stopwords_path: Optional[str] = None) -> Dict[str, Any]:
    """
    计算所有统计指标。
    
    参数:
        df: 清洗后的消息 DataFrame
        stopwords_path: 停用词文件路径
        
    返回:
        包含所有统计结果的字典
    """
    stats = {
        'basic': _get_basic_stats(df),
        'top_users': _get_top_users(df, top_n=10),
        'daily_trend': _get_daily_trend(df),
        'hourly_distribution': _get_hourly_distribution(df),
        'word_frequency': _get_word_frequency(df, stopwords_path, top_n=50),
    }
    
    return stats


def _get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    获取基础统计指标。
    """
    return {
        'total_messages': len(df),
        'total_users': df['user'].nunique(),
        'date_start': df['date'].min(),
        'date_end': df['date'].max(),
        'total_days': df['date'].nunique(),
        'most_active_hour': int(df['hour'].mode().iloc[0]) if not df['hour'].mode().empty else 0,
        'avg_messages_per_day': round(len(df) / max(df['date'].nunique(), 1), 1),
    }


def _get_top_users(df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    获取最活跃用户排行榜（Top N）。
    
    返回:
        [{'user': '用户名', 'count': 消息数, 'percentage': 占比}, ...]
    """
    user_counts = df['user'].value_counts().head(top_n)
    total = len(df)
    
    return [
        {
            'user': user,
            'count': int(count),
            'percentage': round(count / total * 100, 1)
        }
        for user, count in user_counts.items()
    ]


def _get_daily_trend(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    获取每日消息趋势。
    
    返回:
        [{'date': '2025-01-01', 'count': 123}, ...]
    """
    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts = daily_counts.sort_values('date')
    
    return [
        {'date': row['date'], 'count': int(row['count'])}
        for _, row in daily_counts.iterrows()
    ]


def get_monthly_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    获取每月消息统计和样本消息。
    
    返回:
        [{'month': '2025-01', 'month_name': '1月', 'count': 500, 'sample_messages': [...]}, ...]
    """
    if df.empty:
        return []
    
    # 添加月份列
    df = df.copy()
    df['month'] = df['date'].str[:7]  # 'YYYY-MM'
    
    monthly_data = []
    for month, group in df.groupby('month'):
        # 提取月份数字
        month_num = int(month.split('-')[1])
        month_name = f"{month_num}月"
        
        # 随机采样消息（每月最多20条）
        sample_size = min(20, len(group))
        sample_df = group.sample(n=sample_size)
        sample_msgs = [
            {'user': row['user'], 'content': row['content'][:50]}
            for _, row in sample_df.iterrows()
        ]
        
        monthly_data.append({
            'month': month,
            'month_name': month_name,
            'count': len(group),
            'sample_messages': sample_msgs
        })
    
    # 按月份排序
    monthly_data.sort(key=lambda x: x['month'])
    return monthly_data


def _get_hourly_distribution(df: pd.DataFrame) -> List[Dict[str, int]]:
    """
    获取 24 小时活跃分布。
    
    返回:
        [{'hour': 0, 'count': 10}, {'hour': 1, 'count': 5}, ...]
    """
    # 创建完整的 0-23 小时序列
    hourly_counts = df['hour'].value_counts().reindex(range(24), fill_value=0)
    
    return [
        {'hour': hour, 'count': int(count)}
        for hour, count in hourly_counts.items()
    ]


def _load_stopwords(stopwords_path: Optional[str] = None) -> Set[str]:
    """
    加载停用词表。
    """
    stopwords = set()
    
    # 默认停用词路径
    if stopwords_path is None:
        default_path = Path(__file__).parent.parent / 'data' / 'stopwords.txt'
        if default_path.exists():
            stopwords_path = str(default_path)
    
    if stopwords_path and Path(stopwords_path).exists():
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
    
    # 添加一些基本过滤（单字符、数字、标点）
    stopwords.update(['', ' ', '\n', '\t'])
    
    return stopwords


def _get_word_frequency(df: pd.DataFrame, stopwords_path: Optional[str] = None, top_n: int = 50) -> List[Dict[str, Any]]:
    """
    使用 jieba 分词进行词频统计。
    
    返回:
        [{'word': '词语', 'count': 频次}, ...]
    """
    # 加载停用词
    stopwords = _load_stopwords(stopwords_path)
    
    # 合并所有消息内容
    all_text = ' '.join(df['content'].astype(str).tolist())
    
    # jieba 分词
    words = jieba.cut(all_text, cut_all=False)
    
    # 过滤停用词和短词
    filtered_words = [
        word.strip() 
        for word in words 
        if word.strip() 
        and word.strip() not in stopwords 
        and len(word.strip()) >= 2  # 过滤单字
        and not word.strip().isdigit()  # 过滤纯数字
    ]
    
    # 统计词频
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)
    
    return [
        {'word': word, 'count': count}
        for word, count in top_words
    ]


def calculate_memories_stats(df: pd.DataFrame, top_users: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算怀旧数据指标。
    
    参数:
        df: 清洗后的消息 DataFrame（需包含 timestamp 列）
        top_users: 活跃用户列表
        
    返回:
        包含怀旧数据的字典
    """
    return {
        'hot_messages': _find_hot_messages(df),
        'peak_day': _find_peak_day(df),
        'first_messages': _get_first_messages(df, top_users),
    }


def _find_hot_messages(df: pd.DataFrame, time_window_minutes: int = 2, min_replies: int = 5) -> List[Dict[str, Any]]:
    """
    检测引发"回复热潮"的消息。
    逻辑：一条消息发出后 N 分钟内，后续紧跟 >M 条消息。
    
    返回:
        [{'user': '发起者', 'content': '消息', 'reply_count': 回复数, 'timestamp': 时间}, ...]
    """
    if len(df) < 10:
        return []
    
    hot_messages = []
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    for i in range(len(df_sorted) - min_replies):
        current_msg = df_sorted.iloc[i]
        current_time = current_msg['timestamp']
        
        # 计算后续 time_window 分钟内的消息数
        end_time = current_time + pd.Timedelta(minutes=time_window_minutes)
        following_msgs = df_sorted[(df_sorted['timestamp'] > current_time) & 
                                    (df_sorted['timestamp'] <= end_time)]
        
        reply_count = len(following_msgs)
        
        if reply_count >= min_replies:
            # 避免重复记录（同一热潮中的消息）
            if hot_messages and (current_time - hot_messages[-1]['timestamp']).total_seconds() < 120:
                continue
            
            hot_messages.append({
                'user': current_msg['user'],
                'content': current_msg['content'][:100],  # 截断
                'reply_count': reply_count,
                'timestamp': current_time,
                'date': current_msg['date']
            })
    
    # 按回复数排序，返回 Top 20
    hot_messages.sort(key=lambda x: x['reply_count'], reverse=True)
    return hot_messages[:20]


def _find_peak_day(df: pd.DataFrame) -> Dict[str, Any]:
    """
    找出历史上消息数量最多的一天。
    
    返回:
        {'date': '2025-01-01', 'count': 500, 'sample_messages': [...]}
    """
    daily_counts = df.groupby('date').size()
    if daily_counts.empty:
        return {'date': None, 'count': 0, 'sample_messages': []}
    
    peak_date = daily_counts.idxmax()
    peak_count = int(daily_counts.max())
    
    # 获取当天的样本消息（随机抽 10 条）
    peak_df = df[df['date'] == peak_date]
    sample_size = min(10, len(peak_df))
    sample_msgs = peak_df.sample(n=sample_size)[['user', 'content']].to_dict('records')
    
    return {
        'date': peak_date,
        'count': peak_count,
        'sample_messages': sample_msgs
    }




def _get_first_messages(df: pd.DataFrame, top_users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    获取活跃用户的第一句话。
    
    返回:
        [{'user': '用户名', 'first_message': '第一句话', 'date': '日期'}, ...]
    """
    if df.empty:
        return []
    
    df_sorted = df.sort_values('timestamp')
    first_messages = []
    
    for user_info in top_users:
        user = user_info['user']
        user_msgs = df_sorted[df_sorted['user'] == user]
        
        if not user_msgs.empty:
            first_msg = user_msgs.iloc[0]
            first_messages.append({
                'user': user,
                'first_message': first_msg['content'][:80],  # 截断
                'date': first_msg['date']
            })
    
    return first_messages


def format_stats_for_display(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化统计数据用于前端显示。
    """
    basic = stats['basic']
    
    # 格式化时间范围
    time_range = f"{basic['date_start']} ~ {basic['date_end']}"
    
    # 找出最活跃用户
    top_user = stats['top_users'][0] if stats['top_users'] else {'user': '无', 'count': 0}
    
    # 最活跃时段描述
    hour = basic['most_active_hour']
    hour_desc = f"{hour}:00 - {hour+1}:00"
    
    return {
        'overview': {
            'total_messages': basic['total_messages'],
            'total_users': basic['total_users'],
            'time_range': time_range,
            'total_days': basic['total_days'],
            'avg_messages_per_day': basic['avg_messages_per_day'],
            'most_active_hour': hour_desc,
            'top_user': top_user['user'],
            'top_user_count': top_user['count'],
        },
        'charts': {
            'top_users': stats['top_users'],
            'daily_trend': stats['daily_trend'],
            'hourly_distribution': stats['hourly_distribution'],
            'word_cloud': stats['word_frequency'],
        }
    }


# 使用示例
if __name__ == '__main__':
    import sys
    from data_loader import load_chat_data
    
    if len(sys.argv) < 2:
        print("用法: python stats_engine.py <json_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        df, session = load_chat_data(file_path)
        stats = calculate_stats(df)
        formatted = format_stats_for_display(stats)
        
        print("\n=== 基础统计 ===")
        for key, value in formatted['overview'].items():
            print(f"  {key}: {value}")
        
        print("\n=== 话痨排行榜 (Top 5) ===")
        for i, user in enumerate(stats['top_users'][:5], 1):
            print(f"  {i}. {user['user']}: {user['count']} 条 ({user['percentage']}%)")
        
        print("\n=== 高频词汇 (Top 10) ===")
        for word in stats['word_frequency'][:10]:
            print(f"  {word['word']}: {word['count']}")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
