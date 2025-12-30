"""
data_loader.py - 数据加载与清洗模块

负责加载微信群聊导出的 JSON 文件，进行数据清洗和字段标准化。
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

import pandas as pd


def load_chat_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    加载并清洗微信群聊 JSON 数据。
    
    参数:
        file_path: JSON 文件路径
        
    返回:
        清洗后的 DataFrame，包含以下列:
        - user: 发送者昵称
        - content: 消息内容
        - timestamp: datetime 对象
        - hour: 小时 (0-23)
        - date: 日期字符串 (YYYY-MM-DD)
        - weekday: 星期几 (0=周一, 6=周日)
    """
    # 加载 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取会话信息
    session_info = data.get('session', {})
    messages = data.get('messages', [])
    
    if not messages:
        raise ValueError("JSON 文件中没有找到消息数据")
    
    # 转换为 DataFrame
    df = pd.DataFrame(messages)
    
    # 数据清洗
    df = _filter_message_types(df)
    df = _clean_content(df)
    df = _standardize_fields(df)
    df = _add_time_columns(df)
    
    return df, session_info


def _filter_message_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤消息类型，仅保留文本消息和引用消息。
    """
    valid_types = ['文本消息', '引用消息']
    return df[df['type'].isin(valid_types)].copy()


def _clean_content(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗消息内容:
    - 去除空消息
    - 去除 XML 标签（<...>）
    """
    # 去除空消息
    df = df[df['content'].notna() & (df['content'].str.strip() != '')]
    
    # 去除 XML 标签（以 < 开头以 > 结尾的内容）
    df['content'] = df['content'].apply(_remove_xml_tags)
    
    # 再次过滤清洗后变空的消息
    df = df[df['content'].str.strip() != '']
    
    return df


def _remove_xml_tags(text: str) -> str:
    """
    移除文本中的 XML 标签。
    """
    if not isinstance(text, str):
        return ''
    
    # 移除完整的 XML 标签块（如 <msg>...</msg>）
    cleaned = re.sub(r'<[^>]+>', '', text)
    
    return cleaned.strip()


def _standardize_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    字段标准化:
    - senderDisplayName -> user
    - createTime -> timestamp
    - 如果 senderDisplayName 为空，使用 senderUsername 作为回退
    """
    # 处理发送者名称（如果昵称为空则使用用户名）
    df['user'] = df.apply(
        lambda row: row['senderDisplayName'] 
        if pd.notna(row.get('senderDisplayName')) and str(row.get('senderDisplayName')).strip() 
        else row.get('senderUsername', '未知用户'),
        axis=1
    )
    
    # 保留原始时间戳列
    df['timestamp_raw'] = df['createTime']
    
    # 只保留需要的列
    columns_to_keep = ['user', 'content', 'timestamp_raw', 'type']
    df = df[[col for col in columns_to_keep if col in df.columns]]
    
    return df


def _add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加时间相关列:
    - timestamp: datetime 对象（本地时区）
    - hour: 小时 (0-23)
    - date: 日期字符串 (YYYY-MM-DD)
    - weekday: 星期几 (0=周一, 6=周日)
    """
    # 将 Unix 时间戳转换为 datetime（UTC）
    df['timestamp'] = pd.to_datetime(df['timestamp_raw'], unit='s', utc=True)
    
    # 转换为本地时区（北京时间 UTC+8）
    df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Shanghai')
    
    # 移除时区信息（保留本地时间值）
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    # 添加时间维度列
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df['weekday'] = df['timestamp'].dt.weekday  # 0=周一, 6=周日
    
    # 删除原始时间戳列
    df = df.drop(columns=['timestamp_raw'])
    
    return df


def get_session_info(file_path: str) -> Dict[str, Any]:
    """
    获取会话基本信息。
    
    参数:
        file_path: JSON 文件路径
        
    返回:
        包含群名称、wxid、消息总数的字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('session', {})


# 使用示例
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python data_loader.py <json_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        df, session = load_chat_data(file_path)
        print(f"\n=== 会话信息 ===")
        print(f"群名称: {session.get('displayName', '未知')}")
        print(f"原始消息数: {session.get('messageCount', 'N/A')}")
        
        print(f"\n=== 清洗后的数据 ===")
        print(f"有效消息数: {len(df)}")
        print(f"参与人数: {df['user'].nunique()}")
        print(f"时间范围: {df['date'].min()} ~ {df['date'].max()}")
        
        print(f"\n=== 数据预览 (前5条) ===")
        print(df[['user', 'content', 'date', 'hour']].head())
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
