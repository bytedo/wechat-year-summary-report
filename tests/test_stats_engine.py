"""
tests/test_stats_engine.py - stats_engine 模块单元测试

测试统计分析功能，包括基础统计、用户排行、词频分析等。
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# 添加 src 到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stats_engine import (
    calculate_stats,
    format_stats_for_display,
    _get_basic_stats,
    _get_top_users,
    _get_daily_trend,
    _get_hourly_distribution,
    _find_peak_day,
    _find_silence_breaker,
    _find_hot_messages,
    _get_first_messages,
    calculate_memories_stats,
)


# ==================== 测试数据 ====================

@pytest.fixture
def sample_df():
    """创建测试 DataFrame"""
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    
    data = []
    users = ['用户A', '用户B', '用户C']
    
    # 创建 100 条消息
    for i in range(100):
        data.append({
            'user': users[i % 3],
            'content': f'消息内容{i}，这是一条测试消息',
            'timestamp': base_time + timedelta(hours=i),
            'hour': (10 + i) % 24,
            'date': (base_time + timedelta(hours=i)).strftime('%Y-%m-%d'),
            'weekday': (base_time + timedelta(hours=i)).weekday()
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def empty_df():
    """创建空 DataFrame"""
    return pd.DataFrame(columns=['user', 'content', 'timestamp', 'hour', 'date', 'weekday'])


# ==================== 测试用例 ====================

class TestGetBasicStats:
    """测试 _get_basic_stats 函数"""
    
    def test_returns_correct_structure(self, sample_df):
        """测试返回正确的数据结构"""
        result = _get_basic_stats(sample_df)
        
        expected_keys = [
            'total_messages', 'total_users', 'date_start', 
            'date_end', 'total_days', 'most_active_hour', 'avg_messages_per_day'
        ]
        for key in expected_keys:
            assert key in result, f"缺少键: {key}"
    
    def test_total_messages_count(self, sample_df):
        """测试总消息数"""
        result = _get_basic_stats(sample_df)
        assert result['total_messages'] == 100
    
    def test_total_users_count(self, sample_df):
        """测试用户数"""
        result = _get_basic_stats(sample_df)
        assert result['total_users'] == 3


class TestGetTopUsers:
    """测试 _get_top_users 函数"""
    
    def test_returns_list(self, sample_df):
        """测试返回列表"""
        result = _get_top_users(sample_df, top_n=5)
        assert isinstance(result, list)
    
    def test_user_has_required_fields(self, sample_df):
        """测试用户数据包含必需字段"""
        result = _get_top_users(sample_df, top_n=5)
        
        if result:
            user = result[0]
            assert 'user' in user
            assert 'count' in user
            assert 'percentage' in user
    
    def test_respects_top_n(self, sample_df):
        """测试 top_n 参数"""
        result = _get_top_users(sample_df, top_n=2)
        assert len(result) <= 2
    
    def test_sorted_by_count(self, sample_df):
        """测试按消息数排序"""
        result = _get_top_users(sample_df, top_n=10)
        
        if len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i]['count'] >= result[i + 1]['count']


class TestGetDailyTrend:
    """测试 _get_daily_trend 函数"""
    
    def test_returns_list(self, sample_df):
        """测试返回列表"""
        result = _get_daily_trend(sample_df)
        assert isinstance(result, list)
    
    def test_item_has_date_and_count(self, sample_df):
        """测试每项包含 date 和 count"""
        result = _get_daily_trend(sample_df)
        
        if result:
            item = result[0]
            assert 'date' in item
            assert 'count' in item


class TestGetHourlyDistribution:
    """测试 _get_hourly_distribution 函数"""
    
    def test_returns_24_items(self, sample_df):
        """测试返回 24 个小时的数据"""
        result = _get_hourly_distribution(sample_df)
        assert len(result) == 24
    
    def test_hours_0_to_23(self, sample_df):
        """测试小时范围 0-23"""
        result = _get_hourly_distribution(sample_df)
        hours = [item['hour'] for item in result]
        
        for h in range(24):
            assert h in hours


class TestFindPeakDay:
    """测试 _find_peak_day 函数"""
    
    def test_returns_dict(self, sample_df):
        """测试返回字典"""
        result = _find_peak_day(sample_df)
        assert isinstance(result, dict)
    
    def test_has_required_keys(self, sample_df):
        """测试包含必需的键"""
        result = _find_peak_day(sample_df)
        
        assert 'date' in result
        assert 'count' in result
        assert 'sample_messages' in result
    
    def test_empty_df_returns_none_date(self, empty_df):
        """测试空 DataFrame 返回 None date"""
        result = _find_peak_day(empty_df)
        assert result['date'] is None


class TestFindSilenceBreaker:
    """测试 _find_silence_breaker 函数"""
    
    def test_returns_none_for_short_df(self):
        """测试短 DataFrame 返回 None"""
        df = pd.DataFrame({
            'user': ['A'],
            'content': ['hi'],
            'timestamp': [datetime.now()],
            'date': ['2024-01-01']
        })
        result = _find_silence_breaker(df)
        assert result is None
    
    def test_finds_long_silence(self):
        """测试找到长时间沉默"""
        df = pd.DataFrame({
            'user': ['A', 'B'],
            'content': ['消息1', '消息2'],
            'timestamp': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 3, 10, 0)  # 48 小时后
            ],
            'date': ['2024-01-01', '2024-01-03']
        })
        
        result = _find_silence_breaker(df, silence_hours=24)
        assert result is not None
        assert result['user'] == 'B'
        assert result['silence_hours'] >= 48


class TestCalculateStats:
    """测试 calculate_stats 函数"""
    
    def test_returns_correct_structure(self, sample_df):
        """测试返回正确的数据结构"""
        result = calculate_stats(sample_df)
        
        expected_keys = ['basic', 'top_users', 'daily_trend', 'hourly_distribution', 'word_frequency']
        for key in expected_keys:
            assert key in result, f"缺少键: {key}"


class TestFormatStatsForDisplay:
    """测试 format_stats_for_display 函数"""
    
    def test_returns_overview_and_charts(self, sample_df):
        """测试返回 overview 和 charts"""
        stats = calculate_stats(sample_df)
        result = format_stats_for_display(stats)
        
        assert 'overview' in result
        assert 'charts' in result


# ==================== 运行测试 ====================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
