"""
ai - AI 分析模块

将原 ai_analyzer.py 拆分为多个子模块，提高可维护性。

模块结构:
- base.py: 基类，API 调用和基础功能
- weekly.py: 周度分析
- monthly.py: 月度分析
- user_profile.py: 用户画像和 MBTI
- keywords.py: 关键词优化
- golden_quotes.py: 金句和巅峰日
- cluster_naming.py: 聚类命名
"""

from .base import AIAnalyzerBase, retry_on_failure
from .weekly import WeeklyAnalysisMixin
from .monthly import MonthlyAnalysisMixin
from .user_profile import UserProfileMixin
from .keywords import KeywordsMixin
from .golden_quotes import GoldenQuotesMixin
from .cluster_naming import ClusterNamingMixin


class AIAnalyzer(
    AIAnalyzerBase,
    WeeklyAnalysisMixin,
    MonthlyAnalysisMixin,
    UserProfileMixin,
    KeywordsMixin,
    GoldenQuotesMixin,
    ClusterNamingMixin
):
    """
    完整的 AI 分析器，组合所有分析功能。
    
    使用方式:
        from src.ai import AIAnalyzer
        analyzer = AIAnalyzer()
        
        # 周度分析
        summary, weekly_dict = analyzer.analyze_weekly_batches(samples)
        
        # 月度话题
        memories = analyzer.generate_topic_memories(monthly_data)
        
        # 用户画像
        profiles = analyzer.generate_user_profiles_with_mbti(df, users)
        
        # 关键词
        keywords = analyzer.refine_keywords(raw_keywords)
        
        # 金句
        quotes = analyzer.select_golden_quotes(candidates)
        
        # 聚类命名
        cluster_names = analyzer.summarize_clusters(cluster_representatives)
    """
    pass


__all__ = [
    'AIAnalyzer',
    'AIAnalyzerBase',
    'retry_on_failure',
    'WeeklyAnalysisMixin',
    'MonthlyAnalysisMixin', 
    'UserProfileMixin',
    'KeywordsMixin',
    'GoldenQuotesMixin',
    'ClusterNamingMixin',
]
