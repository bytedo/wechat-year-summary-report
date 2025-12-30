"""
poster_builder.py - 海报式报告生成器

生成移动端优先的海报式动态报告。
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from jinja2 import Environment, FileSystemLoader

from .analyzers import get_monthly_analysis, get_yearly_highlights
from .ai_analyzer import AIAnalyzer


class PosterBuilder:
    """海报式报告构建器"""
    
    def __init__(self, template_dir: str = None):
        """
        初始化构建器。
        
        参数:
            template_dir: 模板目录路径
        """
        if template_dir is None:
            template_dir = Path(__file__).parent.parent / 'templates' / 'poster'
        
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )
    
    def build(
        self,
        session_info: dict,
        df,  # pandas DataFrame
        memories_data: dict = None,
        output_path: str = None,
        music_url: str = None,
        use_ai: bool = True,
        vector_data: dict = None
    ) -> str:
        """
        生成海报式报告。
        
        参数:
            session_info: 会话信息（群名等）
            df: 消息 DataFrame
            memories_data: 怀旧数据（金句等）
            output_path: 输出路径
            music_url: 背景音乐 URL
            use_ai: 是否使用 AI 生成话题回忆
            vector_data: 向量聚类数据（用于散点图可视化）
            
        返回:
            生成的 HTML 内容
        """
        # 获取分析数据
        monthly_data = get_monthly_analysis(df)
        yearly_data = get_yearly_highlights(df)
        
        # 提取年份
        year = df['timestamp'].dt.year.mode().iloc[0] if not df.empty else datetime.now().year
        
        # 使用 AI 生成话题回忆 AI 分析
        topic_memories = []
        user_profiles_mbti = []
        weekly_ai_summary = ""
        
        if use_ai:
            print("   🧠 正在调用 AI 进行深度分析...")
            ai_analyzer = AIAnalyzer()
            
            # 1. 生成周度深度总结 & 获取周结构化数据（必须先完成，后续依赖）
            print("   📊 1/4 正在进行周度全量扫描...")
            from .analyzers.weekly_analyzer import get_weekly_samples_for_ai
            weekly_samples = get_weekly_samples_for_ai(df, max_per_week=1000)
            weekly_ai_summary, weekly_summaries_dict = ai_analyzer.analyze_weekly_batches(weekly_samples)
            print("   ✓ 周度深度总结已生成")
            
            # 2. 基于周报生成月度话题回忆 (更精准)
            if monthly_data:
                print("   📅 2/4 正在生成月度话题回忆 (基于周报)...")
                if weekly_summaries_dict:
                    topic_memories = ai_analyzer.generate_monthly_summary_from_weekly(monthly_data, weekly_summaries_dict)
                else:
                    topic_memories = ai_analyzer.generate_topic_memories(monthly_data)
                print(f"   ✓ 已生成 {len(topic_memories)} 个月的话题回忆")
            
            # 3. 生成用户画像及 MBTI（分析所有群友）
            print("   👥 3/4 正在生成用户画像...")
            # 获取所有用户（按发言数量排序）
            user_counts = df['user'].value_counts()
            all_users = user_counts.index.tolist()
            if all_users:
                user_profiles_mbti = ai_analyzer.generate_user_profiles_with_mbti(df, all_users)
                print(f"   ✓ 已生成 {len(user_profiles_mbti)} 位用户的 MBTI 画像")
            
            # 4. AI 筛选年度关键词
            raw_keywords = yearly_data.get('keywords', [])
            if raw_keywords:
                print("   🏷️ 4/4 正在筛选年度关键词...")
                refined_keywords = ai_analyzer.refine_keywords(raw_keywords)
                # 更新 yearly_data 中的关键词
                yearly_data['keywords'] = refined_keywords
                print(f"   ✓ 已筛选 {len(refined_keywords)} 个年度关键词")
        
        # 构建上下文
        context = {
            'group_name': session_info.get('displayName', '微信群聊'),
            'year': int(year),
            'date_range': f"{df['date'].min()} ~ {df['date'].max()}" if not df.empty else '',
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # 年度数据
            'overview': yearly_data.get('overview', {}),
            'rankings': yearly_data.get('rankings', {}),
            'highlights': yearly_data.get('highlights', {}),
            'timeline': yearly_data.get('timeline', []),
            'keywords': yearly_data.get('keywords', []),
            'fun_facts': yearly_data.get('fun_facts', []),
            'user_profiles': yearly_data.get('user_profiles', []),
            
            # AI 分析数据
            'monthly_data': monthly_data,
            'topic_memories': topic_memories,
            'user_profiles_mbti': user_profiles_mbti,
            'weekly_ai_summary': weekly_ai_summary,
            
            # 图表数据
            'charts': yearly_data.get('charts', {}),
            
            # 向量聚类数据（散点图可视化）
            'has_vector_data': vector_data is not None and len(vector_data.get('scatter_data', [])) > 0,
            'vector_data': vector_data,
            
            # 背景音乐
            'music_url': music_url,
        }
        
        # 渲染模板
        template = self.env.get_template('index.html')
        html_content = template.render(**context)
        
        # 保存文件
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(html_content, encoding='utf-8')
            print(f"海报报告已生成: {output_file.absolute()}")
        
        return html_content


def generate_poster_report(
    session_info: dict,
    df,
    memories_data: dict = None,
    output_dir: str = None,
    filename: str = None,
    music_url: str = None,
    vector_data: dict = None
) -> str:
    """
    便捷函数：生成海报式报告。
    
    参数:
        session_info: 会话信息
        df: 消息 DataFrame
        memories_data: 怀旧数据
        output_dir: 输出目录
        filename: 文件名（不含扩展名）
        music_url: 背景音乐 URL
        vector_data: 向量聚类数据（用于散点图）
        
    返回:
        输出文件路径
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'output'
    
    if filename is None:
        group_name = session_info.get('displayName', 'poster')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{group_name}_海报_{timestamp}"
    
    # 清理文件名
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', ' '))
    
    output_path = Path(output_dir) / f"{filename}.html"
    
    builder = PosterBuilder()
    builder.build(session_info, df, memories_data, str(output_path), music_url, vector_data=vector_data)
    
    return str(output_path)
