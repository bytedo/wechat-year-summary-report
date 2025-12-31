"""
poster_builder.py - 海报式报告生成器

生成移动端优先的海报式动态报告。
此模块仅负责渲染，所有 AI 调用由 main.py 统一调度。
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from jinja2 import Environment, FileSystemLoader

from .analyzers import get_monthly_analysis, get_yearly_highlights


class PosterBuilder:
    """海报式报告构建器（纯渲染，不进行 AI 调用）"""
    
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
        ai_data: dict = None,
        memories_data: dict = None,
        output_path: str = None,
        music_url: str = None,
        vector_data: dict = None
    ) -> str:
        """
        生成海报式报告。
        
        参数:
            session_info: 会话信息（群名等）
            df: 消息 DataFrame
            ai_data: AI 分析数据（由 main.py 传入）
            memories_data: 怀旧数据（金句等）
            output_path: 输出路径
            music_url: 背景音乐 URL
            vector_data: 向量聚类数据（用于散点图可视化）
            
        返回:
            生成的 HTML 内容
        """
        # 获取分析数据（非 AI）
        monthly_data = get_monthly_analysis(df)
        yearly_data = get_yearly_highlights(df)
        
        # 提取年份
        year = df['timestamp'].dt.year.mode().iloc[0] if not df.empty else datetime.now().year
        
        # 从 ai_data 获取（如果有）
        ai_data = ai_data or {}
        topic_memories = ai_data.get('topic_memories', [])
        user_profiles_mbti = ai_data.get('user_profiles_mbti', [])
        weekly_ai_summary = ai_data.get('weekly_ai_summary', '')
        refined_keywords = ai_data.get('refined_keywords', None)
        
        # 如果有 AI 优化的关键词，更新 yearly_data
        if refined_keywords:
            yearly_data['keywords'] = refined_keywords
        
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
            
            # 怀旧数据
            'memories_data': memories_data,
            
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
            print(f"   ✓ 海报报告已生成: {output_file.absolute()}")
        
        return html_content


def generate_poster_report(
    session_info: dict,
    df,
    ai_data: dict = None,
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
        ai_data: AI 分析数据
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
    builder.build(
        session_info, df, ai_data, memories_data, 
        str(output_path), music_url, vector_data
    )
    
    return str(output_path)
