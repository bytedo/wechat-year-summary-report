"""
report_builder.py - HTML 报告生成模块

使用 Jinja2 模板渲染静态 HTML 报告。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader


class ReportBuilder:
    """
    HTML 报告生成器。
    """
    
    def __init__(self, template_dir: str = None):
        """
        初始化报告生成器。
        
        参数:
            template_dir: 模板目录路径
        """
        if template_dir is None:
            template_dir = Path(__file__).parent.parent / 'templates'
        
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )
    
    def build(
        self,
        session_info: dict,
        stats_data: dict,
        ai_result: dict,
        output_path: str = None,
        vector_data: dict = None,
        memories_data: dict = None
    ) -> str:
        """
        生成 HTML 报告。
        
        参数:
            session_info: 会话信息
            stats_data: 统计数据
            ai_result: AI 分析结果
            output_path: 输出文件路径
            
        返回:
            生成的 HTML 内容
        """
        template = self.env.get_template('report.html')
        
        # 准备模板数据
        context = {
            'group_name': session_info.get('displayName', '未知群聊'),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # 概览数据
            'overview': stats_data['overview'],
            
            # 图表数据（转为 JSON 字符串供 ECharts 使用）
            'charts_data': {
                'top_users': json.dumps(stats_data['charts']['top_users'], ensure_ascii=False),
                'daily_trend': json.dumps(stats_data['charts']['daily_trend'], ensure_ascii=False),
                'hourly_distribution': json.dumps(stats_data['charts']['hourly_distribution'], ensure_ascii=False),
                'word_cloud': json.dumps(stats_data['charts']['word_cloud'], ensure_ascii=False),
            },
            
            # AI 分析结果
            'ai_analysis': ai_result.get('raw_content', ''),
            'is_mock': ai_result.get('is_mock', False),
            
            # 向量分析数据
            'has_vector_data': vector_data is not None and len(vector_data.get('scatter_data', [])) > 0,
            'vector_data': {
                'scatter_data': json.dumps(vector_data.get('scatter_data', []), ensure_ascii=False) if vector_data else '[]',
                'cluster_stats': json.dumps(vector_data.get('cluster_stats', []), ensure_ascii=False) if vector_data else '[]',
            } if vector_data else None,
            
            # 怀旧数据
            'has_memories_data': memories_data is not None,
            'memories_data': memories_data,
        }
        
        # 渲染模板
        html_content = template.render(**context)
        
        # 保存文件
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(html_content, encoding='utf-8')
            print(f"报告已生成: {output_file.absolute()}")
        
        return html_content


def generate_report(
    session_info: dict,
    stats_data: dict,
    ai_result: dict,
    output_dir: str = None,
    filename: str = None,
    vector_data: dict = None,
    memories_data: dict = None
) -> str:
    """
    便捷函数：生成并保存报告。
    
    参数:
        session_info: 会话信息
        stats_data: 统计数据
        ai_result: AI 分析结果
        output_dir: 输出目录
        filename: 文件名（不含扩展名）
        
    返回:
        输出文件路径
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'output'
    
    if filename is None:
        group_name = session_info.get('displayName', 'report')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{group_name}_{timestamp}"
    
    # 清理文件名中的非法字符
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', ' '))
    
    output_path = Path(output_dir) / f"{filename}.html"
    
    builder = ReportBuilder()
    builder.build(session_info, stats_data, ai_result, str(output_path), vector_data, memories_data)
    
    return str(output_path)


# 使用示例
if __name__ == '__main__':
    # 测试用的模拟数据
    mock_session = {'displayName': '测试群聊'}
    mock_stats = {
        'overview': {
            'total_messages': 1234,
            'total_users': 56,
            'time_range': '2024-01-01 ~ 2024-12-31',
            'total_days': 365,
            'avg_messages_per_day': 3.4,
            'most_active_hour': '21:00 - 22:00',
            'top_user': '活跃用户',
            'top_user_count': 100,
        },
        'charts': {
            'top_users': [{'user': '用户A', 'count': 100, 'percentage': 10}],
            'daily_trend': [{'date': '2024-01-01', 'count': 10}],
            'hourly_distribution': [{'hour': i, 'count': i * 5} for i in range(24)],
            'word_cloud': [{'word': '测试', 'count': 50}],
        }
    }
    mock_ai = {'raw_content': '## 测试分析\n这是一段测试内容。', 'is_mock': True}
    
    print("请先创建 templates/report.html 模板文件")
