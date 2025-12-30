#!/usr/bin/env python3
"""
main.py - 微信群聊分析工具主程序

将各模块串联，实现完整的分析流程：
1. 加载数据
2. 统计分析
3. AI 分析
4. 生成报告
"""

import argparse
import sys
from pathlib import Path

# 将 src 目录添加到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import load_chat_data
from src.stats_engine import calculate_stats, format_stats_for_display, calculate_memories_stats
from src.ai_analyzer import AIAnalyzer
from src.vector_engine import SemanticAnalyzer
from src.poster_builder import generate_poster_report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='微信群聊分析工具 - 生成年度分析报告',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python main.py data/chat_export.json
  python main.py data/chat.json -o reports/
  python main.py data/chat.json --no-ai
        '''
    )
    
    parser.add_argument('input', help='微信群聊导出的 JSON 文件路径')
    parser.add_argument('-o', '--output', default='output', help='输出目录 (默认: output)')
    parser.add_argument('--no-ai', action='store_true', help='跳过 AI 分析，仅生成统计报告')
    parser.add_argument('--no-vector', action='store_true', help='跳过向量语义分析（加速处理）')
    parser.add_argument('--no-gpu', action='store_true', help='禁用 GPU 加速，强制使用 CPU')
    parser.add_argument('--clusters', type=int, default=6, help='聚类数量 (默认: 6)')
    parser.add_argument('--mock', action='store_true', help='强制使用 AI Mock 模式')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细输出')
    parser.add_argument('--music', type=str, default=None, help='报告的背景音乐 URL')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ 错误: 文件不存在 - {input_path}")
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print("🔍 微信群聊分析工具")
    print(f"{'='*50}\n")
    
    try:
        # Step 1: 加载数据
        print("📂 正在加载数据...")
        df, session_info = load_chat_data(str(input_path))
        group_name = session_info.get('displayName', '未知群聊')
        print(f"   ✓ 群名称: {group_name}")
        print(f"   ✓ 有效消息: {len(df)} 条")
        print(f"   ✓ 参与用户: {df['user'].nunique()} 人")
        
        # Step 2: 统计分析
        print("\n📊 正在进行统计分析...")
        stats = calculate_stats(df)
        formatted_stats = format_stats_for_display(stats)
        print(f"   ✓ 时间范围: {formatted_stats['overview']['time_range']}")
        print(f"   ✓ 话痨担当: {formatted_stats['overview']['top_user']}")
        print(f"   ✓ 最活跃时段: {formatted_stats['overview']['most_active_hour']}")
        
        if args.verbose:
            print("\n   📈 话痨排行榜 (Top 5):")
            for i, user in enumerate(stats['top_users'][:5], 1):
                print(f"      {i}. {user['user']}: {user['count']} 条")
        
        # Step 3: 向量语义分析
        vector_data = None
        if args.no_vector:
            print("\n🧠 跳过向量语义分析 (--no-vector)")
        else:
            print("\n🧠 正在进行深度语义分析...")
            print("   ⚠️ 首次运行需要下载模型，可能需要几分钟...")
            
            try:
                semantic_analyzer = SemanticAnalyzer(
                    n_clusters=args.clusters,
                    use_gpu=not args.no_gpu
                )
                vector_data = semantic_analyzer.analyze(df)
                
                if vector_data and vector_data.get('total_analyzed', 0) > 0:
                    print(f"   ✓ 语义分析完成，共分析 {vector_data['total_analyzed']} 条消息")
                    print(f"   ✓ 识别出 {vector_data['n_clusters']} 个话题聚类")
                    
                    # 使用 AI 为聚类命名
                    if not args.no_ai:
                        print("   🎲 正在为话题生成名称...")
                        ai_analyzer_for_naming = AIAnalyzer()
                        cluster_names = ai_analyzer_for_naming.summarize_clusters(
                            vector_data['cluster_representatives']
                        )
                        # 更新聚类统计中的名称
                        for stat in vector_data['cluster_stats']:
                            cluster_id = stat['cluster_id']
                            if cluster_id in cluster_names:
                                stat['name'] = cluster_names[cluster_id]
                        print("   ✓ 话题命名完成")
                else:
                    print("   ⚠️ 有效消息不足，跳过语义分析")
                    vector_data = None
            except Exception as e:
                print(f"   ⚠️ 向量分析失败: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                vector_data = None
        
        # Step 4: AI 分析
        if args.no_ai:
            print("\n🤖 跳过 AI 分析 (--no-ai)")
            ai_result = {
                'raw_content': '## AI 分析已跳过\n\n用户选择跳过 AI 分析功能。',
                'is_mock': True
            }
        else:
            print("\n🤖 正在进行 AI 分析...")
            analyzer = AIAnalyzer()
            
            if args.mock:
                analyzer.mock_mode = True
            
            if analyzer.mock_mode:
                print("   ⚠️ 使用 Mock 模式（未配置 API Key）")
            else:
                print(f"   ✓ 使用模型: {analyzer.model}")
            
            ai_result = analyzer.analyze(df, stats['top_users'])
            print("   ✓ AI 分析完成")
        
        # Step 5: 怀旧数据挖掘
        print("\n⏳ 正在挖掘怀旧数据...")
        memories_data = None
        try:
            memories_stats = calculate_memories_stats(df, stats['top_users'])
            memories_data = {
                'hot_messages': memories_stats['hot_messages'],
                'peak_day': memories_stats['peak_day'],
                'silence_breaker': memories_stats['silence_breaker'],
                'first_messages': memories_stats['first_messages'],
                'golden_quotes': [],
                'peak_day_summary': ''
            }
            
            # AI 甘选金句和生成巅峰日摘要
            if not args.no_ai and memories_stats['hot_messages']:
                print("   🎲 正在甘选金句...")
                ai_for_memories = AIAnalyzer()
                if args.mock:
                    ai_for_memories.mock_mode = True
                memories_data['golden_quotes'] = ai_for_memories.select_golden_quotes(
                    memories_stats['hot_messages']
                )
                print(f"   ✓ 已甘选 {len(memories_data['golden_quotes'])} 条金句")
                
                if memories_stats['peak_day'].get('date'):
                    print("   🏆 正在生成巅峰日摘要...")
                    memories_data['peak_day_summary'] = ai_for_memories.summarize_peak_day(
                        memories_stats['peak_day']
                    )
                    print("   ✓ 巅峰日摘要完成")
            
            if memories_stats['silence_breaker']:
                print(f"   ✓ 找到打破沉默的英雄: {memories_stats['silence_breaker']['user']}")
            
            print(f"   ✓ 怀旧数据挖掘完成")
            
        except Exception as e:
            print(f"   ⚠️ 怀旧数据挖掘失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        
        # Step 6: 生成海报式报告
        print("\n🎬 正在生成海报式报告...")
        output_path = generate_poster_report(
            session_info=session_info,
            df=df,
            memories_data=memories_data,
            output_dir=args.output,
            music_url=args.music,
            vector_data=vector_data
        )
        print(f"   ✓ 报告已生成: {output_path}")
        
        print(f"\n{'='*50}")
        print("✅ 报告生成成功!")
        print(f"{'='*50}")
        print(f"\n📱 报告路径: {output_path}")
        print("\n💡 提示: 用浏览器打开 HTML 文件即可查看报告")
        print("   建议在手机上竖屏查看，效果更佳！")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
