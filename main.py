#!/usr/bin/env python3
"""
main.py - 微信群聊分析工具主程序

将各模块串联，实现完整的分析流程：
1. 加载数据
2. 统计分析
3. 向量语义分析
4. AI 深度分析（统一调度）
5. 生成报告
"""

import argparse
import sys
from pathlib import Path
import os

# === 全局缓存配置 ===
# 必须在导入 transformers/sentence_transformers 之前设置
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_DIR / "models")
# ====================
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import load_chat_data
from src.stats_engine import calculate_stats, format_stats_for_display, calculate_memories_stats
from src.ai import AIAnalyzer
from src.vector_engine import SemanticAnalyzer
from src.poster_builder import generate_poster_report
from src.analyzers import get_monthly_analysis, get_yearly_highlights
from src.analyzers.weekly_analyzer import get_weekly_samples_for_ai


def run_ai_analysis(df, stats, vector_data, args):
    """
    统一的 AI 分析调度器。
    
    返回:
        {
            'topic_memories': [...],
            'user_profiles_mbti': [...],
            'weekly_ai_summary': '...',
            'refined_keywords': [...],
        }
    """
    result = {
        'topic_memories': [],
        'user_profiles_mbti': [],
        'weekly_ai_summary': '',
        'refined_keywords': None,
    }
    
    if args.no_ai:
        return result
    
    print("\n🧠 正在进行 AI 深度分析...")
    analyzer = AIAnalyzer()
    
    if args.mock:
        analyzer.mock_mode = True
    
    if analyzer.mock_mode:
        print("   ⚠️ 使用 Mock 模式（未配置 API Key）")
    else:
        print(f"   ✓ 使用模型: {analyzer.model}")
    
    # 1. 周度分析
    print("   📊 [1/4] 正在进行周度全量扫描...")
    weekly_samples = get_weekly_samples_for_ai(df, max_per_week=1000)
    weekly_ai_summary, weekly_summaries_dict = analyzer.analyze_weekly_batches(weekly_samples)
    result['weekly_ai_summary'] = weekly_ai_summary
    print("   ✓ 周度深度总结已生成")
    
    # 2. 月度话题回忆
    monthly_data = get_monthly_analysis(df)
    if monthly_data:
        print("   📅 [2/4] 正在生成月度话题回忆...")
        if weekly_summaries_dict:
            result['topic_memories'] = analyzer.generate_monthly_summary_from_weekly(
                monthly_data, weekly_summaries_dict
            )
        else:
            result['topic_memories'] = analyzer.generate_topic_memories(monthly_data)
        print(f"   ✓ 已生成 {len(result['topic_memories'])} 个月的话题回忆")
    
    # 3. 用户画像及 MBTI
    print("   👥 [3/4] 正在生成用户画像...")
    user_counts = df['user'].value_counts()
    all_users = user_counts.index.tolist()
    
    if all_users:
        # 计算用户语义特征向量
        user_mbti_vectors = None
        if vector_data and vector_data.get('total_analyzed', 0) > 0:
            try:
                print("   🧬 正在计算用户语义特征向量...")
                semantic_analyzer = SemanticAnalyzer(n_clusters=vector_data.get('n_clusters', 6))
                user_mbti_vectors = semantic_analyzer.analyze_users_for_mbti(df, all_users[:30])
            except Exception as e:
                print(f"   ⚠️ 用户语义特征计算失败: {e}")
        
        result['user_profiles_mbti'] = analyzer.generate_user_profiles_with_mbti(
            df, all_users, user_vectors=user_mbti_vectors
        )
        print(f"   ✓ 已生成 {len(result['user_profiles_mbti'])} 位用户的 MBTI 画像")
    
    # 4. 年度关键词优化
    yearly_data = get_yearly_highlights(df)
    raw_keywords = yearly_data.get('keywords', [])
    if raw_keywords:
        print("   🏷️ [4/4] 正在筛选年度关键词...")
        result['refined_keywords'] = analyzer.refine_keywords(raw_keywords)
        print(f"   ✓ 已筛选 {len(result['refined_keywords'])} 个年度关键词")
    
    print("   ✓ AI 深度分析完成")
    return result


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
                        if args.mock:
                            ai_analyzer_for_naming.mock_mode = True
                        cluster_names = ai_analyzer_for_naming.summarize_clusters(
                            vector_data['cluster_representatives']
                        )
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
        
        # Step 4: AI 深度分析（统一调度）
        ai_data = run_ai_analysis(df, stats, vector_data, args)
        
        # Step 5: 怀旧数据挖掘
        print("\n⏳ 正在挖掘怀旧数据...")
        memories_data = None
        try:
            memories_stats = calculate_memories_stats(df, stats['top_users'])
            memories_data = {
                'hot_messages': memories_stats['hot_messages'],
                'peak_day': memories_stats['peak_day'],
                'first_messages': memories_stats['first_messages'],
                'golden_quotes': [],
                'peak_day_summary': ''
            }
            
            # AI 甄选金句和生成巅峰日摘要
            if not args.no_ai and memories_stats['hot_messages']:
                print("   🎲 正在甄选金句...")
                ai_for_memories = AIAnalyzer()
                if args.mock:
                    ai_for_memories.mock_mode = True
                memories_data['golden_quotes'] = ai_for_memories.select_golden_quotes(
                    memories_stats['hot_messages']
                )
                print(f"   ✓ 已甄选 {len(memories_data['golden_quotes'])} 条金句")
                
                if memories_stats['peak_day'].get('date'):
                    print("   🏆 正在生成巅峰日摘要...")
                    memories_data['peak_day_summary'] = ai_for_memories.summarize_peak_day(
                        memories_stats['peak_day']
                    )
                    print("   ✓ 巅峰日摘要完成")
            
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
            ai_data=ai_data,
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
