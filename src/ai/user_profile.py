"""
ai/user_profile.py - 用户画像模块

包含用户画像和 MBTI 分析功能。
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class UserProfileMixin:
    """
    用户画像混入类，提供用户画像和 MBTI 分析方法。
    
    需要与 AIAnalyzerBase 一起使用。
    """
    
    def generate_user_profiles_with_mbti(
        self, 
        df: pd.DataFrame, 
        top_users: List[str],
        user_vectors: dict = None,
        use_cache: bool = True
    ) -> List[dict]:
        """
        生成用户画像及 MBTI 预测。
        
        参数:
            df: 完整消息 DataFrame
            top_users: 需要分析的用户列表 (用户名)
            user_vectors: 可选，来自 vector_engine 的用户语义特征向量
            use_cache: 是否使用缓存（默认开启）
            
        返回:
            [{'user': '...', 'persona': '...', 'description': '...', 'mbti': '...', 'mbti_analysis': {...}}, ...]
        """
        # === 缓存处理 ===
        cache_dir = Path(__file__).parent.parent.parent / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算缓存键
        cache_key_data = json.dumps({
            'users': sorted(top_users[:30]),
            'msg_count': len(df),
            'has_vectors': user_vectors is not None
        }, ensure_ascii=False, sort_keys=True)
        cache_hash = hashlib.md5(cache_key_data.encode()).hexdigest()[:12]
        cache_file = cache_dir / f"user_profiles_{cache_hash}.json"
        
        # 尝试读取缓存
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_profiles = json.load(f)
                    print(f"   💾 已加载用户画像缓存 ({len(cached_profiles)} 位用户)")
                    return cached_profiles
            except Exception as e:
                logger.warning(f"缓存读取失败: {e}")
        
        profiles = []
        if self.mock_mode:
            return self._mock_user_profiles_mbti(top_users)
        
        print(f"   👥 正在生成用户画像及 MBTI (分析前 {len(top_users)} 位活跃用户)...")
        
        # 尝试使用 tqdm 进度条
        try:
            from tqdm import tqdm
            user_iter = tqdm(top_users, desc="   用户画像", unit="人", ncols=60)
        except ImportError:
            user_iter = top_users
        
        for user in user_iter:
            # 提取该用户的发言样本
            user_df = df[df['user'] == user]
            sample_size = min(2000, len(user_df))
            user_msgs = user_df['content'].sample(n=sample_size).tolist()
            msg_text = "\n".join(user_msgs)[:15000]
            
            # 构建语义特征部分（如果有）
            semantic_section = ""
            if user_vectors and user in user_vectors:
                uv = user_vectors[user]
                style = uv.get('style_features', {})
                topic_dist = uv.get('topic_distribution', {})
                main_topics = list(topic_dist.keys())[:3]
                
                semantic_section = f"""
## 用户语义特征（基于 embedding 分析）：
- 发言数量：{uv.get('message_count', 0)} 条
- 发言风格稳定性：{style.get('stability', '适中')}（标准差: {uv.get('std_value', 0):.3f}）
- 与群体的差异度：{style.get('deviation', '合群')}（偏离分数: {uv.get('deviation_score', 0):.3f}）
- 主要话题偏好：话题 {', '.join(map(str, main_topics))} （占比分别为 {', '.join([f'{topic_dist.get(t, 0):.1%}' for t in main_topics])}）

请结合以上语义特征进行 MBTI 分析：
- 「稳定」风格可能暗示 J（计划型）倾向
- 「多变」风格可能暗示 P（灵活型）倾向
- 「独特」表达可能暗示 N（直觉型）或 I（内向型）
- 「合群」表达可能暗示 S（感知型）或 E（外向型）

"""
            
            prompt = f"""请为这位群友写一份温暖的人物画像，并进行 MBTI 性格分析。

## 用户发言样本：
{msg_text}
{semantic_section}
## 任务：
1. **温暖标签**：给 TA 一个充满喜爱的称号（4-6字）
2. **画像描述**：用温暖的一句话描述 TA（30字内）
3. **MBTI 四维度分析**：对 E/I、S/N、T/F、J/P 四个维度分别判断，给出置信度（0-1）和简短理由

## 写作指南：
- 想象你在向新朋友介绍"我们群里的宝藏朋友"
- MBTI 分析要基于发言内容和语义特征，给出可解释的判断
- 置信度反映你对该维度判断的确定程度

## ❗❗ 输出规则（必须严格遵守）：
1. **只输出纯 JSON**，不要任何额外文字
2. **不要使用 markdown 代码块**
3. **不要输出多个 JSON 对象**

## 输出格式（JSON）：
{{
  "persona": "温暖标签",
  "description": "画像描述",
  "mbti": "XXXX",
  "mbti_analysis": {{
    "E_I": {{"result": "E或I", "confidence": 0.0-1.0, "reason": "简短理由"}},
    "S_N": {{"result": "S或N", "confidence": 0.0-1.0, "reason": "简短理由"}},
    "T_F": {{"result": "T或F", "confidence": 0.0-1.0, "reason": "简短理由"}},
    "J_P": {{"result": "J或P", "confidence": 0.0-1.0, "reason": "简短理由"}}
  }}
}}"""

            try:
                content = self._call_api(
                    messages=[
                        {"role": "system", "content": "你是一位精通 MBTI 性格分析的画像师。基于用户发言和语义特征，给出稳定、可解释的 MBTI 判断。只输出JSON。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6,
                    max_tokens=400
                )
                
                # 解析 JSON
                json_candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}', content)
                parsed = False
                for candidate in json_candidates:
                    try:
                        res = json.loads(candidate)
                        if 'persona' in res or 'mbti' in res:
                            mbti_analysis = res.get('mbti_analysis', {})
                            if mbti_analysis:
                                mbti_type = (
                                    mbti_analysis.get('E_I', {}).get('result', 'I') +
                                    mbti_analysis.get('S_N', {}).get('result', 'N') +
                                    mbti_analysis.get('T_F', {}).get('result', 'F') +
                                    mbti_analysis.get('J_P', {}).get('result', 'P')
                                )
                            else:
                                mbti_type = res.get('mbti', 'INFP')
                            
                            profiles.append({
                                'user': user,
                                'persona': res.get('persona', '神秘群友'),
                                'description': res.get('description', '暂无描述'),
                                'mbti': mbti_type,
                                'mbti_analysis': mbti_analysis
                            })
                            parsed = True
                            break
                    except:
                        continue
                if not parsed:
                    profiles.append({
                        'user': user, 
                        'persona': '神秘群友', 
                        'description': content[:30] if content else '暂无描述', 
                        'mbti': 'UNKNOWN',
                        'mbti_analysis': {}
                    })
                     
            except Exception as e:
                logger.warning(f"分析用户 {user} 失败: {e}")
                profiles.append({
                    'user': user, 
                    'persona': '低调路人', 
                    'description': '保持神秘', 
                    'mbti': 'ISTJ',
                    'mbti_analysis': {}
                })
        
        # 保存缓存
        if use_cache and profiles:
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(profiles, f, ensure_ascii=False, indent=2)
                print(f"   💾 用户画像已缓存")
            except Exception as e:
                logger.warning(f"缓存保存失败: {e}")
        
        return profiles
    
    def _mock_user_profiles_mbti(self, users: List[str]) -> List[dict]:
        """Mock 用户画像数据"""
        roles = [
            ('群里的智囊团', 'INTJ', '总能给出深思熟虑的建议'),
            ('好奇宝宝', 'INTP', '对一切新鲜事物充满探索欲'),
            ('点子大王', 'ENTP', '脑洞大开，创意无限'),
            ('天生领航者', 'ENTJ', '带大家一起冲，从不掉队'),
            ('深夜暖心小精灵', 'INFJ', '总在最需要的时候出现'),
            ('温柔的理想家', 'INFP', '用文字传递温暖和希望'),
            ('群里的小太阳', 'ENFP', '随时都能点亮大家的一天'),
            ('暖场担当', 'ENFJ', '让每个人都感到被欢迎'),
            ('默默守护者', 'ISTJ', '靠谱得让人安心'),
            ('温柔守护者', 'ISFJ', '悄悄关心着每一个人'),
            ('秩序维护员', 'ESTJ', '群里有事，第一个站出来'),
            ('热心大使', 'ESFJ', '总在张罗聚会和活动'),
            ('神秘冷酷侠', 'ISTP', '话不多但句句在点上'),
            ('浪漫生活家', 'ISFP', '把日常过成诗'),
            ('快乐制造机', 'ESTP', '有 TA 的地方就有笑声'),
            ('气氛担当', 'ESFP', '群里的开心果，永远活力满满'),
        ]
        results = []
        for i, user in enumerate(users):
            role = roles[i % len(roles)]
            results.append({
                'user': user,
                'persona': role[0],
                'description': f"我们群里的{role[0]}，{role[2]}，有 TA 的地方就充满温暖",
                'mbti': role[1]
            })
        return results
