"""
ai/cluster_naming.py - 聚类命名混合类

为向量聚类生成有意义的话题名称。
"""

import logging

logger = logging.getLogger(__name__)


class ClusterNamingMixin:
    """聚类命名功能。"""
    
    def summarize_clusters(self, cluster_representatives: dict) -> dict:
        """
        使用 LLM 为聚类生成有意义的话题名称。
        
        参数:
            cluster_representatives: 每个聚类的代表性消息
                {0: [{'content': '...', 'user': '...'}, ...], 1: [...], ...}
                
        返回:
            话题名称字典 {0: '话题名', 1: '话题名', ...}
        """
        if self.mock_mode:
            return self._mock_summarize_clusters(cluster_representatives)
        
        # 构建 Prompt
        prompt = self._build_cluster_prompt(cluster_representatives)
        
        try:
            content = self._call_api(
                messages=[
                    {
                        "role": "system",
                        "content": "你是群聊星系的命名大师，擅长用电影片名或小说章节的风格为话题组起名。你的名字要有故事感、画面感，能让群友一看就想起那些美好时光。严禁使用功能性、事务性、负面的命名。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return self._parse_cluster_names(content, cluster_representatives)
            
        except Exception as e:
            logger.warning(f"话题命名失败: {e}，使用默认名称")
            return self._mock_summarize_clusters(cluster_representatives)
    
    def _build_cluster_prompt(self, cluster_representatives: dict) -> str:
        """构建聚类命名的提示词。"""
        lines = ["以下是通过向量算法自动聚类的群聊消息组，请为每组起一个充满故事感的名字。\n"]
        
        for cluster_id, messages in cluster_representatives.items():
            if not messages:
                continue
            
            lines.append(f"## 分组 {cluster_id}")
            for msg in messages[:8]:  # 每组展示8条增加理解
                content = msg['content'][:80]  # 截断长内容
                lines.append(f"- {msg['user']}: {content}")
            lines.append("")
        
        lines.append("""
请返回 JSON 格式的结果，例如：
{"0": "深夜食堂", "1": "午后摸鱼时光", "2": "周末奇遇记"}

## 命名风格要求：
- **像电影片名**：有画面感、故事感（如"深夜食堂"、"那些年我们一起追的剧"）
- **像小说章节**：温馨有趣（如"午后摸鱼时光"、"打工人的日常"）
- **简短有力**：2-6个字，朗朗上口
- **勾起回忆**：让群友一看就能想起那些对话

## ⛔ 禁止使用：
- 功能性命名（如"需求招募"、"问题解答"、"信息咨询"）
- 抽象命名（如"难以启齿"、"深度交流"、"综合讨论"）
- 负面命名（如"分手"、"吐槽"、"抱怨"、"冲突"）""")
        
        return "\n".join(lines)
    
    def _parse_cluster_names(self, content: str, cluster_representatives: dict) -> dict:
        """解析 LLM 返回的话题名称。"""
        import json
        
        # 尝试提取 JSON
        try:
            # 尝试直接解析
            names = json.loads(content)
            return {int(k): v for k, v in names.items()}
        except:
            pass
        
        # 尝试从文本中提取 JSON 块
        import re
        json_match = re.search(r'\{[^{}]+\}', content)
        if json_match:
            try:
                names = json.loads(json_match.group())
                return {int(k): v for k, v in names.items()}
            except:
                pass
        
        # 失败则返回默认名称
        return self._mock_summarize_clusters(cluster_representatives)
    
    def _mock_summarize_clusters(self, cluster_representatives: dict) -> dict:
        """Mock 模式：返回默认话题名称。"""
        default_names = [
            "日常闲聊", "技术交流", "午餐拼单",
            "吐槽大会", "表情包互动", "深夜emo",
            "摸鱼时间", "周末计划", "生活分享"
        ]
        
        result = {}
        for i, cluster_id in enumerate(cluster_representatives.keys()):
            name = default_names[i % len(default_names)]
            result[cluster_id] = name
        
        return result
