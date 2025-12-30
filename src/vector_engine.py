"""
vector_engine.py - 向量分析引擎模块

基于 Sentence-Transformers 实现语义分析：
1. 文本向量化 (Embedding)
2. K-Means 聚类
3. t-SNE 降维可视化
"""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm


class SemanticAnalyzer:
    """
    语义分析器：实现文本向量化、聚类和降维。
    """
    
    # 默认模型名称（轻量级中文模型）
    DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
    
    def __init__(
        self,
        model_name: str = None,
        n_clusters: int = 6,
        cache_dir: str = None,
        min_content_length: int = 5,
        use_gpu: bool = True
    ):
        """
        初始化语义分析器。
        
        参数:
            model_name: SentenceTransformer 模型名称
            n_clusters: 聚类数量（默认 6）
            cache_dir: 向量缓存目录
            min_content_length: 最小文本长度（过短的消息不参与分析）
            use_gpu: 是否使用 GPU 加速（自动检测可用性）
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.n_clusters = n_clusters
        self.min_content_length = min_content_length
        self.use_gpu = use_gpu
        
        # 检测 GPU 可用性
        self.device = self._detect_device()
        
        # 缓存目录
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "wechat-analyze"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型延迟加载
        self._model = None
    
    def _detect_device(self) -> str:
        """检测可用计算设备（GPU/CPU）"""
        if not self.use_gpu:
            return "cpu"
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   🎮 检测到 GPU: {gpu_name}")
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("   🍎 检测到 Apple Silicon GPU")
                return "mps"
            else:
                print("   💻 未检测到 GPU，使用 CPU")
                return "cpu"
        except ImportError:
            print("   ⚠️ 未安装 PyTorch，使用 CPU")
            return "cpu"
    
    @property
    def model(self):
        """延迟加载 SentenceTransformer 模型。"""
        if self._model is None:
            print(f"   📦 正在加载向量模型 ({self.model_name})...")
            print("   ⏳ 首次运行可能需要下载模型（约 67MB）...")
            
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                print(f"   ✓ 模型加载完成 (设备: {self.device.upper()})")
            except Exception as e:
                raise RuntimeError(f"模型加载失败: {e}")
        
        return self._model
    
    def analyze(
        self,
        df: pd.DataFrame,
        use_cache: bool = True
    ) -> Dict:
        """
        执行完整的语义分析流程。
        
        参数:
            df: 消息数据 DataFrame（需包含 user, content 列）
            use_cache: 是否使用向量缓存
            
        返回:
            包含聚类结果和可视化数据的字典
        """
        # 过滤有效消息
        valid_df = self._filter_valid_messages(df)
        
        if len(valid_df) < self.n_clusters * 2:
            print(f"   ⚠️ 有效消息数量不足（{len(valid_df)}条），跳过语义分析")
            return self._empty_result()
        
        print(f"   ✓ 有效消息: {len(valid_df)} 条")
        
        # Step 1: 向量化
        contents = valid_df['content'].tolist()
        embeddings = self._get_embeddings(contents, use_cache)
        
        # Step 2: 聚类
        cluster_labels, cluster_centers = self._cluster(embeddings)
        
        # Step 3: 降维 (仅用于可视化，数据量大时采样)
        MAX_VIS_SAMPLES = 3000
        n_samples = len(embeddings)
        
        if n_samples > MAX_VIS_SAMPLES:
            print(f"   📉 数据量较大 ({n_samples})，随机采样 {MAX_VIS_SAMPLES} 条用于可视化...")
            np.random.seed(42)
            vis_indices = np.random.choice(n_samples, MAX_VIS_SAMPLES, replace=False)
            vis_embeddings = embeddings[vis_indices]
        else:
            vis_indices = np.arange(n_samples)
            vis_embeddings = embeddings
            
        coords_2d = self._reduce_dimensions(vis_embeddings)
        
        # 构建结果
        result = self._build_result(
            valid_df, embeddings, cluster_labels, cluster_centers, 
            coords_2d, vis_indices
        )
        
        return result
    
    def _filter_valid_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤有效消息（长度 > min_content_length）。"""
        mask = df['content'].str.len() > self.min_content_length
        return df[mask].reset_index(drop=True)
    
    def _get_embeddings(
        self,
        contents: List[str],
        use_cache: bool = True
    ) -> np.ndarray:
        """
        获取文本向量，支持缓存。
        
        参数:
            contents: 文本列表
            use_cache: 是否使用缓存
            
        返回:
            向量矩阵 (n_samples, n_features)
        """
        # 计算内容哈希用于缓存
        cache_key = self._compute_cache_key(contents)
        cache_path = self.cache_dir / f"vectors_{cache_key}.pkl"
        
        # 尝试加载缓存
        if use_cache and cache_path.exists():
            print("   📁 加载已缓存的向量数据...")
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    cached_embeddings = cached.get('embeddings')
                    if (cached.get('model') == self.model_name and 
                        cached_embeddings is not None and 
                        len(cached_embeddings) == len(contents)):
                        print(f"   ✓ 已加载 {len(contents)} 条消息的向量缓存")
                        return cached_embeddings
                    else:
                        print(f"   ⚠️ 缓存失效 (模型或数量不匹配): 缓存={len(cached_embeddings) if cached_embeddings is not None else 0}, 当前={len(contents)}")
            except Exception as e:
                print(f"   ⚠️ 读取缓存出错: {e}")

        # 计算向量
        print("   🔢 正在计算文本向量...")
        embeddings = self._encode_with_progress(contents)
        
        # 保存缓存
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'model': self.model_name,
                        'embeddings': embeddings
                    }, f)
                print(f"   💾 向量已缓存到 {cache_path.name}")
            except Exception as e:
                print(f"   ⚠️ 写入缓存出错: {e}")
        
        return embeddings
    
    def _encode_with_progress(self, contents: List[str]) -> np.ndarray:
        """带进度条的向量编码。"""
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(contents), batch_size), desc="   向量化进度"):
            batch = contents[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def _cluster(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-Means 聚类。
        
        返回:
            (cluster_labels, cluster_centers)
        """
        print(f"   🎯 正在进行 K-Means 聚类 (k={self.n_clusters})...")
        
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(embeddings)
        
        print("   ✓ 聚类完成")
        return labels, kmeans.cluster_centers_
    
    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """
        t-SNE 降维至 2D。
        
        返回:
            2D 坐标矩阵 (n_samples, 2)
        """
        print("   📐 正在进行 t-SNE 降维...")
        
        # 根据样本数量调整 perplexity
        n_samples = embeddings.shape[0]
        perplexity = min(30, max(5, n_samples // 5))
        
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000  # 新版 scikit-learn 使用 max_iter
        )
        coords = tsne.fit_transform(embeddings)
        
        print("   ✓ 降维完成")
        return coords
    
    def _build_result(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        coords: np.ndarray,
        vis_indices: np.ndarray
    ) -> Dict:
        """构建分析结果。"""
        # 散点图数据 (基于采样后的 indices)
        scatter_data = []
        for i, original_idx in enumerate(vis_indices):
            scatter_data.append({
                'x': float(coords[i, 0]),
                'y': float(coords[i, 1]),
                'cluster_id': int(labels[original_idx]),
                'content': df.iloc[original_idx]['content'][:100],  # 截断过长内容
                'user': df.iloc[original_idx]['user']
            })
        
        # 每个聚类的代表性消息（距离中心最近的消息）- 使用全量数据计算
        cluster_representatives = self._find_representatives(
            df, embeddings, labels, centers
        )
        
        # 聚类统计 - 使用全量数据
        cluster_stats = []
        for c in range(self.n_clusters):
            cluster_mask = labels == c
            cluster_stats.append({
                'cluster_id': c,
                'count': int(cluster_mask.sum()),
                'name': f'话题 {c + 1}'  # 默认名称，后续由 AI 命名
            })
        
        return {
            'scatter_data': scatter_data,
            'cluster_representatives': cluster_representatives,
            'cluster_stats': cluster_stats,
            'n_clusters': self.n_clusters,
            'total_analyzed': len(df)
        }
    
    def _find_representatives(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        n_reps: int = 10
    ) -> Dict[int, List[Dict]]:
        """找出每个聚类的代表性消息（距离中心最近）。"""
        representatives = {}
        
        for c in range(self.n_clusters):
            cluster_mask = labels == c
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                representatives[c] = []
                continue
            
            # 计算到中心的距离
            cluster_embeddings = embeddings[cluster_mask]
            distances = np.linalg.norm(cluster_embeddings - centers[c], axis=1)
            
            # 选择最近的 n_reps 条
            n_select = min(n_reps, len(cluster_indices))
            nearest_indices = np.argsort(distances)[:n_select]
            
            reps = []
            for idx in nearest_indices:
                orig_idx = cluster_indices[idx]
                reps.append({
                    'content': df.iloc[orig_idx]['content'],
                    'user': df.iloc[orig_idx]['user'],
                    'distance': float(distances[idx])
                })
            
            representatives[c] = reps
        
        return representatives
    
    def _compute_cache_key(self, contents: List[str]) -> str:
        """计算内容列表的哈希值用于缓存。"""
        # 使用 长度 + 前100条 + 后100条 组合计算哈希，兼顾速度和准确性
        combined = str(len(contents)) + ''.join(contents[:100]) + ''.join(contents[-100:])
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def _empty_result(self) -> Dict:
        """返回空结果。"""
        return {
            'scatter_data': [],
            'cluster_representatives': {},
            'cluster_stats': [],
            'n_clusters': 0,
            'total_analyzed': 0
        }


# 使用示例
if __name__ == '__main__':
    import sys
    from data_loader import load_chat_data
    
    if len(sys.argv) < 2:
        print("用法: python vector_engine.py <json_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        df, session = load_chat_data(file_path)
        print(f"\n=== 向量分析测试 ===")
        print(f"消息总数: {len(df)}")
        
        analyzer = SemanticAnalyzer(n_clusters=5)
        result = analyzer.analyze(df)
        
        print(f"\n=== 分析结果 ===")
        print(f"分析消息数: {result['total_analyzed']}")
        print(f"聚类数量: {result['n_clusters']}")
        
        for stat in result['cluster_stats']:
            print(f"  - {stat['name']}: {stat['count']} 条消息")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
