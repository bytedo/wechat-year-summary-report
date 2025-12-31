# 🔍 微信群聊分析工具

> 基于 Python 的本地命令行工具，用于分析微信聊天记录 JSON 文件，生成包含统计图表和 AI 深度分析的精美 HTML 报告。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/ECharts-5.x-red?logo=apache-echarts&logoColor=white" alt="ECharts 5.x">
  <img src="https://img.shields.io/badge/TailwindCSS-3.x-06B6D4?logo=tailwindcss&logoColor=white" alt="TailwindCSS">
  <img src="https://img.shields.io/badge/AI-DeepSeek%2FOpenAI-yellow" alt="AI Powered">
</p>

---

## 📑 目录

- [✨ 功能特性](#-功能特性)
- [🚀 快速开始](#-快速开始)
- [📖 使用指南](#-使用指南)
- [📁 数据格式](#-数据格式)
- [🏗️ 项目结构](#️-项目结构)
- [🔧 技术栈](#-技术栈)
- [🔒 隐私保护](#-隐私保护)
- [⚡ 性能优化](#-性能优化)
- [💡 最佳实践](#-最佳实践)
- [🐛 常见问题](#-常见问题)
- [📝 开发计划](#-开发计划)

---

## 🎯 核心亮点

🤖 **AI 驱动**：接入 DeepSeek/OpenAI 等 LLM，智能生成年度总结、用户画像、金句甄选  
🧠 **语义理解**：基于 Sentence-BERT 的深度语义分析，自动识别话题聚类  
📊 **可视化**：ECharts 交互式图表 + t-SNE 语义银河，数据一目了然  
📱 **移动优化**：海报式报告，支持滑动浏览、背景音乐，分享朋友圈更有格调  
⚡ **智能缓存**：AI 分析结果自动缓存，重复分析秒级完成  
🔒 **隐私保护**：本地处理 + 数据脱敏 + 采样策略，保护个人隐私

---

## ✨ 功能特性

### 📊 统计分析

- **基础指标**：消息总数、参与人数、时间跨度
- **用户活跃榜**：Top 10 话痨排行榜
- **时间分析**：每日消息趋势、24 小时活跃分布
- **高频词云**：基于 jieba 分词 + 停用词过滤

### 🧠 深度语义分析

- **话题聚类**：基于 Sentence-BERT 向量语义分析
- **智能命名**：AI 自动为聚类生成描述性话题名称
- **话题银河**：t-SNE 降维可视化，呈现消息语义分布
- **消息关联**：识别语义相近的对话内容

### 🤖 AI 智能洞察

- **群聊画像**：年度关键词 & 氛围总结
- **MBTI 用户画像**：Top 活跃用户性格分析 + MBTI 类型预测
- **周度深度分析**：按周批次分析消息，生成连贯的年度总结
- **话题总结**：自动归纳每月讨论主题与回忆
- **金句甄选**：AI 智能挑选年度精彩发言与入选理由
- **分析缓存**：智能缓存 AI 分析结果，加速重复处理

### ⏳ 怀旧数据挖掘

- **巅峰日**：最活跃的一天及 AI 摘要
- **年度初心**：成员首条消息回顾
- **热门消息**：获得最多回复的发言

### 📱 精美报告

- **海报式报告**：移动端优化，支持 Swiper 滑动浏览、背景音乐
- **语义银河可视化**：ECharts 散点图展示话题分布

---

## 🚀 快速开始

### 环境要求

- Python 3.10 或更高版本
- pip 包管理器

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd wechat-analyze

# 安装依赖
pip install -r requirements.txt
```

### 配置 AI（可选）

复制环境变量示例文件并配置 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```ini
# LLM API 配置（支持 OpenAI / DeepSeek / Moonshot 等兼容接口）

# API 基础地址
LLM_BASE_URL=https://api.deepseek.com/v1

# API 密钥
LLM_API_KEY=your-api-key-here

# 模型名称（可选，默认: deepseek-chat）
LLM_MODEL=deepseek-chat
```

> 💡 如未配置 API Key，工具将自动切换到 Mock 模式，使用模拟数据生成报告。

---

## 📖 使用指南

### 基本用法

```bash
python main.py data/chat_export.json
```

### 命令行参数

| 参数            | 说明                         | 默认值   |
| --------------- | ---------------------------- | -------- |
| `input`         | 微信群聊导出的 JSON 文件路径 | (必填)   |
| `-o, --output`  | 输出目录                     | `output` |
| `--no-ai`       | 跳过 AI 分析，仅生成统计报告 | -        |
| `--no-vector`   | 跳过向量语义分析（加速处理） | -        |
| `--no-gpu`      | 禁用 GPU 加速，强制使用 CPU  | -        |
| `--clusters`    | 话题聚类数量                 | `6`      |
| `--mock`        | 强制使用 AI Mock 模式        | -        |
| `--poster`      | 生成海报式报告（移动端优化） | -        |
| `--music`       | 报告的背景音乐 URL           | -        |
| `-v, --verbose` | 显示详细输出                 | -        |

### 使用示例

```bash
# 标准模式：完整分析
python main.py data/qun.json

# 指定输出目录
python main.py data/qun.json -o reports/2025/

# 跳过 AI 分析（快速模式，仅统计图表）
python main.py data/qun.json --no-ai

# 跳过向量分析（加速处理，不生成语义银河）
python main.py data/qun.json --no-vector

# 自定义话题聚类数量
python main.py data/qun.json --clusters 8

# 强制使用 CPU（禁用 GPU 加速）
python main.py data/qun.json --no-gpu

# 添加背景音乐（支持 MP3/M4A 等格式的 URL）
python main.py data/qun.json --music "https://example.com/bgm.mp3"

# 完整模式 + 详细日志
python main.py data/qun.json -v

# 组合参数：快速模式 + 详细输出
python main.py data/qun.json --no-vector --no-ai -v

# 使用 Mock 模式测试（无需 API Key）
python main.py data/qun.json --mock
```

---

## 📁 数据格式

工具接受特定格式的 JSON 文件，需要使用 [EchoTrace](https://github.com/ycccccccy/echotrace) 导出微信聊天记录：

```json
{
  "session": {
    "wxid": "43778531350@chatroom",
    "displayName": "群聊名称",
    "messageCount": 197
  },
  "messages": [
    {
      "localId": 6,
      "createTime": 1743570874,
      "formattedTime": "2025-04-02 13:14:34",
      "type": "文本消息",
      "content": "消息内容...",
      "senderUsername": "wxid_xxxxxx",
      "senderDisplayName": "发送者昵称",
      "isSend": 1
    }
  ]
}
```

### 支持的消息类型

- `文本消息` - 普通文本
- `引用消息` - 回复消息

> 其他类型（红包、图片、语音等）会在分析时自动过滤。

---

## 🏗️ 项目结构

```
wechat-analyze/
├── main.py                 # 主程序入口 & AI 分析调度
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量示例
│
├── src/                    # 核心源码
│   ├── data_loader.py      # 数据加载与清洗
│   ├── stats_engine.py     # 统计分析引擎
│   ├── vector_engine.py    # 向量语义分析 (Sentence-BERT + t-SNE)
│   ├── poster_builder.py   # 海报报告生成器
│   │
│   ├── ai/                 # AI 分析模块（模块化设计）
│   │   ├── __init__.py     # AIAnalyzer 主入口
│   │   ├── base.py         # 基础配置 & Mock 模式
│   │   ├── weekly.py       # 周度深度分析
│   │   ├── monthly.py      # 月度话题回忆
│   │   ├── user_profile.py # 用户画像 & MBTI 分析
│   │   ├── keywords.py     # 年度关键词提炼
│   │   ├── golden_quotes.py# 金句甄选
│   │   └── cluster_naming.py# 话题聚类命名
│   │
│   └── analyzers/          # 扩展分析器
│       ├── monthly_analyzer.py # 月度数据处理
│       ├── weekly_analyzer.py  # 周度数据处理
│       └── yearly_analyzer.py  # 年度数据处理
│
├── templates/              # HTML 模板（Jinja2）
│   └── poster/             # 海报式报告模板
│       ├── index.html      # 主模板
│       └── components/     # 组件模板
│
├── data/                   # 数据目录
│   └── stopwords.txt       # 中文停用词表
│
├── assets/                 # 静态资源
│   ├── audio/              # 音频文件
│   └── fonts/              # 字体文件
│
├── .cache/                 # 模型缓存（自动生成）
│   ├── huggingface/        # HuggingFace 模型
│   └── models/             # Sentence-BERT 模型
│
└── output/                 # 输出目录（自动生成）
    └── tmp/                # AI 分析缓存
```

---

## 🔧 技术栈

### 核心架构

```
数据加载 → 统计分析 → 向量语义分析 → AI 深度分析 → 报告生成
   ↓           ↓              ↓              ↓            ↓
Pandas    jieba分词     Sentence-BERT    OpenAI API   Jinja2模板
          词频统计      DBSCAN聚类       DeepSeek     ECharts可视化
                        t-SNE降维        GPT-4        Swiper交互
```

### 后端依赖

| 依赖                    | 版本要求 | 用途                                                  |
| ----------------------- | -------- | ----------------------------------------------------- |
| `pandas`                | >=2.0.0  | 数据清洗、转换与统计分析                              |
| `jieba`                 | >=0.42.1 | 中文分词、关键词提取                                  |
| `openai`                | >=1.0.0  | LLM API 调用（兼容多种服务商）                        |
| `Jinja2`                | >=3.1.0  | HTML 模板渲染引擎                                     |
| `sentence-transformers` | >=2.2.0  | 语义向量嵌入（paraphrase-multilingual-MiniLM-L12-v2） |
| `scikit-learn`          | >=1.3.0  | DBSCAN 聚类、t-SNE 降维                               |
| `python-dotenv`         | >=1.0.0  | 环境变量管理                                          |
| `tqdm`                  | >=4.66.0 | 进度条显示                                            |
| `numpy`                 | >=1.24.0 | 数值计算                                              |

**可选依赖**：

- `torch` (CUDA 版本)：GPU 加速向量计算

### 前端技术（CDN 引入）

| 依赖         | 版本 | 用途            |
| ------------ | ---- | --------------- |
| TailwindCSS  | 3.x  | 原子化 CSS 框架 |
| ECharts      | 5.x  | 数据可视化图表  |
| Swiper.js    | 11.x | 移动端滑动交互  |
| Marked.js    | 12.x | Markdown 渲染   |
| Font Awesome | 6.x  | 图标库          |

### AI 模型

- **语义分析**：`paraphrase-multilingual-MiniLM-L12-v2`

  - 384 维向量
  - 支持 100+ 种语言
  - 约 118M 参数
  - 推理速度：~50-100 句/秒 (CPU)

- **LLM 服务**：兼容 OpenAI API 格式
  - 推荐：DeepSeek Chat (性价比高)
  - 备选：GPT-4, GPT-3.5-turbo, Moonshot, GLM-4 等

### 数据流架构

```
JSON 原始数据
    ↓
data_loader.py (数据清洗 & 隐私脱敏)
    ↓
stats_engine.py (统计分析 & 词频统计)
    ↓
vector_engine.py (语义向量化 & 聚类)
    ↓
src/ai/* (AI 深度分析 & 智能缓存)
    ├── weekly.py      → 周度总结
    ├── monthly.py     → 月度话题
    ├── user_profile.py → 用户画像
    ├── keywords.py    → 关键词提炼
    └── golden_quotes.py → 金句甄选
    ↓
poster_builder.py (模板渲染)
    ↓
HTML 报告输出
```

---

## 🔒 隐私保护

工具内置隐私保护机制：

- **手机号脱敏**：自动将手机号替换为 `138****0000` 格式
- **本地处理**：所有数据在本地处理，不上传云端
- **采样策略**：发送给 AI 的数据经过采样，避免全量数据泄露

---

## ⚡ 性能优化

### 智能缓存机制

工具内置了多层缓存策略，大幅提升重复分析速度：

#### 1. **模型缓存**（`.cache/` 目录）

- Sentence-BERT 模型（约 400MB）
- HuggingFace 预训练模型
- 首次下载后永久缓存

#### 2. **AI 分析缓存**（`output/tmp/` 目录）

- 周度深度分析
- 月度话题回忆
- 用户画像 & MBTI
- 话题聚类命名
- 金句甄选

**缓存标识格式**：`{source_filename}_{content_hash}.json`

只要源文件内容不变，所有 AI 分析结果都会从缓存加载，避免重复调用 API。

### 加速策略

| 场景       | 建议参数              | 说明                        |
| ---------- | --------------------- | --------------------------- |
| 快速预览   | `--no-ai --no-vector` | 仅统计分析，10 秒内完成     |
| 重复分析   | 无需额外参数          | 自动使用缓存，秒级完成      |
| 大规模群聊 | `--no-gpu`            | 避免 GPU 内存溢出           |
| 调试阶段   | `--mock`              | 使用模拟数据，无需 API 调用 |

### GPU 加速

支持 CUDA 加速向量计算（Sentence-BERT 编码）：

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证 GPU 可用性
python -c "import torch; print(f'GPU 可用: {torch.cuda.is_available()}')"
```

**加速效果**：

- CPU（8 核）：约 2-5 条消息/秒
- GPU（RTX 3060）：约 20-50 条消息/秒

---

## 💡 最佳实践

### 1. 首次使用

```bash
# Step 1: 克隆项目
git clone https://github.com/bytedo/wechat-year-summary-report.git
cd wechat-year-summary-report

# Step 2: 安装依赖
pip install -r requirements.txt

# Step 3: 配置环境变量（可选）
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key

# Step 4: 使用 Mock 模式测试
python main.py data/sample.json --mock -v
```

### 2. 日常使用

```bash
# 标准分析（推荐）
python main.py data/group_2025.json

# 快速预览（跳过耗时分析）
python main.py data/group_2025.json --no-vector

# 完整分析 + GPU 加速
python main.py data/group_2025.json -v
```

### 3. 批量处理

```bash
# 批量分析多个群聊
for file in data/*.json; do
    echo "正在分析: $file"
    python main.py "$file" -o "output/$(basename $file .json)/"
done
```

### 4. 生产环境

```bash
# 使用 DeepSeek（性价比高）
# .env 配置:
# LLM_BASE_URL=https://api.deepseek.com/v1
# LLM_API_KEY=your-key
# LLM_MODEL=deepseek-chat

# 运行完整分析
python main.py data/important_group.json --music "url-to-bgm.mp3" -v
```

### 5. 清理缓存

```bash
# 清除 AI 分析缓存（保留模型）
rm -rf output/tmp/

# 完全清理（需重新下载模型）
rm -rf .cache/
```

---

## 🐛 常见问题

### Q: 首次运行很慢？

**A:** 首次运行需要下载 Sentence-BERT 模型（约 400MB），模型会缓存到 `.cache/` 目录。后续运行会直接使用缓存，速度更快。

可以使用 `--no-vector` 跳过向量分析加速处理。

### Q: 如何获取微信聊天记录 JSON 文件？

**A:** 本工具需要使用 [EchoTrace](https://github.com/ycccccccy/echotrace) 导出微信聊天记录。EchoTrace 支持提取 iOS 微信备份文件中的聊天记录并导出为 JSON 格式。

### Q: 如何跳过 AI 分析？

**A:** 使用 `--no-ai` 参数，工具将只生成统计图表报告，不调用 LLM API。

### Q: AI 分析结果会缓存吗？

**A:** 会！所有 AI 分析结果（周度分析、月度话题、用户画像等）都会缓存到 `output/tmp/` 目录。重复分析同一文件时会自动加载缓存，大幅提升速度。

缓存使用 `{source_filename}_{content_hash}` 作为标识，只要源文件内容不变，就会命中缓存。

### Q: GPU 加速不工作？

**A:**

1. 确保已安装支持 CUDA 的 PyTorch：

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. 检查 GPU 是否可用：

   ```python
   import torch
   print(torch.cuda.is_available())  # 应该输出 True
   ```

3. 如果没有 GPU 或不需要加速，使用 `--no-gpu` 强制使用 CPU。

### Q: 报告在电脑上显示异常？

**A:** 海报式报告专为移动端优化（竖屏 9:16 比例），建议在手机上查看。

在电脑上查看时，使用浏览器开发者工具（F12）切换到移动端模拟器：

- Chrome: 设备工具栏 → iPhone 或其他移动设备
- 推荐分辨率：375x812 (iPhone X)

### Q: 支持哪些 LLM 服务商？

**A:** 支持所有兼容 OpenAI API 的服务商，包括：

- OpenAI (GPT-4, GPT-3.5)
- DeepSeek (推荐，性价比高)
- Moonshot (Kimi)
- 智谱 AI (GLM)
- 阿里通义千问
- 其他自建兼容服务

只需在 `.env` 中配置对应的 `LLM_BASE_URL` 和 `LLM_API_KEY` 即可。

### Q: 数据会被上传到云端吗？

**A:** 不会。除了发送采样数据给 LLM API 进行分析外，所有数据处理都在本地完成。工具内置了隐私保护机制：

- 手机号自动脱敏
- AI 分析采用采样策略，不发送全量数据
- 向量分析完全离线

### Q: 如何删除缓存？

**A:**

- 删除 `.cache/` 目录：清除模型缓存（需重新下载）
- 删除 `output/tmp/` 目录：清除 AI 分析缓存

---

## 📝 开发计划

### ✅ 已完成

**核心功能**

- [x] 基础统计分析（消息量、活跃度、时间分布）
- [x] 中文分词与词云生成
- [x] 向量语义聚类（Sentence-BERT + DBSCAN）
- [x] t-SNE 降维可视化（语义银河）
- [x] 海报式移动端报告

**AI 智能分析**

- [x] 周度深度批次分析（连贯性叙事）
- [x] 月度话题回忆生成
- [x] MBTI 用户画像分析
- [x] 年度关键词智能提炼
- [x] AI 话题聚类命名
- [x] 金句智能甄选
- [x] 巅峰日摘要生成

**性能优化**

- [x] AI 分析结果智能缓存
- [x] 模型本地化存储（.cache/）
- [x] GPU 加速支持（CUDA）
- [x] 模块化架构重构（src/ai/）

**数据安全**

- [x] 手机号自动脱敏
- [x] 本地数据处理
- [x] AI 采样策略（隐私保护）

### 🚧 进行中

- [ ] **性能提升**
  - [ ] 向量计算并行化
  - [ ] 增量分析支持
- [ ] **功能增强**
  - [ ] 表情包统计与分析
  - [ ] 互动关系网络图

### 🎯 未来规划

**短期计划（v2.0）**

- [ ] 个人年度报告模式
- [ ] 多群聊对比分析
- [ ] 自定义报告主题色
- [ ] 更多 MBTI 维度分析

**中期计划（v3.0）**

- [ ] 交互式 Web 界面
- [ ] 情感分析（正/负/中性）
- [ ] 话题演化时间线
- [ ] 自定义 Prompt 模板

**长期愿景**

- [ ] 支持 Android 微信记录
- [ ] 多人协作分析
- [ ] 实时分析服务
- [ ] 桌面客户端（Electron）

---

## 🤝 贡献指南

欢迎贡献代码、功能建议或 Bug 报告！

### 提交 Issue

请在 Issue 中包含：

- 问题描述
- 重现步骤
- 环境信息（Python 版本、操作系统）
- 错误日志（如有）

### 提交 Pull Request

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'feat: 添加某个功能'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

### 代码规范

- Python 代码遵循 PEP 8 规范
- 函数和类添加中文 docstring
- 提交信息使用中文，遵循 Conventional Commits
  - `feat: 新功能`
  - `fix: 修复 Bug`
  - `docs: 文档更新`
  - `refactor: 代码重构`
  - `perf: 性能优化`

---

## 📄 许可证

MIT License

Copyright (c) 2025 wechat-analyze contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 🙏 致谢

感谢以下开源项目和服务：

- [EchoTrace](https://github.com/ycccccccy/echotrace) - 微信聊天记录提取工具
- [jieba](https://github.com/fxsjy/jieba) - 优秀的中文分词库
- [ECharts](https://echarts.apache.org/) - 强大的数据可视化库
- [Sentence-Transformers](https://www.sbert.net/) - 语义向量嵌入框架
- [DeepSeek](https://deepseek.com/) - 高性价比 LLM API 服务
- [TailwindCSS](https://tailwindcss.com/) - 现代化 CSS 框架
- [Swiper](https://swiperjs.com/) - 移动端滑动组件

特别感谢所有贡献者和使用本工具的朋友们！

---

<p align="center">
  Made with ❤️ for WeChat Group Chat Analysis
</p>

<p align="center">
  <a href="#-功能特性">功能特性</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-使用指南">使用指南</a> •
  <a href="#-常见问题">常见问题</a>
</p>
