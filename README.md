# 🔍 微信群聊分析工具

> 基于 Python 的本地命令行工具，用于分析微信聊天记录 JSON 文件，生成包含统计图表和 AI 深度分析的精美 HTML 报告。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/ECharts-5.x-red?logo=apache-echarts&logoColor=white" alt="ECharts 5.x">
  <img src="https://img.shields.io/badge/TailwindCSS-3.x-06B6D4?logo=tailwindcss&logoColor=white" alt="TailwindCSS">
  <img src="https://img.shields.io/badge/AI-DeepSeek%2FOpenAI-yellow" alt="AI Powered">
</p>

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
| `--music`       | 报告的背景音乐 URL           | -        |
| `-v, --verbose` | 显示详细输出                 | -        |

### 使用示例

```bash
# 生成报告
python main.py data/qun.json

# 指定输出目录
python main.py data/qun.json -o reports/

# 跳过 AI 分析（快速模式）
python main.py data/qun.json --no-ai

# 带背景音乐的报告
python main.py data/qun.json --music "https://example.com/bgm.mp3"

# 完整模式（详细输出）
python main.py data/qun.json -v
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
├── main.py                 # 主程序入口
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量示例
├── prompt_spec.md          # 项目规格说明
│
├── src/                    # 核心源码
│   ├── data_loader.py      # 数据加载与清洗
│   ├── stats_engine.py     # 统计分析引擎
│   ├── ai_analyzer.py      # AI 分析代理
│   ├── vector_engine.py    # 向量语义分析
│   ├── report_builder.py   # 标准报告生成
│   ├── poster_builder.py   # 海报报告生成
│   └── analyzers/          # 扩展分析器
│
├── templates/              # HTML 模板
│   ├── report.html         # 标准报告模板
│   └── poster/             # 海报式报告模板
│
├── data/                   # 数据目录
│   └── stopwords.txt       # 中文停用词表
│
├── assets/                 # 静态资源
│   ├── audio/              # 音频文件
│   └── fonts/              # 字体文件
│
└── output/                 # 输出目录（自动生成）
```

---

## 🔧 技术栈

### 后端

| 依赖                    | 用途           |
| ----------------------- | -------------- |
| `pandas`                | 数据清洗与分析 |
| `jieba`                 | 中文分词       |
| `openai`                | LLM API 调用   |
| `Jinja2`                | HTML 模板渲染  |
| `sentence-transformers` | 向量语义分析   |
| `scikit-learn`          | 聚类算法       |
| `python-dotenv`         | 环境变量管理   |

### 前端（CDN 引入）

| 依赖            | 用途          |
| --------------- | ------------- |
| TailwindCSS 3.x | 样式框架      |
| ECharts 5.x     | 数据可视化    |
| Swiper.js       | 滑动交互      |
| Marked.js       | Markdown 渲染 |

---

## 🔒 隐私保护

工具内置隐私保护机制：

- **手机号脱敏**：自动将手机号替换为 `138****0000` 格式
- **本地处理**：所有数据在本地处理，不上传云端
- **采样策略**：发送给 AI 的数据经过采样，避免全量数据泄露

---

## 🐛 常见问题

### Q: 首次运行很慢？

**A:** 首次运行需要下载 Sentence-BERT 模型（约 400MB），后续运行会使用缓存。可以使用 `--no-vector` 跳过向量分析加速处理。

### Q: 如何跳过 AI 分析？

**A:** 使用 `--no-ai` 参数，工具将只生成统计图表报告。

### Q: GPU 加速不工作？

**A:** 确保已安装 CUDA 版本的 PyTorch。可使用 `--no-gpu` 强制使用 CPU。

### Q: 报告在电脑上显示异常？

**A:** 报告专为移动端优化，建议在手机上竖屏查看，或使用浏览器开发者工具切换到移动端视图。

---

## 📝 开发计划

- [x] 基础统计分析
- [x] AI 深度分析
- [x] 向量语义聚类
- [x] 海报式报告
- [x] 怀旧数据挖掘
- [x] MBTI 用户画像
- [x] 语义银河可视化
- [x] AI 分析结果缓存
- [ ] 个人年度报告
- [ ] 多群聊对比分析
- [ ] 交互式网页版

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- [jieba](https://github.com/fxsjy/jieba) - 优秀的中文分词库
- [ECharts](https://echarts.apache.org/) - 强大的数据可视化库
- [Sentence-Transformers](https://www.sbert.net/) - 语义向量嵌入
- [DeepSeek](https://deepseek.com/) - 高性价比 LLM API

---

<p align="center">
  Made with ❤️ for WeChat group chat analysis
</p>
