# UniLLMClient

`UniLLMClient` 是一个轻量级、统一的 LLM 客户端包装器，旨在简化与多个大模型提供商（如 SiliconFlow, DeepSeek, vLLM, PoloAI）的交互。

它特别针对 **推理型模型（Reasoning Models）** 进行了优化，支持在终端中实时流式展示“思考过程”与“最终答案”，并提供结构化的数据返回。

## ✨ 核心特性

- **多供应商统一接口**：支持 SiliconFlow, DeepSeek, vLLM, PoloAI。
- **推理/内容分离**：完美支持 `reasoning_content`（思维链），支持在控制台独立展示思考过程。
- **优雅的流式输出**：集成 `rich_logger`，在控制台提供美观的、带颜色标识的流式交互体验。
- **健壮性设计**：自动处理流式传输末尾的空数据包及 JSON 解析异常，彻底解决 `IndexError`。
- **灵活的返回格式**：通过 `return_dict` 参数切换纯文本或结构化字典。

## 📦 安装

你可以直接通过 GitHub 仓库安装此包：

```bash
pip install --no-cache-dir --force-reinstall git+https://github.com/vexento-zero/llm-uniclient.git
```

## 🚀 快速开始

```python
from llm_uniclient import UniLLMClient

# 初始化客户端
client = UniLLMClient(
    provider="deepseek",
    model_name="deepseek-reasoner",
    api_key="your_api_key_here"
)

# 基础调用（仅获取答案字符串）
answer = client.chat(
    messages=[{"role": "user", "content": "你好"}]
)
print(answer)
```

## 💡 高级用法

### 获取推理过程 (Reasoning)

对于支持思维链的模型（如 DeepSeek-R1, Qwen-2.5-7B-Instruct-1M 等），你可以开启 `return_dict`。

```python
response = client.chat(
    messages=[{"role": "user", "content": "解方程 2x + 5 = 11"}],
    return_dict=True,  # 返回字典格式，包含推理内容
    stream=True        # 开启控制台流式显示，将看到 [Reasoning] 和 [Answer] 分隔
)

# 响应格式:
# {
#     "reasoning_content": "首先我们需要把常数项移到右边...",
#     "content": "方程的解为 x = 3"
# }
```

### 使用不同供应商

```python
# SiliconFlow
client = UniLLMClient(provider="siliconflow", model_name="deepseek-ai/DeepSeek-V3", api_key="...")

# 本地 vLLM
client = UniLLMClient(provider="vllm", model_name="qwen-7b", port=8000)

# PoloAI
client = UniLLMClient(provider="poloai", model_name="gpt-4o", api_key="...")
```

## 🛠 参数参考

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `messages` | `List[Dict]` | 必填 | 符合 OpenAI 格式的消息列表 |
| `return_dict` | `bool` | `False` | 为 `True` 时返回字典；为 `False` 时仅返回内容字符串 |
| `stream` | `bool` | `False` | 是否开启流式传输并在控制台实时打印 |
| `**kwargs` | `Any` | - | 支持透传 `temperature`, `top_p`, `max_tokens` 等参数 |


## 📄 开源协议

MIT License.

---