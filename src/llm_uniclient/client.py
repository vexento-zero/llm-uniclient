import json
import requests
from typing import Dict, List, Union, Optional
from openai import OpenAI
from rich_logger import logger


class UniLLMClient:
    def __init__(
        self,
        provider: str = "siliconflow",
        model_name: str = "qwen3-235b-a22b",
        port: int = 8000,
        api_key: Optional[str] = None,
        base_url: str = None,
    ):
        """
        Unified LLM Client with Support for OpenAI SDK, PoloAI, and Azure Responses API.
        Enhanced for n > 1 support and client-side emulation.
        """
        self.provider = provider
        self.model = model_name
        self.port = port
        self.api_key = api_key
        self.base_url = base_url
        self.load_provider(self.provider, self.api_key)

    def load_provider(self, provider: str, api_key: str):
        """Initialize client instances for different providers"""
        if not api_key and provider != "vllm":
            logger.error(f"Provider '{provider}' requires an API Key")
            return

        if provider in ["siliconflow", "deepseek", "vllm", "ark", "ark_code_plan"]:
            base_urls = {
                "siliconflow": "https://api.siliconflow.cn/v1",
                "deepseek": "https://api.deepseek.com/v1",
                "vllm": f"http://localhost:{self.port}/v1",
                "ark_code_plan": "https://ark.cn-beijing.volces.com/api/coding/v3",
                "ark": "https://ark.cn-beijing.volces.com/api/v3",
            }
            self.client = OpenAI(
                api_key=api_key or "no-key", base_url=base_urls[provider], timeout=3600.0
            )
        elif provider == "poloapi":
            self.base_url = "https://poloapi.top/v1/chat/completions"
            self.headers = {
                "Authorization": api_key,
                "Content-Type": "application/json",
            }
        elif provider == "azure":
            assert self.base_url is not None
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        return_dict: bool = False,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Dict, List[Union[str, Dict]]]:
        """
        Main entry point. Automatically handles n > 1.
        If provider doesn't support n > 1, it emulates it via multiple calls.
        """
        n = kwargs.get("n", 1)

        # 检查是否需要客户端模拟 (SiliconFlow 和 DeepSeek 官方通常只支持 n=1)
        if n > 1 and self.provider in ["siliconflow", "deepseek", "ark", "ark_code_plan"]:
            results = []
            kwargs_single = kwargs.copy()
            kwargs_single["n"] = 1
            for i in range(n):
                res = self._dispatch_chat(
                    messages, return_dict, stream, **kwargs_single
                )
                if res:
                    results.append(res)
            return results

        # 否则尝试原生调用
        return self._dispatch_chat(messages, return_dict, stream, **kwargs)

    def complete(
        self,
        prompt: str,
        return_dict: bool = False,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Dict, List[Union[str, Dict]]]:
        """
        Text completion entry point. Automatically handles n > 1.
        If provider doesn't support n > 1, it emulates it via multiple calls.
        """
        n = kwargs.get("n", 1)

        # 检查是否需要客户端模拟 (SiliconFlow 和 DeepSeek 官方通常只支持 n=1)
        if n > 1 and self.provider in ["siliconflow", "deepseek", "ark", "ark_code_plan"]:
            results = []
            kwargs_single = kwargs.copy()
            kwargs_single["n"] = 1
            for i in range(n):
                res = self._dispatch_complete(
                    prompt, return_dict, stream, **kwargs_single
                )
                if res:
                    results.append(res)
            return results

        # 否则尝试原生调用
        return self._dispatch_complete(prompt, return_dict, stream, **kwargs)

    def _dispatch_chat(self, messages, return_dict, stream, **kwargs):
        """内部路由方法"""
        if self.provider in ["siliconflow", "vllm", "deepseek", "ark", "ark_code_plan"]:
            return self._chat_openai_client(messages, return_dict, stream, **kwargs)
        elif self.provider == "poloapi":
            return self._call_poloai_client(messages, return_dict, stream, **kwargs)
        elif self.provider == "azure":
            return self._call_azure_client(messages, return_dict, stream, **kwargs)
        return None

    def _dispatch_complete(self, prompt, return_dict, stream, **kwargs):
        """内部补全路由方法"""
        if self.provider in ["siliconflow", "vllm", "deepseek", "ark", "ark_code_plan"]:
            return self._complete_openai_client(prompt, return_dict, stream, **kwargs)
        elif self.provider == "poloapi":
            logger.error("PoloAI provider does not support completion API")
            return None
        elif self.provider == "azure":
            logger.error("Azure provider does not support completion API")
            return None
        return None

    def _chat_openai_client(self, messages, return_dict, stream, **kwargs):
        """Standard OpenAI SDK handler (Handles n > 1)"""
        try:
            # logger.info(f"kwargs = {json.dumps(kwargs, indent=4)}")
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=stream, **kwargs
            )
            if not stream:
                results = []
                for choice in response.choices:
                    msg = choice.message
                    reasoning = getattr(msg, "reasoning_content", None)
                    res = (
                        {"reasoning_content": reasoning or "", "content": msg.content}
                        if return_dict
                        else msg.content
                    )
                    results.append(res)
                return results if len(results) > 1 else results[0]

            return self._handle_sdk_stream(response, return_dict, n=kwargs.get("n", 1))
        except Exception as e:
            logger.error(f"OpenAI SDK error: {e}")
            return None

    def _complete_openai_client(self, prompt, return_dict, stream, **kwargs):
        """Standard OpenAI SDK completion handler (Handles n > 1)"""
        try:
            response = self.client.completions.create(
                model=self.model, prompt=prompt, stream=stream, **kwargs
            )
            if not stream:
                results = []
                for choice in response.choices:
                    res = (
                        {"content": choice.text}
                        if return_dict
                        else choice.text
                    )
                    results.append(res)
                return results if len(results) > 1 else results[0]

            return self._handle_complete_sdk_stream(response, return_dict, n=kwargs.get("n", 1))
        except Exception as e:
            logger.error(f"OpenAI SDK completion error: {e}")
            return None

    def _call_poloai_client(self, messages, return_dict, stream, **kwargs):
        """Requests handler for PoloAI (Handles n > 1)"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }
        # try:
        response = requests.post(
            self.base_url, json=payload, headers=self.headers, stream=stream
        )
        if not stream:
            res_json = response.json()
            # logger.info(f"res_json = {res_json}")
            results = []
            for choice in res_json.get("choices", []):
                msg = choice.get("message", {})
                res = (
                    {
                        "reasoning_content": msg.get("reasoning_content", ""),
                        "content": msg.get("content", ""),
                    }
                    if return_dict
                    else msg.get("content", "")
                )
                results.append(res)
            return results if len(results) > 1 else results[0]

        return self._handle_requests_stream(
            response, return_dict, mode="standard", n=kwargs.get("n", 1)
        )
        # except Exception as e:
        #     logger.error(f"PoloAI error: {e}")
        #     return None

    def _call_azure_client(self, messages, return_dict, stream, **kwargs):
        """Azure OpenAI Responses API (Handles n > 1 output items)"""
        max_tokens = (
            kwargs.get("max_completion_tokens")
            or kwargs.get("max_output_tokens")
            or 16384
        )
        payload = {
            "model": self.model,
            "input": messages,
            "stream": stream,
            "max_output_tokens": max_tokens,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "max_completion_tokens",
                    "max_output_tokens",
                    "stream",
                    "input",
                    "messages",
                ]
            },
        }

        try:
            response = requests.post(
                self.base_url, json=payload, headers=self.headers, stream=stream
            )
            if response.status_code != 200:
                logger.error(f"Azure API Error: {response.status_code} {response.text}")
                return None

            if not stream:
                res_json = response.json()
                results = []
                for item in res_json.get("output", []):
                    if item.get("type") == "message":
                        content = "".join(
                            [
                                p.get("text", "")
                                for p in item.get("content", [])
                                if p.get("type") == "output_text"
                            ]
                        )
                        res = (
                            {"reasoning_content": "", "content": content}
                            if return_dict
                            else content
                        )
                        results.append(res)
                return results if len(results) > 1 else results[0]

            return self._handle_requests_stream(
                response, return_dict, mode="azure", n=kwargs.get("n", 1)
            )
        except Exception as e:
            logger.error(f"Azure error: {e}")
            return None

    def _handle_sdk_stream(self, response, return_dict, n=1):
        """Streams with support for multiple choices (index indexing)"""
        contents, reasonings = {}, {}

        with logger.stream(level="INFO", function="UniLLM") as s:
            for chunk in response:
                if not chunk.choices:
                    continue
                for choice in chunk.choices:
                    idx = choice.index
                    delta = choice.delta

                    if idx not in contents:
                        contents[idx] = ""
                    if idx not in reasonings:
                        reasonings[idx] = ""

                    reasoning = getattr(delta, "reasoning_content", None)
                    content = getattr(delta, "content", None)

                    if reasoning:
                        reasonings[idx] += reasoning
                        if idx == 0:
                            s.update(
                                reasoning, style="dim white"
                            )  # 默认只显示主路径推理

                    if content:
                        contents[idx] += content
                        if idx == 0:
                            s.update(
                                content, style="bold white"
                            )  # 默认只显示主路径回答

        results = []
        for i in range(max(list(contents.keys()) + [n - 1]) + 1):
            c, r = contents.get(i, ""), reasonings.get(i, "")
            results.append({"reasoning_content": r, "content": c} if return_dict else c)

        return results if n > 1 else results[0]

    def _handle_complete_sdk_stream(self, response, return_dict, n=1):
        """Streams completion responses with support for multiple choices"""
        contents = {}

        with logger.stream(level="INFO", function="UniLLM") as s:
            for chunk in response:
                if not chunk.choices:
                    continue
                for choice in chunk.choices:
                    idx = choice.index
                    text = choice.text

                    if idx not in contents:
                        contents[idx] = ""

                    if text:
                        contents[idx] += text
                        if idx == 0:
                            s.update(text, style="bold white")

        results = []
        for i in range(max(list(contents.keys()) + [n - 1]) + 1):
            c = contents.get(i, "")
            results.append({"content": c} if return_dict else c)

        return results if n > 1 else results[0]

    def _handle_requests_stream(self, response, return_dict, mode="standard", n=1):
        """Requests stream with support for multiple indices"""
        contents, reasonings = {}, {}

        with logger.stream(level="INFO", function="UniLLM") as s:
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        idx, content, reasoning = 0, "", ""

                        if mode == "standard":
                            choices = data_json.get("choices", [])
                            if choices:
                                choice = choices[0]
                                idx = choice.get("index", 0)
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")
                                reasoning = delta.get("reasoning_content", "")
                        elif mode == "azure":
                            # Azure 模式通常单路输出，如有多路需适配其 unique_id
                            content = (
                                data_json.get("delta", "")
                                if data_json.get("type") == "response.output_text.delta"
                                else ""
                            )

                        if idx not in contents:
                            contents[idx] = ""
                        if idx not in reasonings:
                            reasonings[idx] = ""

                        if reasoning:
                            reasonings[idx] += reasoning
                            if idx == 0:
                                s.update(reasoning, style="dim white")
                        if content:
                            contents[idx] += content
                            if idx == 0:
                                s.update(content, style="bold white")
                    except:
                        continue

        results = []
        max_idx = max(list(contents.keys()) + [n - 1]) if (contents or n > 1) else 0
        for i in range(max_idx + 1):
            c, r = contents.get(i, ""), reasonings.get(i, "")
            results.append({"reasoning_content": r, "content": c} if return_dict else c)

        return results if n > 1 else results[0]

    def __del__(self):
        if hasattr(self, 'client') and self.client is not None:
            try:
                self.client.close()
            except:
                pass