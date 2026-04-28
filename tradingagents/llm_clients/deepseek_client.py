"""DeepSeek LLM client.

DeepSeek 推理模型（deepseek-reasoner 等）的响应中包含 reasoning_content 字段。
分析师节点在多轮工具调用循环中每次传回消息历史时必须带上这个字段，否则报错：
  'The reasoning_content in the thinking mode must be passed back to the API.'

NormalizedChatOpenAI 已在 _create_message_dicts 里统一处理了 reasoning_content
的回传，因此 DeepSeekClient 直接复用它，只需配置正确的 base_url 和 api_key。
"""

import os
from typing import Any, Optional

from .base_client import BaseLLMClient
from .openai_client import NormalizedChatOpenAI, _PASSTHROUGH_KWARGS
from .validators import validate_model


class DeepSeekClient(BaseLLMClient):
    """DeepSeek LLM 客户端，支持普通模型和推理模型（deepseek-reasoner 等）。"""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.provider = "deepseek"

    def get_llm(self) -> Any:
        """返回配置好的 NormalizedChatOpenAI 实例（指向 DeepSeek endpoint）。"""
        self.warn_if_unknown_model()

        llm_kwargs = {
            "model": self.model,
            "base_url": self.base_url or "https://api.deepseek.com",
        }

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if api_key:
            llm_kwargs["api_key"] = api_key

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("deepseek", self.model)
