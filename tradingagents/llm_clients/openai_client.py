import os
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output and reasoning_content support.

    Two fixes over the base class:
    1. Normalizes response content from list-of-blocks to plain string
       (needed for OpenAI Responses API and some third-party providers).
    2. Captures reasoning_content from DeepSeek reasoning model responses and
       stores it in AIMessage.additional_kwargs, then re-injects it into
       assistant messages when building the next API request, so DeepSeek
       reasoning models don't reject multi-turn calls with:
         'The reasoning_content in the thinking mode must be passed back to the API.'
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def _create_chat_result(self, response, generation_info=None):
        """解析 API 响应，将 reasoning_content 存入 AIMessage.additional_kwargs。

        DeepSeek 推理模型在 response.choices[i].message 里返回 reasoning_content，
        但 langchain-openai 默认不把它存到 AIMessage 里，这里补上这一步。
        OpenAI SDK 用 Pydantic extra='allow'，所以字段可能在 model_extra 里。
        """
        result = super()._create_chat_result(response, generation_info)
        try:
            for i, gen in enumerate(result.generations):
                if i >= len(response.choices):
                    break
                raw_msg = response.choices[i].message
                # 先尝试直接属性，再尝试 Pydantic model_extra
                rc = getattr(raw_msg, "reasoning_content", None) or (
                    (getattr(raw_msg, "model_extra", None) or {}).get("reasoning_content")
                )
                if rc:
                    gen.message.additional_kwargs["reasoning_content"] = rc
        except Exception:
            pass
        return result

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """构建 API 请求 payload，并把 reasoning_content 补回 assistant 消息。

        DeepSeek 推理模型要求：上一轮 assistant 消息里的 reasoning_content
        必须原样传回。LangChain 把它存在 AIMessage.additional_kwargs 里但
        默认不序列化到请求 dict，这里补上这一步。
        """
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # 从 input_ 里提取原始 LangChain 消息列表
        if isinstance(input_, PromptValue):
            lc_messages = input_.to_messages()
        elif isinstance(input_, list):
            lc_messages = [m for m in input_ if isinstance(m, BaseMessage)]
        else:
            return payload

        # 和 payload 里的 messages 一一对应，注入 reasoning_content
        for lc_msg, api_msg in zip(lc_messages, payload.get("messages", [])):
            if (
                isinstance(lc_msg, AIMessage)
                and isinstance(api_msg, dict)
                and api_msg.get("role") == "assistant"
            ):
                rc = lc_msg.additional_kwargs.get("reasoning_content")
                if rc:
                    api_msg["reasoning_content"] = rc

        return payload

    def with_structured_output(self, schema, *, method=None, **kwargs):
        """Wrap with structured output, defaulting to function_calling for OpenAI.

        langchain-openai's Responses-API-parse path (the default for json_schema
        when use_responses_api=True) calls response.model_dump(...) on the OpenAI
        SDK's union-typed parsed response, which makes Pydantic emit ~20
        PydanticSerializationUnexpectedValue warnings per call. The function-calling
        path returns a plain tool-call shape that does not trigger that
        serialization, so it is the cleaner choice for our combination of
        use_responses_api=True + with_structured_output. Both paths use OpenAI's
        strict mode and produce the same typed Pydantic instance.
        """
        if method is None:
            method = "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4/", "ZHIPU_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        # Provider-specific base URL and auth
        if self.provider in _PROVIDER_CONFIG:
            base_url, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = base_url
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
