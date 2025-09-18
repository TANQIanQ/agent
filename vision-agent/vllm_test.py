import base64
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

import requests
from openai import AzureOpenAI, OpenAI
from vision_agent.models import Message
from vision_agent.utils.agent import extract_tag
from vision_agent.utils.image_utils import encode_media


class LMM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        media: Optional[Sequence[Union[str, Path]]] = None,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        pass

    @abstractmethod
    def chat(
        self,
        chat: Sequence[Message],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        pass

    @abstractmethod
    def __call__(
        self,
        input: Union[str, Sequence[Message]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        pass


class VLLMLMM(LMM):
    r"""An LMM class for the VLLM models."""

    def __init__(
        self,
        model_name: str = "mistral-7b-instruct-v0.1",
        api_key: Optional[str] = None,
        base_url: str = "https://api.vllm.ai/v1",
        max_tokens: int = 4096,
        image_size: int = 768,
        image_detail: str = "low",
        **kwargs: Any,
    ):
        # vLLM typically uses "EMPTY" as placeholder for API key
        if not api_key:
            api_key = os.getenv("VLLM_API_KEY", "EMPTY")

        # Use OpenAI client for compatibility with vLLM's OpenAI-compatible API
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.image_size = image_size
        self.image_detail = image_detail

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = max_tokens
        self.kwargs = kwargs

    def __call__(
        self,
        input: Union[str, Sequence[Message]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        if isinstance(input, str):
            return self.generate(input, **kwargs)
        return self.chat(input, **kwargs)

    def chat(
        self,
        chat: Sequence[Message],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        """Chat with the VLLM model.

        Parameters:
            chat (Sequence[Dict[str, str]]): A list of dictionaries containing the chat
                messages. The messages can be in the format:
                [{"role": "user", "content": "Hello!"}, ...]
                or if it contains media, it should be in the format:
                [{"role": "user", "content": "Hello!", "media": ["image1.jpg", ...]}, ...]
        """
        fixed_chat = []
        for msg in chat:
            fixed_c = {"role": msg["role"]}
            fixed_c["content"] = [{"type": "text", "text": msg["content"]}]

            if "media" in msg and msg["media"] is not None:
                for media in msg["media"]:
                    resize = kwargs.get("resize", self.image_size)
                    image_detail = kwargs.get("image_detail", self.image_detail)
                    encoded_media = encode_media(cast(str, media), resize=resize)

                    fixed_c["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    encoded_media
                                    if encoded_media.startswith(("http", "https"))
                                    or encoded_media.startswith("data:image/")
                                    else f"data:image/png;base64,{encoded_media}"
                                ),
                                "detail": image_detail,
                            },
                        }
                    )
            fixed_chat.append(fixed_c)

        # Use OpenAI client for vLLM compatibility
        tmp_kwargs = self.kwargs | kwargs
        response = self.client.chat.completions.create(
            model=self.model_name, messages=fixed_chat, **tmp_kwargs
        )

        if tmp_kwargs.get("stream", False):

            def f() -> Iterator[Optional[str]]:
                for chunk in response:
                    chunk_message = chunk.choices[0].delta.content
                    yield chunk_message

            return f()
        else:
            return cast(str, response.choices[0].message.content)

    def generate(
        self,
        prompt: str,
        media: Optional[Sequence[Union[str, Path]]] = None,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        message_content = [{"type": "text", "text": prompt}]

        if media and len(media) > 0:
            for m in media:
                resize = kwargs.get("resize", self.image_size)
                image_detail = kwargs.get("image_detail", self.image_detail)
                encoded_media = encode_media(m, resize=resize)

                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                encoded_media
                                if encoded_media.startswith(("http", "https"))
                                or encoded_media.startswith("data:image/")
                                else f"data:image/png;base64,{encoded_media}"
                            ),
                            "detail": image_detail,
                        },
                    }
                )

        messages = [{"role": "user", "content": message_content}]

        # Use OpenAI client for vLLM compatibility
        tmp_kwargs = self.kwargs | kwargs
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **tmp_kwargs
        )

        if tmp_kwargs.get("stream", False):

            def f() -> Iterator[Optional[str]]:
                for chunk in response:
                    chunk_message = chunk.choices[0].delta.content
                    yield chunk_message

            return f()
        else:
            return cast(str, response.choices[0].message.content)


if __name__ == "__main__":
    import os
    from pathlib import Path
    from typing import List, Dict, Union

    API_BASE = "http://10.26.9.88:8001/v1"  # 替换为你的 vLLM 部署地址
    MODEL_NAME = "Qwen2.5-VL-72B-Instruct"  # 确认模型名称匹配你的部署
    # 定义消息类型
    Message = Dict[str, Union[str, List[str]]]

    # 创建测试实例
    vllm_lmm = VLLMLMM(
        model_name=MODEL_NAME,
        base_url=API_BASE,
        max_tokens=512,
        image_size=512,
        image_detail="high",
    )

    def test_text_chat():
        """测试纯文本聊天"""
        print("\n=== 测试纯文本聊天 ===")
        messages: List[Message] = [
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ]
        response = vllm_lmm.chat(messages)
        print(f"模型回复: {response}")

    def test_image_chat():
        """测试带图片的聊天"""
        print("\n=== 测试带图片的聊天 ===")
        messages: List[Message] = [
            {
                "role": "user",
                "content": "描述这张图片的内容",
                "media": ["image.jpg"],
            }
        ]
        response = vllm_lmm.chat(messages)
        print(f"模型回复: {response}")

    def test_generate_mode():
        """测试生成模式"""
        print("\n=== 测试生成模式 ===")
        prompt = "写一首关于秋天的诗"
        response = vllm_lmm.generate(prompt)
        print(f"生成结果: {response}")

    def test_streaming():
        """测试流式响应"""
        print("\n=== 测试流式响应 ===")
        messages: List[Message] = [{"role": "user", "content": "用三句话描述大海"}]
        print("流式响应:")
        for chunk in vllm_lmm.chat(messages, stream=True):
            print(chunk or "", end="", flush=True)
        print("\n--- 流式结束 ---")

    # 执行测试
    try:
        test_text_chat()
        test_image_chat()
        test_generate_mode()
        test_streaming()
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
