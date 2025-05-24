import openai
import logging
import requests
import json
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ToolCall,
    convert_to_openai_messages,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from pydantic import BaseModel, Field
from pydantic import BaseModel as LCBaseModel
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from typing import Type

from functools import partial
from langchain_core.runnables import RunnableLambda

# 配置日志以支持 verbose 打印
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatQwenAI(BaseChatModel):
    model: str = "Qwen/Qwen3-235B-A22B"
    temperature: float = 0.7
    max_tokens: int = None
    logprobs: bool = False
    stream_options: dict = {}
    use_responses_api: bool = False
    timeout: float = None
    max_retries: int = 2
    api_key: str = None
    base_url: str = None
    organization: str = None
    enable_thinking: bool = False
    _endpoint: str = "/chat/completions"  # 将双下划线改为单下划线，弱化私有属性概念
    verbose: bool = False  
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def api_url(self) -> str:
        """生成完整 API 端点"""
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def with_structured_output(self, output_model: Type[BaseModel], method: str = "json_mode"):
        def generate_wrapper(messages, stop=None, run_manager=None, **kwargs):
            # 构建系统消息，指导模型输出结构化的 JSON
            system_message = SystemMessage(
                content=f"""
                你的任务是严格按照以下 JSON 结构输出结果：
                {output_model.schema_json(indent=2)}
                请确保输出的是有效的 JSON 数据，不要添加任何额外的文本、解释或注释。
                """
            )
            new_messages = [system_message] + messages

            # 调用原始的 _generate 方法
            response = self._generate(new_messages, stop=stop, run_manager=run_manager, **kwargs)

            try:
                # 解析 JSON 数据
                output_json = json.loads(response.generations[0].message.content)
                # 验证并转换为 Pydantic 模型实例
                return output_model(**output_json)
            except json.JSONDecodeError:
                raise ValueError("模型输出不是有效的 JSON 数据")
            except Exception as e:
                raise ValueError(f"解析模型输出时出错: {e}")

        return RunnableLambda(partial(generate_wrapper))
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            openai_messages = convert_to_openai_messages(messages)
            request_data = {
                "model": self.model,
                "messages": openai_messages,
                "stream": False,
                "enable_thinking": self.enable_thinking
            }

            if "tools" in kwargs:
                request_data["tools"] = kwargs["tools"]
                if "tool_choice" in kwargs:
                    request_data["tool_choice"] = kwargs["tool_choice"]

            if stop:
                request_data["stop"] = stop

            # 根据 verbose 属性控制日志输出
            if self.verbose:
                logger.info(f"Sending request to Qwen API: {request_data}")

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(self.api_url, json=request_data, timeout=self.timeout, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            if self.verbose:
                logger.info(f"Received response from Qwen API: {response_data}")

            return self._convert_response_to_chat_result(response_data)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                # 打印服务器返回的错误信息
                try:
                    error_data = e.response.json()
                    logger.error(f"Bad Request error details: {error_data}")
                except json.JSONDecodeError:
                    logger.error(f"Bad Request error, response content: {e.response.text}")
            raise ValueError(f"Error calling Qwen API: {e}")
        except Exception as e:
            raise ValueError(f"Error calling Qwen API: {e}")

    def _llm_type(self):
        return "qwen"

    def bind_tools(self, tools, tool_choice="auto", **kwargs):
        functions = []
        for tool in tools:
            try:
                openai_tool = convert_to_openai_tool(tool)
                functions.append(openai_tool)
            except Exception as e:
                logger.error(f"Error converting tool {tool} to OpenAI tool: {e}")

        new_kwargs = {
            "tool_choice": tool_choice,
            "tools": functions,
            **kwargs
        }
        return self.bind(**new_kwargs)

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, BaseMessage):
            messages = [input]
        else:
            messages = input

        response = self._generate(messages, **kwargs)
        return response.generations[0].message

    def stream(self, input, config=None, **kwargs):
        if isinstance(input, BaseMessage):
            messages = [input]
        else:
            messages = input

        request_data = {
            "model": self.model,
            "messages": convert_to_openai_messages(messages),
            "stream": True,
            "enable_thinking": self.enable_thinking
        }

        if "tools" in kwargs:
            request_data["tools"] = kwargs["tools"]
            if "tool_choice" in kwargs:
                request_data["tool_choice"] = kwargs["tool_choice"]

        if "stop" in kwargs:
            request_data["stop"] = kwargs["stop"]
        if self.verbose:
            logger.info(f"Sending streaming request to Qwen API: {request_data}")
        
        with requests.post(self.api_url, json=request_data, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = line.lstrip(b'data: ').decode('utf-8')
                    if chunk == '[DONE]':
                        break
                    try:
                        chunk_data = json.loads(chunk)
                        if self.verbose:
                            logger.info(f"Received streaming chunk from Qwen API: {chunk_data}")
                        
                        yield self._convert_chunk_to_chat_generation_chunk(chunk_data)
                    except json.JSONDecodeError:
                        continue

    def _convert_response_to_chat_result(self, response):
        generations = []
        for choice in response.get('choices', []):
            message = self._convert_dict_to_message(choice.get('message', {}))
            generation = ChatGeneration(message=message)
            generations.append(generation)
        return ChatResult(generations=generations)

    def _convert_chunk_to_chat_generation_chunk(self, chunk):
        choice = chunk.get('choices', [{}])[0]
        delta = choice.get('delta', {})
        message_chunk = self._convert_dict_to_message_chunk(delta)
        return ChatGenerationChunk(message=message_chunk)

    def _convert_dict_to_message(self, _dict):
        role = _dict.get("role")
        name = _dict.get("name")
        if role == "user":
            return HumanMessage(content=_dict.get("content", ""), name=name)
        elif role == "assistant":
            content = _dict.get("content", "") or ""
            additional_kwargs = {}
            tool_calls = []
            if function_call := _dict.get("function_call"):
                additional_kwargs["function_call"] = function_call
            if raw_tool_calls := _dict.get("tool_calls"):
                for raw_tool_call in raw_tool_calls:
                    try:
                        # 尝试将 arguments 字符串解析为字典
                        args = json.loads(raw_tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    tool_call = {
                        "id": raw_tool_call.get("id", ""),
                        "name": raw_tool_call["function"]["name"],
                        "args": args
                    }
                    tool_calls.append(tool_call)
                additional_kwargs["tool_calls"] = tool_calls
            return AIMessage(
                content=content,
                additional_kwargs=additional_kwargs,
                name=name,
                tool_calls=tool_calls
            )
        elif role == "system":
            return SystemMessage(content=_dict.get("content", ""), name=name)
        elif role == "function":
            return FunctionMessage(content=_dict.get("content", ""), name=name)
        elif role == "tool":
            return ToolMessage(
                content=_dict.get("content", ""),
                tool_call_id=_dict.get("tool_call_id"),
                name=name
            )

    def _convert_dict_to_message_chunk(self, _dict):
        role = _dict.get("role")
        name = _dict.get("name")
        if role == "assistant":
            content = _dict.get("content", "") or ""
            additional_kwargs = {}
            tool_calls = []
            if function_call := _dict.get("function_call"):
                additional_kwargs["function_call"] = function_call
            if raw_tool_calls := _dict.get("tool_calls"):
                for raw_tool_call in raw_tool_calls:
                    try:
                        # 尝试将 arguments 字符串解析为字典
                        args = json.loads(raw_tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    tool_call = {
                        "id": raw_tool_call.get("id", ""),
                        "name": raw_tool_call["function"]["name"],
                        "args": args
                    }
                    tool_calls.append(tool_call)
                additional_kwargs["tool_calls"] = tool_calls
            return AIMessageChunk(
                content=content,
                additional_kwargs=additional_kwargs,
                name=name,
                tool_calls=tool_calls
            )

    def __getattr__(self, attr):
        # 模拟支持 model_dump_json 方法
        if attr == "model_dump_json":
            def model_dump_json(*args, **kwargs):
                # 暂时用空列表代替
                result = self._generate([], **kwargs)
                return json.dumps({
                    "content": result.generations[0].message.content,
                    "tool_calls": result.generations[0].message.additional_kwargs.get("tool_calls", [])
                })
            return model_dump_json
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

