"""Qwen3 XML tool call parser for verl AgentLoop.

Qwen3 generates tool calls in XML format inside tool-call wrapper tokens:
    <function=func_name>
    <parameter=key>value</parameter>
    </function>

This parser extracts them into FunctionCall objects, matching verl's
ToolParser interface so it can be registered as multi_turn.format=qwen3_xml.
"""
import json
import logging
import os
import regex

from verl.experimental.agent_loop.tool_parser import ToolParser, FunctionCall
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Construct tool-call start/end tokens from char codes to avoid
# the literal strings being interpreted as tool-call markers.
# These match HermesToolParser and the Qwen3.5 tokenizer (IDs 248058/248059).
_TC_START = chr(60) + "tool_call" + chr(62)
_TC_END = chr(60) + chr(47) + "tool_call" + chr(62)

_FUNC_RE = regex.compile(r"<function=([^>]+)>(.*?)</function>", regex.DOTALL)
_PARAM_RE = regex.compile(r"<parameter=([^>]+)>(.*?)</parameter>", regex.DOTALL)
_JSON_RE = regex.compile(r"\{(?:[^{}]|(?R))*\}", regex.DOTALL)


def _trim_newlines(value):
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


@ToolParser.register("qwen3_xml")
class Qwen3XMLToolParser(ToolParser):
    """Parses Qwen3 XML tool calls inside the model's tool-call wrapper tokens.

    Qwen3 format (inside start/end tokens):
        <function=name>
        <parameter=key>value</parameter>
        </function>
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.tool_call_start_token = _TC_START
        self.tool_call_end_token = _TC_END
        start_esc = regex.escape(self.tool_call_start_token)
        end_esc = regex.escape(self.tool_call_end_token)
        self.tool_call_regex = regex.compile(
            start_esc + r"(.*?)" + end_esc, regex.DOTALL
        )

    @rollout_trace_op
    async def extract_tool_calls(self, responses_ids):
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)

        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []

        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            try:
                func_matches = _FUNC_RE.findall(match)
                if func_matches:
                    for name, args_body in func_matches:
                        params = {}
                        for pm in _PARAM_RE.finditer(args_body):
                            params[pm.group(1).strip()] = _trim_newlines(pm.group(2))
                        function_calls.append(FunctionCall(
                            name=name.strip(),
                            arguments=json.dumps(params, ensure_ascii=False),
                        ))
                else:
                    # Qwen3-VL chat template trains the model to emit JSON
                    # inside <tool_call>: {"name": ..., "arguments": {...}}
                    parsed_json = False
                    for jm in _JSON_RE.finditer(match):
                        try:
                            obj = json.loads(jm.group(0))
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict) and obj.get("name"):
                            args = obj.get("arguments") or {}
                            if not isinstance(args, dict):
                                try:
                                    args = json.loads(args)
                                except (json.JSONDecodeError, TypeError):
                                    args = {}
                            function_calls.append(FunctionCall(
                                name=str(obj["name"]).strip(),
                                arguments=json.dumps(args, ensure_ascii=False),
                            ))
                            parsed_json = True
                    if not parsed_json:
                        logger.error("No <function=> or JSON object found in tool call block")
            except Exception as e:
                logger.error(f"Failed to decode tool call: {e}")

        content = self.tool_call_regex.sub("", text)
        return content, function_calls
