import os
import sys
import json
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from certifi import contents
from dotenv import load_dotenv
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools.\n"
    "Use tools to help the user with file and time queries.\n"
    "Be concise. If a session has prior context, use it."
)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent / "workspace"

CONTEXT_SAFE_LIMIT = 180000

MAX_TOOL_OUTPUT = 50000

# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
MAGENTA = "\033[35m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_tool(name: str, detail: str) -> None:
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


def print_warn(text: str) -> None:
    print(f"{YELLOW}{text}{RESET}")


def print_session(text: str) -> None:
    print(f"{MAGENTA}{text}{RESET}")


def safe_path(raw: str) -> Path:
    target = (WORKSPACE_DIR / raw).resolve()
    if not str(target).startswith(str(WORKSPACE_DIR.resolve())):
        raise ValueError(f"Path traversal blocked: {raw}")
    return target


# SessionStore -- 基于 JSONL 的 session 持久化
class SessionStore:
    """管理 agent 会话的持久化存储。"""
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.base_dir = WORKSPACE_DIR / ".sessions" / "agents" / agent_id / "sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir.parent / "sessions.json"
        self._index: dict[str, dict] = self._load_index()
        self.current_session_id: str | None = None

    def _load_index(self) -> dict[str, dict]:
        """加载会话索引文件，异常时返回空字典（容错处理）。"""
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self) -> None:
        """将会话索引写入 JSON 文件（格式化存储，保证可读性）。"""
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _session_path(self, session_id: str) -> Path:
        """生成指定会话ID的消息存储文件路径（JSONL 格式，按行存储消息）。"""
        return self.base_dir / f"{session_id}.jsonl"

    def create_session(self, label: str = "") -> str:
        """创建新会话，生成唯一ID，记录元信息并初始化空文件。"""
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._index[session_id] = {
            "label": label,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }
        self._save_index()
        self._session_path(session_id).touch()
        self.current_session_id = session_id
        return session_id

    def load_session(self, session_id: str) -> list[dict]:
        """从 JSONL 重建 API 格式的 messages[]"""
        path = self._session_path(session_id)
        if not path.exists():
            return []
        self.current_session_id = session_id
        return self._rebuild_history(path)

    def save_turn(self, role: str, content: Any) -> None:
        if not self.current_session_id:
            return
        self.append_transcript(self.current_session_id,{
            "type": role,
            "content": content,
            "ts": time.time()
        })

    def save_tool_result(self, tool_use_id: str, name:str, tool_input: dict, result: str) -> None:
        if not self.current_session_id:
            return
        ts = time.time()
        self.append_transcript(self.current_session_id,{
            "type": "tool_use",
            "tool_use_id": tool_use_id,
            "name": name,
            "input": tool_input,
            "ts": ts
        })
        self.apeend_transcript(self.current_session_id,{
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result,
            "ts": ts
        })

    def append_transcript(self, session_id: str, record: dict) -> None:
        path = self._session_path(session_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if session_id in self._index:
            self._index[session_id]["last_active"] = (
                datetime.now(timezone.utc).isoformat()
            )
            self._index[session_id]["message_count"] += 1
            self._save_index()

    def _rebuild_history(self, path: Path) -> list[dict]:
        """
        从 JSONL 行重建 API 格式的消息列表。

        Anthropic API 规则决定了这种重建方式：
        - 消息必须 user/assistant 交替
        - tool_use 块属于 assistant 消息
        - tool_result 块属于 user 消息
        """
        messages: list[dict] = []
        lines = path.read_text(encoding="utf-8").strip().split("\n")

        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = record.get("type")

            if rtype == 'user':
                messages.append({
                    "role": "user",
                    "content": record["content"],
                })

            elif rtype == "assistant":
                content = record["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                messages.append({
                    "role": "assistant",
                    "content": content,
                })

            elif rtype == "tool_use":
                block = {
                    "type": "tool_use",
                    "id": record["tool_use_id"],
                    "name": record["name"],
                    "input": record["input"]
                }
                if messages and messages[-1]["role"] == "assistant":
                    content = messages[-1]["content"]
                    if isinstance(content, list):
                        content.append(block)
                    else:
                        messages[-1]["content"] = [
                            {"type": "text", "text": str(content)},
                            block,
                        ]
                else:
                    messages.append({
                        "role": "assistant",
                        "content": [block],
                    })
            elif rtype == "tool_result":
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": record["tool_use_id"],
                    "content": record["content"],
                }
                # 将连续的 tool_result 合并到同一个 user 消息中
                if (messages and messages[-1]["role"] == "user"
                        and isinstance(messages[-1]["content"], list)
                        and messages[-1]["content"]
                        and isinstance(messages[-1]["content"][0], dict)
                        and messages[-1]["content"][0].get("type") == "tool_result"):
                    messages[-1]["content"].append(result_block)
                else:
                    messages.append({
                        "role": "user",
                        "content": [result_block],
                    })
        return messages

    def list_sessions(self) -> list[tuple[str, dict]]:
        items = list(self._index.items())
        items.sort(key=lambda x: x[1].get("last_active", ""), reverse=True)
        return items


def _serialize_messages_for_summary(messages: list[dict]) -> str:
    """将消息列表扁平化为纯文本，用于 LLM 摘要"""
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(f"[{role}]: {block['content']}")
                    elif btype == "tool_use":
                        parts.append(
                            f"[{role} called {block.get('name', '?')}]: "
                            f"{json.dumps(block.get('input', []), ensure_ascii=False)}"
                        )
                    elif btype == "tool_result":
                        rc = block.get("content", "")
                        preview = rc[:500] if isinstance(rc, str) else str(rc)[:500]
                        parts.append(f"[tool_result]: {preview}")
                elif hasattr(block, "text"):
                    parts.append(f"[{role}]: {block.text}")
    return "\n".join(parts)

# ContextGuard 上下文溢出保护
# 1. 截断过大的工具结果
# 2. 将旧消息压缩为 LLM 生成的摘要（固定50%的比例）
# 3. 仍然溢出则抛出异常

class ContextGuard:
    """保护agent免受上下文窗口溢出"""
    def __init__(self, max_token: int = CONTEXT_SAFE_LIMIT):
        self.max_token = max_token

    @staticmethod
    def estimate_tokens(text: str):
        return len(text) // 4

    def estimate_messages_tokens(self, messages: list[[dict]]) -> int:
        """估算tokens"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if "text" in block:
                            total += self.estimate_tokens(block["text"])
                        elif block.get("type") == "tool_result":
                            rc = block.get("content", "")
                            if isinstance(rc, str):
                                total += self.estimate_tokens(rc)
                        elif block.get("type") == "tool_use":
                            total += self.estimate_tokens(
                                json.dumps(block.get("input", {})),
                            )
                    else:
                        if hasattr(block, "text"):
                            total += self.estimate_tokens(block.text)
                        elif hasattr(block, "input"):
                            total += self.estimate_tokens(
                                json.dumps(block.input)
                            )
        return total

    def truncate_tool_result(self, result: str, max_fraction: float = 0.3) -> str:
        """在换行边界处只保留头部进行截断"""
        max_chars = int(self.max_token * 4 * max_fraction)
        if len(result) <= max_chars:
            return result
        cut = result.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        head = result[:cut]
        return head + f"\n\n[... truncated ({len(result)} chars total, showing first {len(head)}) ...]"


    def compact_history(self, messages: list[dict],
                        api_client: Anthropic, model: str) -> list[dict]:
        """
        将前 50% 的消息压缩为LLM生成的摘要
        保留最后N条信息（N = max(4, total's 20%)）不变
        """
        total = len(messages)
        if total <= 4:
            return messages

        keep_count = max(4, int(total * 0.2))
        compress_count = max(2, int(total * 0.5))
        compress_count = min(compress_count, total - keep_count)

        if compress_count < 2:
            return messages

        old_messages = messages[:compress_count]
        recent_messages = messages[compress_count:]

        old_text = _serialize_messages_for_summary(old_messages)

        summary_prompt = (
            "Summarize the following conversation concisely, "
            "preserving key facts and decisions."
            "Output only the summary, no preamble.\n\n"
            f"{old_text}"
        )

        try:
            summary_resp = api_client.messages.create(
                model=model,
                max_tokens=2048,
                system="You are a conversation summarizer. Be concise and factual.",
                messages=[{"role": "user", "content": summary_prompt}],
            )
            summary_text = ""
            for block in summary_resp.content:
                if hasattr(block, "text"):
                    summary_text += block.text

            print_session(
                f" [compact] {len(old_messages)} messages -> summary"
                f"({len(summary_text)} chars)"
            )

        except Exception as exc:
            print_warn(f"  [compact] Summary failed ({exc}), dropping old messages")
            return recent_messages

        compacted = [
            {
                "role": "user",
                "content": "[Previous conversation summary]\n" + summary_text,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Understood, I have the context from our previous conversation."}]
            },
        ]
        compacted.extend(recent_messages)
        return compacted

    def _truncate_large_tool_results(self, messages: list[dict]) -> list[dict]:
        """遍历消息列表，截断过大的 tool_result 块"""
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                new_blocks = []
                for block in content:
                    if (isinstance(block, dict)
                            and block.get("type") == "tool_result"
                            and isinstance(block.get("content"), str)):
                        block = dict(block)
                        block["content"] = self.truncate_tool_result(
                            block["content"]
                        )
                    new_blocks.append(block)
                result.append({"role": msg["role"], "content": new_blocks})
            else:
                result.append(msg)
        return result


    def guard_api_call(
            self,
            api_client: Anthropic,
            model: str,
            system: str,
            messages: list[dict],
            tools: list[dict] | None = None,
            max_retries: int = 2,
    ) -> Any:
        """
        三阶段重试：
            第0次尝试：正常调用
            第1次尝试：截断过大的工具结果
            第2次尝试：通过 LLM 摘要压缩历史
        """






