# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 主要语言使用中文回答，包括文档撰写等

## Project Overview

claw0 is an educational project teaching AI Agent Gateway development from scratch. It contains 10 progressive sections, each introducing exactly one new concept while preserving all prior code. The codebase is trilingual (English/Chinese/Japanese) with identical logic across languages.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment (copy template and edit)
cp .env.example .env

# Run a section (pick your language)
python sessions/zh/s01_agent_loop.py      # Chinese
python sessions/en/s01_agent_loop.py      # English
python sessions/ja/s01_agent_loop.py      # Japanese

# Sections range from s01 to s10
python sessions/zh/s10_concurrency.py
```

## Architecture

```
Layers (bottom to top):
s01: Agent Loop   -- while True + stop_reason
s02: Tool Use     -- dispatch table pattern
s03: Sessions     -- JSONL persistence, ContextGuard for overflow
s04: Channels     -- Telegram + Feishu platform abstraction
s05: Gateway      -- 5-tier binding table, WebSocket routing
s06: Intelligence -- 8-layer prompt assembly (SOUL, IDENTITY, TOOLS, etc.)
s07: Heartbeat    -- Proactive agent + cron scheduler
s08: Delivery     -- Write-ahead queue with exponential backoff
s09: Resilience   -- 3-layer retry onion, auth rotation
s10: Concurrency  -- Named lanes with FIFO queues
```

**Section Dependencies:**
- s01-s02: Foundation (no dependencies)
- s03: Builds on s02 (persistence)
- s04: Builds on s03 (channels produce InboundMessages)
- s05: Builds on s04 (routes messages to agents)
- s06: Builds on s03 (prompt layers)
- s07: Builds on s06 (heartbeat uses soul/memory)
- s08: Builds on s07 (delivery queue)
- s09: Builds on s03+s06 (resilience)
- s10: Builds on s07 (named lanes)

## Key Design Patterns

**Dispatch Table** (s02): Tool routing via `TOOL_HANDLERS: dict[str, Callable]`
```python
TOOL_HANDLERS = {
    "read_file": handle_read_file,
    "write_file": handle_write_file,
}
result = TOOL_HANDLERS[name](**tool_input)
```

**JSONL Persistence** (s03): Append-only session logs with replay capability
- `SessionStore.append_transcript()` -- append on write
- `SessionStore._rebuild_history()` -- replay on read

**ContextGuard** (s03): Three-stage overflow handling
1. Normal API call
2. Truncate tool results
3. Compress history (50%)
4. Raise exception

**5-Tier Binding** (s05): Most specific match wins
```
(channel, peer, thread) > (channel, peer) > (channel) > (peer) > default
```

**8-Layer Prompt Assembly** (s06): Files in `workspace/` define agent behavior
- SOUL.md (personality), IDENTITY.md (role), TOOLS.md (tool docs)
- MEMORY.md (long-term preferences), USER.md (user info)
- HEARTBEAT.md, BOOTSTRAP.md, AGENTS.md

## Configuration

Required in `.env`:
- `ANTHROPIC_API_KEY` -- API key for Claude or compatible provider
- `MODEL_ID` -- Model identifier (default: claude-sonnet-4-20250514)

Optional:
- `ANTHROPIC_BASE_URL` -- For OpenRouter or other compatible providers
- `TELEGRAM_BOT_TOKEN` -- For s04 channels
- `FEISHU_APP_ID`, `FEISHU_APP_SECRET` -- For Feishu/Lark integration
- `HEARTBEAT_INTERVAL`, `HEARTBEAT_ACTIVE_START/END` -- For s07

## Code Organization

```
sessions/
  en/     -- English (s01_agent_loop.py ~ s10_concurrency.py + .md docs)
  zh/     -- Chinese version
  ja/     -- Japanese version
  test/   -- Test files

workspace/
  SOUL.md, IDENTITY.md, TOOLS.md, MEMORY.md  -- Agent configuration files
  .sessions/  -- Runtime session data (JSONL files)
```

Each session file is self-contained and runnable. Code logic is identical across languages; only comments and documentation differ.