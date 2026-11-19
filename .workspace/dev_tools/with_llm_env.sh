#!/usr/bin/env bash
# Usage: source dev_tools/with_llm_env.sh && python3 ... --mode live
# Reads the local key file into an environment variable. Never prints the token.
#
# 支持三类 OpenAI-compatible 提供方（key / endpoint 均只从环境变量读取）：
# 1. Google AI Studio（默认）: RSI_LLM_PROVIDER=google_studio
#    RSI_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
# 2. 任意 OpenAI-compatible API: RSI_LLM_PROVIDER=openai
#    RSI_LLM_API_KEY=sk-... RSI_LLM_BASE_URL=https://api.openai.com/v1
#    （或直接使用标准变量 OPENAI_API_KEY / OPENAI_BASE_URL）
# 3. PyTRiO: RSI_LLM_PROVIDER=pytrio（或仅设置 PYTRIO_API_KEY 自动识别）
#    端点默认 https://pytrio.com（/api/openai/v1/chat/completions），模型自动发现（优先 4B）。
if [ -z "${RSI_LLM_API_KEY:-}" ] && [ -f "$HOME/api-key.txt" ]; then
  RSI_LLM_API_KEY="$(tr -d '\r\n' < "$HOME/api-key.txt" | head -n 1)"
  export RSI_LLM_API_KEY
fi
PROVIDER="${RSI_LLM_PROVIDER:-auto}"
if [ "$PROVIDER" = "google_studio" ] || [ "$PROVIDER" = "auto" ]; then
  # 本地代理与 Gemini 模型默认值仅适用于 Google AI Studio 通道。
  export RSI_LLM_PROXY="${RSI_LLM_PROXY:-http://127.0.0.1:7890}"
  export RSI_LLM_MODEL="${RSI_LLM_MODEL:-gemini-flash-lite-latest}"
  export RSI_LLM_BASE_URL="${RSI_LLM_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}"
else
  # pytrio / openai：不注入代理与模型默认值，端点由提供方决定，模型可自动发现。
  export RSI_LLM_PROXY="${RSI_LLM_PROXY:-}"
fi
export RSI_LLM_PROVIDER="${PROVIDER}"
