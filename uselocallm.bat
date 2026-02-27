# 1. Connection & Key
$env:ANTHROPIC_BASE_URL = "http://spark-ai:4000"
$env:ANTHROPIC_API_KEY = "sk-12345678"

# 2. THE FIX FOR MEMORY BLOW-UPS
# This removes the 'x-anthropic-billing-header' that crashes local proxies
$env:CLAUDE_CODE_ATTRIBUTION_HEADER = "0"
$env:CLAUDE_CODE_USE_TEXT_CONTENT = "1"
$env:CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS = "1"

# 3. Model & Context Mapping
$env:ANTHROPIC_MODEL = "qwen3-thinking"
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = "qwen3-thinking"
$env:ANTHROPIC_SMALL_FAST_MODEL = "qwen3-thinking"
$env:CLAUDE_CODE_MAX_CONTEXT_TOKENS = "95000"

# 4. Launch with Permission Bypass
# This prevents the model from getting stuck in 'approval' loops
claude --dangerously-skip-permissions