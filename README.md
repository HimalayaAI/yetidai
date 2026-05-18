# YetiDai

A Discord agent powered by an OpenAI-compatible API that integrates natively with external tools like NepalOSINT for live news, macroeconomic data, and public information.

## Architecture

YetiDai uses a **multi-turn tool-calling architecture** and relies on the OpenAI-compatible `tool_choice="auto"` feature. Instead of explicitly trying to figure out intent with keywords, the bot exposes a **Tool Registry** to the LLM. 

The LLM is provided with declarative JSON-Schema specs for tools like `get_nepal_live_context`, intelligently selects which tools to call, and waits for YetiDai to execute them via the registry before producing the final formatted Nepali response.

### Key Components

- **`core/tool_contracts.py`**: Pydantic models mapping tool requirements (e.g. `ToolSpec`, `ToolParam`, `ToolResult`).
- **`core/tool_registry.py`**: A thread-safe Tool Registry that tracks tools, handles asynchronous executions, and generates the `tools` array for the API client.
- **`tools/osint/plugin.py`**: The NepalOSINT plugin which encapsulates retrieving recent Nepal news, macro data, public debt clocks, missing information, etc.
- **`tools/social/`**: A Nitter-backed social-media feed that automatically posts new AI/X account updates into `#social-media` and exposes the same feed state to YetiDai through `get_social_media_feed`.

## Setup

### Prerequisites
- Python 3.8 or higher.
- A Discord Bot Token from the [Discord Developer Portal](https://discord.com/developers/applications).
- An API Key from an OpenAI-compatible provider.
- optional [our model usages pre-trained model trained from himalayan ai nepali text corpus dataset (https://huggingface.co/datasets/himalaya-ai/nepali-corpus-compile)]

### Installation

1. Clone this repository (or copy the files).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.env` from the example values below:
   ```bash
   DISCORD_TOKEN=your-discord-bot-token
   API_KEY=your-openai-api-key
   BASE_URL=https://api.openai.com/v1
   MODEL_NAME=gpt-4.1-mini
   TIME_OUT_SECONDS=25
   NEPALOSINT_BASE_URL=https://nepalosint.com/api/v1
   NEPALOSINT_PUBLIC_AUTH_ENABLED=true
   NEPALOSINT_TIMEOUT_SECONDS=8
   NEPALOSINT_MAX_CONTEXT_ITEMS=8
   YETI_SOCIAL_FEED_ENABLED=true
   YETI_SOCIAL_CHANNEL_NAME=social-media
   ```

*(Note: `NEPALOSINT_*` variables are optional and default to the public API).*

### Automatic Social-Media Feed

YetiDai can post new X/Twitter updates into a Discord social-media channel without waiting for a user prompt. It uses public Nitter instances, stores seen tweet IDs in SQLite, and posts only new, direct timeline posts from the configured accounts.

Default behavior:

- Watches the AI account list in `tools/social/twitter_feed.py`.
- Finds `#social-media` automatically, including emoji-prefixed names such as `📱・social-media`.
- Marks currently visible tweets as seen on first run, so startup does not spam old posts.
- Persists state in `logs/social_feed.sqlite3`; no main database is required.
- Sends images as embeds and video posts as direct video links plus thumbnail embeds.

Useful environment variables:

```bash
YETI_SOCIAL_FEED_ENABLED=true
YETI_SOCIAL_CHANNEL_ID=
YETI_SOCIAL_CHANNEL_NAME=social-media
YETI_SOCIAL_POLL_SECONDS=300
YETI_SOCIAL_STATE_DB=logs/social_feed.sqlite3
YETI_SOCIAL_POST_EXISTING_ON_FIRST_RUN=false
YETI_SOCIAL_MAX_POSTS_PER_POLL=25
YETI_SOCIAL_ACCOUNTS=karpathy,fchollet,ylecun,AndrewYNg,rasbt,dair_ai,lilianweng,jeremyphoward,simonw,_akhaliq,ID_AA_Carmack,gwern,goodside,drfeifei,demishassabis,sama,nlethetech,HimalayaAILabs
YETI_SOCIAL_NITTER_INSTANCES=https://nitter.poast.org,https://nitter.privacydev.net
YETI_SOCIAL_INCLUDE_RETWEETS=false
YETI_SOCIAL_INCLUDE_REPLIES=false
```

YetiDai also registers `get_social_media_feed` as a normal auto tool-call. When users ask what the AI handles or social feed are saying, the model can read recent stored posts without triggering a fresh scrape.

## Running the Bot

```bash
python bot.py
```

The bot listens for any message in the channels it has access to. Make sure the bot has `Message Content Intent` enabled in the Discord Developer Portal.

## How Tool Calling Works

When a user asks `"नेपालमा आज के भइरहेको छ?"` (What's happening in Nepal today?):

1. The bot gives the LLM the user's message plus a list of tools from the `ToolRegistry`.
2. The LLM responds with `finish_reason: "tool_calls"` and requests `get_nepal_live_context`.
3. The registry executes the `tools.osint` plugin handler, fetching data from the NepalOSINT API.
4. The bot attaches the tool execution result into the conversation and queries the LLM again.
5. The LLM processes the live contextual data and returns a final text answer to the user containing the most relevant sources.

## How to Add New Tools

Adding plugins to the bot is seamless and requires minimal rewiring in the main loop. Every tool needs to define a `ToolSpec` and an asynchronous handler.

### 1. Create your plugin
Create a new file in `tools/` (e.g. `tools/n8n/plugin.py`):

```python
from core.tool_contracts import ToolSpec, ToolParam, ToolCategory, ToolResult, ToolContext
from core.tool_registry import get_registry

# Define your tool spec
MY_TOOL_SPEC = ToolSpec(
    tool_id="automation.n8n.trigger_workflow",
    name="trigger_n8n_workflow",
    description="Trigger an n8n automation workflow.",
    category=ToolCategory.AUTOMATION,
    parameters=[
        ToolParam(name="workflow_name", type="string", description="Name of the workflow to run.", required=True),
    ],
)

# Define the async handler
async def handle_n8n(ctx: ToolContext, arguments: dict) -> ToolResult:
    workflow = arguments.get("workflow_name")
    
    # ... your custom logic here ...
    
    return ToolResult(
        tool_id=MY_TOOL_SPEC.tool_id,
        success=True,
        content=f"Successfully triggered {workflow}."
    )

# Self-Register
def register() -> None:
    get_registry().register(MY_TOOL_SPEC, handle_n8n)
```

### 2. Register it in `bot.py`
To expose your new tool, import it and trigger the `register()` method at the top of `bot.py`:

```python
# In bot.py, near the top:
import tools.n8n.plugin as n8n_plugin
n8n_plugin.register()
```

That's it! The registry will now pass the new JSON schema spec to the LLM on every message, and automatically route matching `tool_calls` generated by the model to your `handle_n8n` function.

## Unit Testing
We use `pytest` for all core, logic, component, and plugin tests.

```bash
python -m pytest tests/ -v
```

For live integration tests interacting with actual external services (e.g., Sarvam APIs and NepalOSINT):
```bash
python tests/test_tool_call_local.py "नेपालमा आज के भइरहेको छ?"
```
