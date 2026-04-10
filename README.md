# YetiDAI

NepalOSINT, live news, and macroeconomic awareness.

## Prerequisites
- Python 3.8 or higher.
- A Discord Bot Token from the [Discord Developer Portal](https://discord.com/developers/applications).
- An API Key from the [Sarvam AI Dashboard](https://dashboard.sarvam.ai/).  
- optional [our model usages pre-trained model trained from himalayan ai nepali text corpus dataset (https://huggingface.co/datasets/himalaya-ai/nepali-corpus-compile)]

## Setup
1. Clone this repository (or copy the files).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.env` from the example values below:
   ```bash
   DISCORD_TOKEN=your-discord-bot-token
   SARVAM_API_KEY=your-sarvam-api-key
   NEPALOSINT_BASE_URL=https://nepalosint.com/api/v1
   NEPALOSINT_PUBLIC_AUTH_ENABLED=true
   NEPALOSINT_TIMEOUT_SECONDS=8
   NEPALOSINT_MAX_CONTEXT_ITEMS=8
   ```
4. Update `.env` with your real `DISCORD_TOKEN` and `SARVAM_API_KEY`.

`NEPALOSINT_BASE_URL`, `NEPALOSINT_PUBLIC_AUTH_ENABLED`, `NEPALOSINT_TIMEOUT_SECONDS`, and `NEPALOSINT_MAX_CONTEXT_ITEMS` are optional. The defaults point to the public NepalOSINT API and are already set in `nepalosint_client.py`.

## Running the Bot
```bash
python bot.py
```

## How it works
The bot listens for any message in the channels it has access to. Make sure the bot has `Message Content Intent` enabled in the Discord Developer Portal.

Before sending a user message to Sarvam AI, YetiDAI decides whether it needs live Nepal public-information context. This is handled by `retrieval_planner.py` and `context_router.py`:

1. `retrieval_planner.py` starts with keyword routing and, for ambiguous follow-up questions, asks Sarvam to return a small JSON retrieval plan.
2. `context_router.py` maps the plan into NepalOSINT intents: `general_news`, `macro`, `government`, `debt`, `parliament`, and `trading`.
3. `nepalosint_client.py` fetches the matching live data from NepalOSINT.
4. `context_formatter.py` compresses those payloads into a short system message named `Current NepalOSINT context`.
5. `bot.py` sends the system prompt, the NepalOSINT context, recent Discord history, and the current user message to Sarvam AI.

The system prompt tells Yeti to treat NepalOSINT as authoritative for current Nepal public-information questions. When NepalOSINT context is used, Yeti must answer in Nepali and end with a `स्रोत:` section containing the most relevant sources.

## NepalOSINT coverage
YetiDAI currently uses NepalOSINT for:

- Recent Nepal news and consolidated story history.
- NRB-style macroeconomic snapshot data such as inflation, remittance, reserves, trade, tourism, migration, money supply, and banking indicators.
- Government decisions and official announcements.
- Nepal public debt clock and debt-related news.
- Federal Parliament session summaries and tracked bills.
- NEPSE, ticker-like, company, IPO, dividend, right-share, and market-related searches.

For historical questions such as `yesterday`, `last week`, or explicit `YYYY-MM-DD` ranges, YetiDAI requests NepalOSINT history data and preserves the time range in the answer.

If NepalOSINT cannot be reached, the formatter injects a fallback context telling the model that live context could not be fetched. The system prompt then requires Yeti to say that limitation instead of inventing current facts.
