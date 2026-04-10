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
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Update `.env` with your `DISCORD_TOKEN` and `SARVAM_API_KEY`.

## Running the Bot
```bash
python bot.py
```

## How it works
The bot listens for any message in the channels it has access to (make sure the bot has `Message Content Intent` enabled in the Discord Developer Portal). It then sends the message content to Sarvam AI and replies with the generated response.
