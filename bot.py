import discord
import os
import asyncio
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI
from functionality import Functional   # Note: We use the improved Functional class

load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')

if not DISCORD_TOKEN or not SARVAM_API_KEY:
    raise ValueError("Missing DISCORD_TOKEN or SARVAM_API_KEY in .env file")

client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

# Load system prompt
try:
    with open('systemPrompt.txt', 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read().strip()
except FileNotFoundError:
    SYSTEM_PROMPT = "तिमी एक helpful, truthful र witty AI assistant हौ जसले नेपाली र English दुवैमा राम्रोसँग कुरा गर्न सक्छ। तिमी Grok जस्तै सत्यतर्फ उन्मुख र curious छौ।"
    print("Warning: systemPrompt.txt not found. Using default prompt.")

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)
chad = Functional(bot=bot)   # Your improved helper class

@bot.event
async def on_ready():
    print(f'✅ Logged in as {bot.user} (ID: {bot.user.id})')
    print('Bot is ready to chat! Use !chat or mention the bot.')

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or message.author.bot:
        return

    # Let Functional class handle input detection
    user_input = await chad.call(message)
    if not user_input:
        return

    async with message.channel.typing():
        try:
            # Get conversation history (now properly formatted strings)
            history = await chad.get_message_history(message.channel, limit=8)

            # Build messages list in proper OpenAI-style format
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

            # Add history (already formatted as "Username: message")
            for hist_msg in history:
                messages.append({"role": "user", "content": hist_msg})

            # Add current user message (clean input)
            messages.append({"role": "user", "content": user_input})

            # Call Sarvam AI (sarvam-30b is good for speed + Indic languages)
            response = await client.chat.completions(
                model="sarvam-30b",      # or try "sarvam-m" if available (free tier)
                messages=messages,
                temperature=0.7,         # Balanced creativity
                max_tokens=1024
            )

            if hasattr(response, 'choices') and response.choices:
                ai_response = response.choices[0].message.content.strip()
            else:
                ai_response = "माफ गर्नुहोस्, मलाई अहिले जवाफ दिन समस्या भयो। फेरि प्रयास गर्नुहोस्।"

            # Send response (split if too long for Discord)
            if ai_response:
                for i in range(0, len(ai_response), 2000):
                    chunk = ai_response[i:i+2000]
                    await message.channel.send(chunk)

        except Exception as e:
            print(f"❌ Error calling Sarvam API: {e}")
            await message.channel.send(
                "माफ गर्नुहोस्, API मा समस्या आयो। केही समय पछि फेरि प्रयास गर्नुहोस्।"
            )

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
