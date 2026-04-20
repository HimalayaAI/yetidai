import discord
import os
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI
from functionality import Functional

load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')

if not DISCORD_TOKEN or not SARVAM_API_KEY:
    raise ValueError("Missing DISCORD_TOKEN or SARVAM_API_KEY in .env file")

client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

try:
    with open('systemPrompt.txt', 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read().strip()
except FileNotFoundError:
    SYSTEM_PROMPT = (
        "You are a helpful and truthful AI assistant. "
        "You can respond in both Nepali and English. "
        "Always match the primary language used by the user. "
        "If the user mixes languages, respond using the same combination. "
        "Maintain consistent language throughout the conversation "
        "and avoid switching to Hindi unless specifically requested."
    )
    print("Warning: systemPrompt.txt not found. Using default prompt.")

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)
chad = Functional(bot=bot)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('Bot is ready.')

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or message.author.bot:
        return

    user_input = await chad.call(message)
    if not user_input:
        return

    async with message.channel.typing():
        try:
            history = await chad.get_message_history(message.channel, limit=10)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

            for hist in history:
                if isinstance(hist, dict) and "role" in hist and "content" in hist:
                    messages.append(hist)
                else:
                    messages.append({"role": "user", "content": str(hist)})

            messages.append({"role": "user", "content": user_input})

            response = await client.chat.completions(
                model="sarvam-30b",
                messages=messages,
                temperature=0.75,
                max_tokens=1024
            )

            if hasattr(response, 'choices') and response.choices:
                ai_response = response.choices[0].message.content.strip()
            else:
                ai_response = "Sorry, I encountered an issue while generating a response. Please try again."

            if ai_response:
                for i in range(0, len(ai_response), 2000):
                    chunk = ai_response[i:i+2000]
                    await message.channel.send(chunk)

        except Exception as e:
            print(f"Error calling Sarvam API: {e}")
            await message.channel.send(
                "Sorry, there was a problem with the AI service. Please try again in a moment."
            )


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
