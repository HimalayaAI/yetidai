import discord
import os
import asyncio
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI
import re
from functionality import functional

#<--------------------Initializing project---------------------------------->

load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')

client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

with open('system_prompt.txt', 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read()

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

chad = functional(bot=bot)


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await chad.call(message)

    if not chad.user_input:
        return

    async with message.channel.typing():
        try:
            # Get message history for context
            previous_messages = await chad.get_message_history(message.channel, limit=5)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            
            # Format previous messages
            for prev_msg in previous_messages:
                if prev_msg.id != message.id and prev_msg.content.strip():
                    messages.append({
                        "role": "user",
                        "content": f"{prev_msg.author.name}: {prev_msg.content}"
                    })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": chad.user_input
            })

            # Generate response from API
            response = await client.chat.completions(
                model="sarvam-30b", 
                messages=messages
            )

            # Process response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                ai_response = response.choices[0].message.content
            else:
                ai_response = "I couldn't generate a response."

            if ai_response:
                for i in range(0, len(ai_response), 2000):
                    await message.channel.send(ai_response[i:i+2000])

        except Exception as e:
            print("Failed calling API")
            await message.channel.send(
                "Sorry, I encountered an error while processing your request."
            )


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
