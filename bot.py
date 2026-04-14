import discord
import os
import asyncio
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI
from functionality import functional

# <-------------------- Initialization ---------------------------------->

load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')

# Initialize the AI Client
client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

# Load Syllabus from systemPrompt.txt
try:
    with open('systemPrompt.txt', 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    SYSTEM_PROMPT = "You are a helpful assistant. Stick strictly to the syllabus provided."

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

# Initialize the logic class
chad = functional(bot=bot)

@bot.event
async def on_ready():
    print(f'✅ Logged in as {bot.user} (ID: {bot.user.id})')
    print('--- Bot is online and ready ---')

@bot.event
async def on_message(message):
    # Ignore self
    if message.author == bot.user:
        return

    # Process if the bot is being called (!chat or @mention)
    await chad.call(message)

    # If no valid input was detected, stop here
    if not chad.user_input:
        return

    async with message.channel.typing():
        try:
            # 1. Get Conversation History
            previous_messages = await chad.get_message_history(message.channel, limit=6)
            
            # 2. Build the Message Payload
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            for prev_msg in previous_messages:
                # Skip the message the user just sent (we add it at the very end)
                if prev_msg.id == message.id:
                    continue
                
                # Assign roles correctly so the AI knows who said what
                role = "assistant" if prev_msg.author == bot.user else "user"
                
                messages.append({
                    "role": role,
                    "content": prev_msg.content
                })
            
            # Add the new prompt
            messages.append({"role": "user", "content": chad.user_input})

            # 3. Request Completion from Sarvam AI
            response = await client.chat.completions(
                model="sarvam-30b", 
                messages=messages,
                temperature=0.2,  # Low temperature = Strict syllabus adherence
                top_p=0.9
            )

            # 4. Handle and Send Response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                ai_response = response.choices[0].message.content
            else:
                ai_response = "I'm sorry, I couldn't process that request right now."

            # Send in chunks if response exceeds Discord's 2000 character limit
            if ai_response:
                for i in range(0, len(ai_response), 2000):
                    await message.channel.send(ai_response[i:i+2000])

        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            await message.channel.send("An error occurred while connecting to the AI brain.")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
