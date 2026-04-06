import discord
import os
import asyncio
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')

# Initialize Async Sarvam AI Client
client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

# Load system prompt from external file
def load_system_prompt():
    try:
        with open('system_prompt.txt', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print("Warning: system_prompt.txt not found. Using default prompt.")
        return "You are a helpful AI assistant that responds in Nepali."

SYSTEM_PROMPT = load_system_prompt()

# Initialize Discord Bot
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content
bot = discord.Client(intents=intents)

async def get_message_history(channel, limit=10):
    """
    Fetch up to 'limit' previous messages from the channel.
    Returns messages in chronological order (oldest first).
    """
    messages = []
    async for msg in channel.history(limit=limit, oldest_first=False):
        # Skip bot's own messages and system messages
        if msg.author != bot.user and not msg.author.bot:
            messages.append(msg)
    
    # Reverse to get chronological order (oldest to newest)
    messages.reverse()
    return messages

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('Bot is ready to respond to chats!')
    print('------')

@bot.event
async def on_message(message):
    # Don't respond to ourselves
    if message.author == bot.user:
        return

    # Check if message starts with !chat command or mentions the bot
    user_input = None
    
    if message.content.startswith('!chat'):
        # Handle !chat command
        user_input = message.content[len('!chat'):].strip()
    elif bot.user.mentioned_in(message):
        # Handle mentions - remove the bot mention and get the rest of the message
        # Remove all mentions of the bot from the message
        user_input = message.content
        for mention in message.mentions:
            if mention == bot.user:
                user_input = user_input.replace(f'<@{bot.user.id}>', '').replace(f'<@!{bot.user.id}>', '').strip()
    
    if not user_input:
        if message.content.startswith('!chat'):
            await message.channel.send("कृपया आफ्नो प्रश्न लेख्नुहोस्। उदाहरण: `!chat नमस्ते`")
        return

    async with message.channel.typing():
        try:
            # Fetch up to 10 previous messages for context
            previous_messages = await get_message_history(message.channel, limit=10)
            
            # Prepare messages for Sarvam AI - include message history for context
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            
            # Add previous messages as context (excluding the current message)
            for prev_msg in previous_messages:
                if prev_msg.id != message.id:  # Don't include the current message yet
                    # Determine role based on author
                    role = "user"
                    # Filter out empty messages and attachments-only messages
                    if prev_msg.content.strip():
                        messages.append({
                            "role": role,
                            "content": f"{prev_msg.author.name}: {prev_msg.content}"
                        })
            
            # Add current user input
            messages.append({
                "role": "user",
                "content": user_input
            })

            # Call Sarvam AI API using the async client
            # Model choices: "sarvam-30b" or "sarvam-105b"
            response = await client.chat.completions(
                model="sarvam-30b", 
                messages=messages
            )
            
            # Extract content from the response
            # Note: Sarvam AI SDK returns an object similar to OpenAI
            if hasattr(response, 'choices') and len(response.choices) > 0:
                ai_response = response.choices[0].message.content
            else:
                ai_response = "माफ गर्नुहोस्, मले जवाफ तैयार गर्न सकेन।"

            # Send the response back to Discord
            if ai_response:
                # Split message if it exceeds Discord's 2000 character limit
                for i in range(0, len(ai_response), 2000):
                    await message.channel.send(ai_response[i:i+2000])

        except Exception as e:
            print(f"Error calling Sarvam AI: {e}")
            await message.channel.send("माफ गर्नुहोस्, अनुरोध प्रक्रिया गर्दा एक त्रुटि भेटिएको छ।")

if __name__ == "__main__":
    if not DISCORD_TOKEN or not SARVAM_API_KEY:
        print("Error: DISCORD_TOKEN and SARVAM_API_KEY must be set in your .env file.")
    else:
        # Run the Discord bot
        bot.run(DISCORD_TOKEN)
