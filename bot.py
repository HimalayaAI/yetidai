import discord
import os
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI
from functionality import functional
from context_formatter import build_context_brief
from context_router import fetch_context_bundle
from nepalosint_client import NepalOSINTClient
from retrieval_planner import resolve_route_plan

#<--------------------Initializing project---------------------------------->

load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')

client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
nepalosint_client = NepalOSINTClient()

with open('systemPrompt.txt', 'r', encoding='utf-8') as f:
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
            route_plan = await resolve_route_plan(client, chad.user_input, previous_messages)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

            if route_plan.use_nepalosint:
                context_bundle = await fetch_context_bundle(nepalosint_client, chad.user_input, route_plan)
                context_message = build_context_brief(context_bundle, max_chars=1800)
                if context_message:
                    messages.append({
                        "role": "system",
                        "content": f"Current NepalOSINT context:\n{context_message}"
                    })
            
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
