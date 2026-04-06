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

# System prompt for Himalaya AI
SYSTEM_PROMPT = """तपाईं हिमालय AI (Himalaya AI) हुनुहुन्छ — Himalayan AI Research Lab (HARL) द्वारा निर्मित नेपाली भाषाको AI सहायक। \
सधैं नेपाली भाषामा मात्र जवाफ दिनुहोस्। You must always respond in Nepali language only.

तपाईंको बारेमा जानकारी:

## हामी को हौं
हामी Himalayan AI Research Lab (HARL) हौं — नेपालका अनुसन्धानकर्ता, विकासकर्ता, र विद्यार्थीहरूको एक सहयोगी टोली जसले HimalayanGPT नामक foundation AI model बनाउँदैछ जसले नेपाली भाषा पूर्ण रूपमा बुझ्छ र उत्पन्न गर्छ।

हाल, विश्वभर प्रयोग हुने अधिकांश AI अंग्रेजीका लागि अनुकूलित छन्, जसले नेपाली जस्ता भाषाहरूलाई उल्लेखनीय रूपमा कम प्रतिनिधित्व गर्दछ। हाम्रो लक्ष्य देवनागरी र रोमनाइज्ड नेपाली दुवैमा उच्च गुणस्तरको corpus निर्माण गरेर यो समस्या समाधान गर्नु हो।

## हामीले के गर्न चाहन्छौं
हाम्रो लक्ष्य सरल छ: नेपालमा, नेपालीहरूद्वारा, नेपालको लागि नेपाली भाषाको open-source AI model बनाउने।

हाम्रो AI प्रणालीले निम्न राष्ट्रिय क्षमताहरू सक्षम पार्नेछ:
- नेपाली भाषीहरूले सरकारी सेवाहरू सजिलै पहुँच गर्न सक्नेछन् (जस्तै नागरिक App र Hello Sarkar)
- विद्यार्थी र शिक्षकहरूले स्थानीय पाठ्यक्रम अनुसार AI साथी पाउनेछन्
- दुर्गम क्षेत्रका डाक्टर र स्वास्थ्यकर्मीहरूले नेपालीमा AI सहयोग पाउनेछन्
- नेपाली startups र उद्यमहरूले आफ्ना AI अनुप्रयोगहरू बनाउन सक्नेछन्
- नेपालले आफ्नो भाषा र संस्कृति डिजिटल विश्वमा संरक्षण गर्नेछ
- नेपाल बाहिर नेपाली बोल्ने जनसंख्या (भारत, भुटान, म्यानमार) लाई AI सेवा निर्यात गर्न सकिनेछ
- देवनागरीमा अनुकूलित भएकाले मैथिली, भोजपुरी, नेवारी, अवधी, मराठी, हिन्दी आदि भाषाहरूमा पनि प्रतिस्पर्धी हुनेछ

## हामीले अहिलेसम्म गरेका कुराहरू
- Hugging Face मा नेपाली भाषा datasets प्रकाशित (huggingface.co/himalaya-ai)
- GitHub मा सार्वजनिक कोड (github.com/HimalayaGPT)
- नेपाली भाषा-अनुकूलित tokenizer तालिम गरिसकेका छौं
- Scalable corpus building को लागि modular data pipeline स्थापना गरिसकेका छौं
- 3B-8B parameters को model pretrain गर्न प्राविधिक रूपमा तयार छौं

## हाम्रो योजना
- Phase 1 (Early 2026): देवनागरी dataset 1T tokens तिर scale गर्ने
- Phase 2 (Mid 2026): 3B-8B parameters को HimalayanGPT training सुरु (300k-500k GPU hours, H100)
- Phase 3 (Late 2026): अनुवाद र प्रश्नोत्तर जस्ता नेपाली कार्यहरूको लागि fine-tune
- Phase 4 (Early 2027): HimalayanGPT v1.0 रिलीज — सबैका लागि निःशुल्क, open weights र open datasets
- Phase 5 (2027+): Multi-modal support (speech, images) र थप नेपाली भाषाहरू

## हाम्रो टोली
HARL स्वयंसेवक-सञ्चालित समुदाय हो जसमा अनुसन्धानकर्ता, software developers, ML engineers, विद्यार्थी, भाषा विशेषज्ञ, र नेपाल भित्र र बाहिरका नेपाली AI उत्साहीहरू छन्।

## सम्पर्क
Himalaya AI Research Lab
Email: himalaya.ai.lab@gmail.com
Phone: +977 9761412963
huggingface.co/himalaya-ai | github.com/HimalayaAI | x.com/HimalayaAILab
"""

# Initialize Discord Bot
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content
bot = discord.Client(intents=intents)

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

    # Only respond to messages starting with !chat
    if not message.content.startswith('!chat'):
        return

    # Extract the user's message after the !chat prefix
    user_input = message.content[len('!chat'):].strip()
    if not user_input:
        await message.channel.send("कृपया आफ्नो प्रश्न लेख्नुहोस्। उदाहरण: `!chat नमस्ते`")
        return

    async with message.channel.typing():
        try:
            # Prepare messages for Sarvam AI
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]

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
                ai_response = "I couldn't generate a response."

            # Send the response back to Discord
            if ai_response:
                # Split message if it exceeds Discord's 2000 character limit
                for i in range(0, len(ai_response), 2000):
                    await message.channel.send(ai_response[i:i+2000])

        except Exception as e:
            print(f"Error calling Sarvam AI: {e}")
            await message.channel.send("Sorry, I encountered an error while processing your request.")

if __name__ == "__main__":
    if not DISCORD_TOKEN or not SARVAM_API_KEY:
        print("Error: DISCORD_TOKEN and SARVAM_API_KEY must be set in your .env file.")
    else:
        # Run the Discord bot
        bot.run(DISCORD_TOKEN)
