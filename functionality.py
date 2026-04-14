import asyncio
import re

class functional:
    def __init__(self, bot):
        self.bot = bot
        self.user_input = None

    async def get_message_history(self, channel, limit=6):
        """Fetches previous messages including the bot's own responses for context."""
        messages = []
        # Fetch history (newest first)
        async for msg in channel.history(limit=limit, oldest_first=False):
            # Include the bot's own messages and messages from real users
            if msg.author == self.bot.user or not msg.author.bot:
                messages.append(msg)
        
        # Reverse to maintain chronological order (oldest to newest)
        messages.reverse() 
        return messages

    async def call(self, message):
        """Checks if the bot was pinged or the !chat command was used."""
        if message.author == self.bot.user:
            return

        self.user_input = None
        
        # Scenario A: User uses !chat prefix
        if message.content.startswith('!chat'):
            self.user_input = message.content[len('!chat'):].strip()
            # If they just typed "!chat" without a question
            if not self.user_input:
                await message.channel.send("कृपया आफ्नो प्रश्न लेख्नुहोस्। उदाहरण: `!chat नमस्ते`")
                return

        # Scenario B: User mentions/pings the bot
        elif self.bot.user.mentioned_in(message):
            self.user_input = message.content
            # Remove the @mention tag from the text so the AI doesn't see raw IDs
            self.user_input = re.sub(rf'<@!?{self.bot.user.id}>', '', self.user_input).strip()
