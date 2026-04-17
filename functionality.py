import asyncio
import re
from typing import Optional, List

class Functional:
    def __init__(self, bot):
        self.bot = bot
        self.user_input: Optional[str] = None

    # ====================== MEMORY ======================
    async def get_message_history(self, channel, limit: int = 10) -> List[str]:
        """
        Gets recent user messages from the channel for context.
        Ignores bot messages and system messages.
        """
        messages = []
        
        async for msg in channel.history(limit=limit, oldest_first=False):
            # Skip bot's own messages and other bots
            if msg.author == self.bot.user or msg.author.bot:
                continue
                
            # Clean the message content (remove mentions, extra spaces)
            content = msg.content.strip()
            if content:
                # Optional: You can format it as "Username: message"
                formatted = f"{msg.author.display_name}: {content}"
                messages.append(formatted)
        
        # Reverse so oldest message comes first (better for context)
        messages.reverse()
        return messages

    # ====================== CHECK IF BOT IS CALLED ======================
    async def call(self, message):
        """
        Checks if the bot should respond to this message.
        Returns the cleaned user input if triggered, else None.
        """
        if message.author == self.bot.user or message.author.bot:
            return None

        self.user_input = None

        # Command: !chat
        if message.content.startswith('!chat'):
            self.user_input = message.content[len('!chat'):].strip()

        # Mention: @Bot
        elif self.bot.user.mentioned_in(message):
            self.user_input = message.content
            # Remove the bot mention cleanly
            for mention in message.mentions:
                if mention == self.bot.user:
                    self.user_input = re.sub(rf'<@!?{self.bot.user.id}>', '', self.user_input).strip()
                    break  # No need to check further

        # If no valid input found
        if not self.user_input:
            if message.content.startswith('!chat'):
                await message.channel.send(
                    "कृपया आफ्नो प्रश्न लेख्नुहोस्।\n"
                    "उदाहरण: `!chat नमस्ते कस्तो छ?`"
                )
            return None

        return self.user_input
