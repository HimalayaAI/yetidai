import asyncio
import re
class functional:
    def __init__(self, bot):
        self.bot = bot
        self.user_input = None


#<-----------------------------------memory---------------------------------------->

    async def get_message_history(self, channel, limit=5):
        messages = []
        async for msg in channel.history(limit=limit, oldest_first=False):
            # Include our own bot's messages, but ignore other bots
            if msg.author == self.bot.user or not msg.author.bot:
                messages.append(msg)

        messages.reverse() 
        return messages
    

#<-----------------------------------check if bot is called---------------------------------------->

    async def call(self, message):
        if message.author == self.bot.user:
            return

        self.user_input= None

        content = (message.content or "").strip()
        if not content:
            return

        if self.bot.user.mentioned_in(message):
            self.user_input = content
            for mention in message.mentions:
                if mention == self.bot.user:
                    self.user_input = re.sub(
                        rf'<@!?{self.bot.user.id}>',
                        '',
                        self.user_input,
                    ).strip()  # bot lai ping ra mention gareko msg filter garxa
        else:
            # Only respond to direct mentions/pings
            return

        if not self.user_input:
            return
