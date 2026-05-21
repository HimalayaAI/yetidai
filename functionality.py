import re
class functional:
    def __init__(self, bot):
        self.bot = bot
        self.user_input = None
        self._processing: set[int] = set()  # track message IDs being processed


#<-----------------------------------memory---------------------------------------->

    async def get_message_history(self, channel, limit=5, *, include_bot_messages=True):
        messages = []
        async for msg in channel.history(limit=limit, oldest_first=False):
            # Include our own bot's messages optionally, but ignore other bots
            if msg.author == self.bot.user:
                # Skip embed-only messages (e.g. citation embeds the bot sends
                # after the main answer). These have no text content and would
                # just add noise to the history context.
                if include_bot_messages and (msg.content or "").strip():
                    messages.append(msg)
            elif not msg.author.bot:
                messages.append(msg)

        messages.reverse() 
        return messages
    

#<-----------------------------------check if bot is called---------------------------------------->

    async def call(self, message):
        """Parse the user's message and set self.user_input.

        Returns early (leaving user_input=None) if:
          - the author is the bot itself
          - the bot was not mentioned
          - the message is empty after stripping the mention
          - this exact message ID is already being processed (dedup guard)
        """
        if message.author == self.bot.user:
            self.user_input = None
            return

        # Dedup guard: Discord can fire on_message more than once for the
        # same message in rare edge cases (reconnects, gateway replays).
        # Storing the message ID prevents processing the same event twice.
        if message.id in self._processing:
            self.user_input = None
            return
        self._processing.add(message.id)
        # Keep the set small — only need to remember recent IDs.
        if len(self._processing) > 200:
            self._processing.clear()

        self.user_input = None

        content = (message.content or "").strip()
        if not content:
            return

        if self.bot.user.mentioned_in(message):
            user_input = content
            for mention in message.mentions:
                if mention == self.bot.user:
                    user_input = re.sub(
                        rf'<@!?{self.bot.user.id}>',
                        '',
                        user_input,
                    ).strip()  # bot lai ping ra mention gareko msg filter garxa
        else:
            # Only respond to direct mentions/pings
            return

        if not user_input:
            return

        self.user_input = user_input
