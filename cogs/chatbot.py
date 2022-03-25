import functools
import re

from discord.ext import commands

import model


class Chatbot(commands.Cog):
    def __init__(self, bot, name: str = "denisplay", max_length: int = 2 ** 15):
        """

        :param bot: ?
        :param name: Name of the user it will generate text as. This helps keep styles consistent. Ideally, it's a name seen during training.
        :param max_length: Should be >4 * model_context_size as the GPT's BPE tokenizer encodes ~4 characters as 1 token and you don't want to cut off a token.
        """
        self.bot = bot
        self.model = model.Inference()
        self.messages = ""
        self.name = name
        self.max_length = max_length

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return

        # adding message to self.messages (for prompt generation)
        self.messages += f"<{message.author.name}>: {message.clean_content}\n-----\n"  # TODO: Extract split fn

        if len(self.messages) > self.max_length:
            self.messages = self.messages[-self.max_length:]

        messages = self.messages.replace("Project Cybertard", self.name)  # TODO: extract name from bot instance

        if self.bot.user in message.mentions:
            # partial function needed for async
            function = functools.partial(
                self.model.generate,
                prompt=messages + f"<{self.name}>:",
            )

            response = await self.bot.loop.run_in_executor(None, function)
            response = response.split('\n-----\n')[0]
            await message.reply(response, mention_author=False)


def setup(bot):
    bot.add_cog(Chatbot(bot))
