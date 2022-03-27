import collections
import functools
import random
import re

from discord.ext import commands
import discord

from model import Inference, ModelParams

_key = "[a-zA-Z0-9\.]+"
_AUTO_MESSAGE = re.compile(f"(Attribute `[a-zA-Z0-9.]+(` is `{_key}`|` does not exist in the inference config)|"
                           f"Successfully set `{_key}` from `{_key}` to `{_key}`|"
                           f"config:\n    {key}: {key})")


class Chatbot(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.model = Inference(parameters=ModelParams(**self.bot.config["Model"]), config=self.bot.infer_config, )
        self.prompt = collections.defaultdict(str)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot and message.author != self.bot.user:
            return

        channel_id = message.channel.id
        if channel_id not in self.bot.config["channels"]:
            return

        content = message.clean_content
        if message.author == self.bot.user and _AUTO_MESSAGE.match(content):
            return

        # adding message to the prompt
        # TODO: Extract split fn
        self.prompt[channel_id] += f"<{message.author.name}>: {content}\n-----\n"
        prompt = self.prompt[channel_id]

        if len(prompt) > self.bot.infer_config.prompt_length:
            self.prompt[channel_id] = prompt = prompt[-self.bot.infer_config.prompt_length:]

        if message.author == self.bot.user:
            return

        if self.bot.user not in message.mentions and random.random() >= self.bot.infer_config.response_probability:
            return

        messages = prompt.replace(self.bot.user.name, self.bot.infer_config.name)

        async with message.channel.typing():
            # partial function needed for async
            function = functools.partial(self.model.generate, prompt=messages + f"<{self.bot.infer_config.name}>:")
            response = await self.bot.loop.run_in_executor(None, function)
            response = response.split("\n-----\n")[0]
            response = re.sub('@' + ''.join(f'({c.lower()}|{c.upper()})' for c in self.bot.infer_config.name),
                              f'<@{self.bot.user.id}>', response)
            # replacing names with actual mentions so that mentions actually work
            for user in self.bot.users:
                response = response.replace(f"@{user.name}", f"<@{user.id}>")

        if response:
            await message.reply(response, mention_author=False)


def setup(bot):
    bot.add_cog(Chatbot(bot))
