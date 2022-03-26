import collections
import functools
import random
import re

from discord.ext import commands

import model


class Chatbot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.model = model.Inference(parameters=model.ModelParams(**self.bot.config["Model"]),
                                     config=model.InferConfig(**self.bot.config["Inference"]), )
        self.prompt = collections.defaultdict(str)

        self.name = self.bot.config["name"]
        self.max_length = self.bot.config["max_length"]

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot and message.author != self.bot.user:
            return

        channel_id = message.channel.id
        if channel_id not in  self.bot.config["channel_id"]:
            return

        # adding message to the prompt
        # TODO: Extract split fn
        self.prompt[channel_id] += f"<{message.author.name}>: {message.clean_content}\n-----\n"
        prompt = self.prompt[channel_id]

        if len(prompt) > self.max_length:
            self.prompt[channel_id] = prompt = prompt[-self.max_length:]

        messages = prompt.replace(self.bot.user.name, self.name)

        if message.author != self.bot.user:
            if self.bot.user in message.mentions or random.random() < self.bot.config["response_probability"]:
                # partial function needed for async
                function = functools.partial(self.model.generate, prompt=messages + f"<{self.name}>:", )

                async with message.channel.typing():
                    response = await self.bot.loop.run_in_executor(None, function)
                    response = response.split("\n-----\n")[0]
                    response = response.replace(f'@{self.name}', f'<@{self.bot.user.id}>')
                    for user in self.bot.users:
                        response = response.replace(f'@{user.name}', f'<@{user.id}>')
                if response:
                    await message.reply(response, mention_author=False)


def setup(bot):
    bot.add_cog(Chatbot(bot))
