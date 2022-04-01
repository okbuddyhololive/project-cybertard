import collections
import functools
import random

import discord
from discord.ext import commands

from cogs import utils
from model import Inference, ModelParams


class Chatbot(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

        self.model = Inference(parameters=ModelParams(**self.bot.config["Model"]), config=self.bot.infer_config)
        self.prompt = collections.defaultdict(str)
        self.previous_responses = collections.defaultdict(list)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot and message.author != self.bot.user:
            return

        channel_id = message.channel.id
        if str(channel_id) not in self.bot.infer_config.channels.split(';'):
            return

        content = message.clean_content
        if message.author == self.bot.user and utils.is_auto_message(content):
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
        function = functools.partial(self.model.generate, prompt=messages + f"<{self.bot.infer_config.name}>:")
        # partial function needed for async

        async with message.channel.typing():
            for _ in range(self.bot.infer_config.max_response_retries):
                response = await self.bot.loop.run_in_executor(None, function)

                response = response.split("\n-----\n")[0]

                if not response:
                    continue

                response = utils.replace_ping(self.bot.infer_config.name, self.bot.user.id, response)
                response = utils.fix_emojis_and_mentions(response, self.bot.users, self.bot.emojis)

                if utils.filter_message(response):
                    continue

                prev_responses = self.previous_responses[channel_id]
                if prev_responses.count(response) > self.bot.infer_config.max_same_replies:
                    continue

                prev_responses.append(response)
                self.previous_responses[channel_id] = prev_responses[-self.bot.infer_config.same_reply_saved_messages:]
                await message.reply(response, mention_author=False)
                break


def setup(bot):
    bot.add_cog(Chatbot(bot))
