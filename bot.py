import os

from discord.ext import commands
from discord import Intents

bot = commands.Bot(command_prefix="%", help_command=None)

# loading cogs
bot.load_extension("cogs.events")
bot.load_extension("cogs.chatbot")

bot.run(os.environ["DISCORD_TOKEN"])
