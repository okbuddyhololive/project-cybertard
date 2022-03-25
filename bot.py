import os

from discord.ext import commands
from discord import Intents
import toml

with open("config.toml", "r") as file:
    config = toml.load(file)

bot = commands.Bot(command_prefix=config["prefix"], help_command=None)
bot.config = config  # for global access between cogs

# loading cogs
bot.load_extension("jishaku")
bot.load_extension("cogs.events")
bot.load_extension("cogs.chatbot")

bot.run(os.environ["DISCORD_TOKEN"])
