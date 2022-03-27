import os

from discord.ext import commands
from discord import Intents
import toml

from model import InferConfig

with open("config.toml", "r") as file:
    config = toml.load(file)

bot = commands.Bot(intents=discord.Intents.all(), command_prefix=config["prefix"], help_command=None)

bot.config = config  # for global access between cogs
bot.infer_config = InferConfig(**bot.config["Inference"])

# loading cogs
bot.load_extension("cogs.events")
bot.load_extension("cogs.chatbot")
bot.load_extension("cogs.debug")

bot.load_extension("jishaku")

bot.run(os.environ["DISCORD_TOKEN"])