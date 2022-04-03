from typing import Optional

from discord.ext import commands
from discord import TextChannel

from cogs import utils

class Debug(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # config stuff
    async def _get_config(self, ctx, key: str):
        key = key.lower()
        value = getattr(self.bot.infer_config, key, None)

        if value is None:
            return await ctx.send(f"Attribute `{key}` does not exist in the inference config")

        await ctx.send(f"Attribute `{key}` is `{value}`")

    async def _set_config(self, ctx, key: str, value):
        key = key.lower()
        original_value = getattr(self.bot.infer_config, key, None)

        if original_value is None:
            return await ctx.send(f"Attribute `{key}` does not exist in the inference config")

        # since type(x) returns a class for a variable type, 
        # you can call it to convert an another value
        value = type(original_value)(value)
        setattr(self.bot.infer_config, key, value)

        await ctx.send(f"Successfully set `{key}` from `{original_value}` to `{value}`")

    @commands.command()
    @commands.is_owner()
    async def config(self, ctx, key: Optional[str] = None, *, value=None):
        if key is None or key.lower() == "list":
            return await ctx.send(utils.format_config_message(self.bot.infer_config.__dict__))     
        if value is None:
            return await self._get_config(ctx, key)
        
        await self._set_config(ctx, key, value)

    # channel whitelist stuff
    async def _add_channel(self, ctx, channel: TextChannel):
        if channel.id in self.bot.config["channels"]:
            return await ctx.send(f"Channel <#{channel.id}> is already in the list")

        self.bot.config["channels"].append(channel.id)

        await ctx.send(f"Successfully added <#{channel.id}> to the channel list")
    
    async def _remove_channel(self, ctx, channel: TextChannel):
        if channel.id not in self.bot.config["channels"]:
            return await ctx.send(f"Channel <#{channel.id}> is not in the list")

        self.bot.config["channels"].remove(channel.id)

        await ctx.send(f"Successfully added <#{channel.id}> to the channel list")
    
    @commands.command(aliases=["whitelist", "channel"])
    @commands.is_owner()
    async def channels(self, ctx, action: Optional[str] = None, channel: Optional[TextChannel] = None):
        if action is None or action.lower() == "list":
            return await ctx.send(utils.format_channel_message(self.bot.config["channels"]))
        
        if channel is None:
            return await ctx.send("You must specify a channel!")
        
        if action.lower() in ("add", "append"):
            return await self._add_channel(ctx, channel)
        if action.lower() in ("remove", "delete"):
            return await self._remove_channel(ctx, channel)

        await ctx.send(f"Invalid action `{action}`")

def setup(bot):
    bot.add_cog(Debug(bot))
