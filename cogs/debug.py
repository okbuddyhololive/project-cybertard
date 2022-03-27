from typing import Optional

from discord.ext import commands


class Debug(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    async def _get_config(self, ctx, key: str):
        key = key.lower()
        value = getattr(
            self.bot.infer_config,
            key,
            default=None
        )
        
        if value is None:
            return await ctx.send(f"Attribute `{key}` does not exist in the inference config")
        
        await ctx.send(f"Attribute `{key}` is `{value}`")
    
    async def _set_config(self, ctx, key: str, value):
        key = key.lower()
        original_value = getattr(
            self.bot.infer_config,
            key,
            default=None
        )
        
        if original_value is None:
            return await ctx.send(f"Attribute `{key}` does not exist in the inference config")
        
        # since type(x) returns a class for a variable type, 
        # you can call it to convert an another value
        value = type(original_value)(value)
        
        setattr(
            self.bot.infer_config,
            key,
            value
        )
        
        await ctx.send(f"Succesfully set `{key}` from `{original_value}` to `{value}`")
        
    @commands.command()
    @commands.is_owner()
    async def config(self, ctx, key: str, *, value=None):
        if value is None:
            return await self._get_config(ctx, key)
        
        await self._set_config(ctx, key, value)
        

def setup(bot):
    bot.add_cog(Debug(bot))