from discord.ext import commands


class Events(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_ready(self):
        print(f"Logged in as {self.bot.user}!")

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.CommandNotFound):
            return

        # TODO: look into it a bit more and maybe implement something with `logging`

        print(error)
        await ctx.send(f"```\n{error}\n```")


def setup(bot):
    bot.add_cog(Events(bot))
