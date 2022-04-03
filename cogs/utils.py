import re
from typing import List

from discord import Emoji, User

from model.constants import BAD_WORDS

_key = "[a-zA-Z0-9\.]+"
_AUTO_MESSAGE = re.compile(f"(Attribute `[a-zA-Z0-9.]+(` is `{_key}`|` does not exist in the inference config)|"
                           f"Successfully set `{_key}` from `{_key}` to `{_key}`|"
                           f"config:\n    {_key}: {_key})|"
                           f"Successfully added <{_key}> to the channel list|"
                           f"Successfully removed <{_key}> from the channel list|"
                           f"Channel <{_key}> is already in the list|"
                           f"Channel <{_key}> is not in the list|"
                           f"Invalid action `{_key}`|"
                           "You must specify a channel!"
                           )
_EMOJI_RE = re.compile(f"<?(:{_key}:)[0-9]+>?")


def replace_ping(original_name: str, new_id: int, text: str) -> str:
    return re.sub('@' + ''.join(f'({c.lower()}|{c.upper()})' for c in original_name), f'<@{new_id}>', text)


def is_auto_message(text: str) -> bool:
    return _AUTO_MESSAGE.match(text) is not None


def fix_emojis_and_mentions(text: str, users: List[User], emojis: List[Emoji]) -> str:
    for user in users:
        # text = replace_ping(user.name, user.id, text)
        text = text.replace(f"@{user.name}", f"<@{user.id}>")

    text = _EMOJI_RE.sub("\\g<1>", text)

    for emoji in emojis:
        if f"<:{emoji.name}:{emoji.id}>" in text:
            continue

        text = text.replace(f":{emoji.name}:", f"<:{emoji.name}:{emoji.id}>")

    return text


def filter_message(text: str) -> bool:
    text = text.lower()
    return any(word in text for word in BAD_WORDS)

def format_config_message(config: dict) -> str:
    text = "```\n"

    for key, value in config.items():
        text += f"{key}: {value}\n"

    text += "```"
    return text