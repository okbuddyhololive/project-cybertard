from discord import Emoji, User

from typing import List
import re

_key = "[a-zA-Z0-9\.]+"
_AUTO_MESSAGE = re.compile(f"(Attribute `[a-zA-Z0-9.]+(` is `{_key}`|` does not exist in the inference config)|"
                           f"Successfully set `{_key}` from `{_key}` to `{_key}`|"
                           f"config:\n    {_key}: {_key})")
_EMOJI_RE = re.compile(f"<(:{_key}:)[0-9]+>")

def replace_ping(original_name: str, new_id: int, text: str) -> str:
    return re.sub('@' + ''.join(f'({c.lower()}|{c.upper()})' for c in original_name), f'<@{new_id}>', text)


def is_auto_message(text: str) -> bool:
    return _AUTO_MESSAGE.match(text) is not None


def fix_emojis_and_mentions(text: str, users: List[User], emojis: List[Emoji]) -> str:
    for user in users:
        #text = replace_ping(user.name, user.id, text)
        text = text.replace(f"@{user.name}", f"<@{user.id}>")

    text = _EMOJI_RE.replace("\\g<1>", text)

    for emoji in emojis:
        if f"<:{emoji.name}:{emoji.id}>" in text:
            continue
        
        text = text.replace(f":{emoji.name}:", f"<:{emoji.name}:{emoji.id}>")

    return text
