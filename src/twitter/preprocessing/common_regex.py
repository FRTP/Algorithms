import re

re_money = re.compile(r"\$( )?\d+([\.,]\d+)*[mMkKbB]?")

re_number = re.compile(r"\d+([\.,]\d+)*")

re_url = re.compile(r"(http[s]?|ftp)://[\w/$.?#]+.?\w*$@[\w.1-9]*")


re_hashtag = re.compile(r"#(\w+)")

re_user = re.compile(r"@(\w+)")

# detecting repetitions like "hurrrryyyyyy"
re_repeat = re.compile(r"\b(\w*?)(.)\2{2,}\b", re.IGNORECASE)

# detecting sign repetitions
re_sign_repeat = re.compile(r"([!?.]){2,}")

re_allcaps = re.compile(r"\b([A-Z]{2,})\b")


# TODO: replace '/' with ' / '

def money_repl(match):
    return ' <NUMBER> money '


def number_repl(match):
    return ' <NUMBER> '


def url_repl(match):
    return ' <URL> '


def hashtag_repl(match):
    # TODO: consider breaking by capital letters
    return ' <HASHTAG> ' + " ".join(match.group(1).split())


def user_repl(match):
    return ' <USER> '  # _'+match.group(1).upper()


def repeat_repl(match):
    return match.group(1) + match.group(2) + ' <ELONG> '


def sign_repeat_repl(match):
    return ' ' + match.group(1) + ' <REPEAT> '


def allcaps_repl(match):
    return ' ' + match.group(1).lower() + ' <ALLCAPS> '


# emoticons replacements

def smile_repl(match):
    return ' <SMILE> '


def sadface_repl(match):
    return ' <SADFACE> '


def lolface_repl(match):
    return ' <LOLFACE> '


def neutral_repl(match):
    return ' <NEUTRALFACE> '


def heart_repl(match):
    return ' <HEART> '


def re_from_list(list):
    escaped_list = [re.escape(item) for item in list]
    return re.compile(r"(" + "|".join(escaped_list) + ")")
