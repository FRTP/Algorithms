import re

re_money = re.compile("\$( )?\d+([\.,]\d+)*[mMkKbB]?")

re_number = re.compile("\d+([\.,]\d+)*")

re_link = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
                     "[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

re_hashtag = re.compile(r"#(\w+)")

re_hndl = re.compile(r"@(\w+)")

re_word_bound = re.compile(r"\W+")

# detecting repetitions like "hurrrryyyyyy"
re_rpt = re.compile(r"(.)\1{1,}", re.IGNORECASE)


def hashtag_repl(match):
    return '__HASH_' + match.group(1).upper()


def hndl_repl(match):
    return '__HNDL'  # _'+match.group(1).upper()


def rpt_repl(match):
    return match.group(1) + match.group(1)
