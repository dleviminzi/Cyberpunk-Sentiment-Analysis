from time import sleep, strftime, gmtime
from collections import defaultdict
from utilities import register_praw
import datetime as dt
import json

def get_post(url, r):
    thread = r.submission(url=url)
    return thread


def collect_comments(submission):
    for num, top_level_comment in enumerate(submission.comments):
        print(num, top_level_comment.body)


def main():
    urls = {}

    with open("./raw_data/cyberpunk_urls.json") as f:
        urls = json.load(f)

    url = urls["2020-11-30"][0]["url"]
    r = register_praw()

    collect_comments(get_post(url, r))


if __name__ == "__main__":
    main()
