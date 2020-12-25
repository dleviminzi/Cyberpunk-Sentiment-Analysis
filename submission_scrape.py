from psaw import PushshiftAPI
from time import sleep, strftime, gmtime
from collections import defaultdict
import pandas as pd
import json
import datetime as dt
import praw

interval = 10
one_day = 86400

def gen_interval(release_date):
    # cast year, month, day from df date entry (type: str)
    year = int(release_date[0])
    month = int(release_date[1])
    day = int(release_date[2])

    # TODO: determine reasonable range of dates
    # one option might be to overshoot and then only start counting when the
    # number of mentions reaches some threshold (maybe 1/10 of release day?)
    days_before_release = interval * one_day
    days_after_release = interval * one_day
    start = int(dt.datetime(year, month, day).timestamp() - days_before_release)
    end = int(dt.datetime(year, month, day).timestamp() + days_after_release)

    return start, end


def submission_scrape():
    # open list of games that we will observe
    df = pd.read_csv("./raw_data/games2010.csv")

    # pushshift upvote's are incorrect w/o praw (backend?)
    creds = {}
    with open('./reddit_creds.json') as f:
        creds = json.load(f)
        
    # register with reddit api
    r = praw.Reddit(client_id=creds['client_id'],
                     client_secret=creds['client_secret'],
                     user_agent=creds['user_agent'],
                     username=creds['username'],
                     password=creds['password'])

    # initialize push shift reddit api
    reddit = PushshiftAPI(r)

    # test case
    subreddits = df['Subreddits'][68].split(', ')
    release_date = df['Date'][68].split(', ')

    # get interval of dates to check
    start, end = gen_interval(release_date)

    # dict to be dumped into json upon completion
    stored = defaultdict(list)

    for subreddit in subreddits:
        for i in range(2*interval):
            s = start + i*one_day
            e = s + one_day

            # retrieve top comments for the day in this subreddit
            top_posts = reddit.search_submissions(after=s, 
                                                before=e,
                                                filter=['url', 'score'],
                                                sort_type='score',
                                                subreddit=subreddit)

            # format url and place into 
            for post in top_posts:
                # TODO: figure out if comments can be recovered from img posts'
                if post.url[8] == 'i' or post.score < 10:
                    continue

                human_time = strftime("%Y-%m-%d", gmtime(post.created_utc))

                # format to jump into json
                stored[human_time].append({
                    "url"       : post.url,
                    "upvotes"   : post.score,
                    "timestamp" : post.created_utc
                })

    with open('urls.json', 'w') as out:
        json.dump(stored, out)


def main():
    submission_scrape()


if __name__ == "__main__":
    main()
