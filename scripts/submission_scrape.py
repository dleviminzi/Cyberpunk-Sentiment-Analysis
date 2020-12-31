from psaw import PushshiftAPI
from time import sleep, strftime, gmtime
from collections import defaultdict
from utilities import register_praw, connect_rds
import pandas as pd
import json
import datetime as dt

interval = 15
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
    start = int(dt.datetime(year, month, day).timestamp() - days_before_release)

    return start


def snag_posts(row):
    # pushshift upvote's are incorrect w/o praw (backend?)
    # register with reddit api
    r = register_praw()

    # initialize push shift reddit api
    reddit = PushshiftAPI(r)

    rds_conn, cur = connect_rds(True)

    # shitty way to track duplicates
    urls = set()

    game = row['game']
    print("Beginning scraping for: ", game)
    subreddits = row['subreddits'].split(', ')
    release_date = str(row['release_date']).split(' ')[0].split('-')

    # get interval of dates to check
    start = gen_interval(release_date)

    log_count = 0

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
                try:
                    if post.url[8] == 'i' or post.score < 10:
                        continue
                except IndexError:
                    continue

                
                # get points to enter into post table
                date = dt.datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                url = post.url
                upvotes = post.score

                if url in urls:
                    continue

                # note this is done inefficiently by design to slow calls to psaw api
                # if I do batch writes, the api will bounce me for make too many requests
                cur.execute("""INSERT INTO Posts (url,upvotes,game,date)
                               values(%s, %s, %s, TIMESTAMP %s);""", (url, upvotes, game, date))
                rds_conn.commit()
                log_count += 1
                print("Posts logged: ", log_count)
                urls.add(url)


def submission_scrape():
    # connect to postgre db
    rds_conn = connect_rds(False)

    # collect games that are from 2020
    fetch_cmd = """SELECT game, release_date, subreddits FROM Games 
                   WHERE release_date >= TO_TIMESTAMP('2020-01-01', 'yyyy-mm-dd');"""

    # read results into dataframe 
    df = pd.read_sql_query(fetch_cmd, rds_conn)

    # internet went out so skip bops cold war cus that one was finished
    for _ in range(18):
        df = df.drop(df.index[0])

    rds_conn = None
    df.apply(snag_posts, axis=1)


def main():
    submission_scrape()


if __name__ == "__main__":
    main()
