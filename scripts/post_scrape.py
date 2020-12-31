from time import sleep, strftime, gmtime
from collections import defaultdict
import datetime as dt
import json
import sys
sys.path.insert(1, '/Users/dminzi-lt/dev/games_2020/scripts')
from utilities import register_praw, connect_rds
import pandas as pd

def log_comment(is_parent, cur, parent_url, parent_upvotes, 
                comment_upvotes, date, game, body):
    if body == "[deleted]" or body == "[removed]":
        return


    if is_parent:
        body = "PARENT: " + body

    cur.execute("""INSERT INTO Comments (
                       parent_url, 
                       parent_upvotes, 
                       comment_upvotes, 
                       _date, 
                       game, 
                       body
                   )values(%s, %s, %s, TIMESTAMP %s, %s, %s);""",(
                       parent_url, 
                       parent_upvotes, 
                       comment_upvotes,
                       date, 
                       game, 
                       body[:4560]
                   )
    )


def process_url(row, reddit, rds_conn, cur):
    parent_url = row['url']

    # skip non-reddit urls
    if "reddit" not in parent_url:
        return        

    parent_upvotes = row['upvotes']
    game = row['game']
    date = row['date']
    
    # fetch post data
    try:
        post = reddit.submission(url=parent_url)
    except:     # on account of the whims of praw
        return

    # log the post itself (marked as parent of thread)
    if post.selftext:
        log_comment(True, cur, parent_url, parent_upvotes, 
                    post.score, date, game, post.selftext)
        rds_conn.commit()
        print(game, " - post logged")

    # replace "more comments" with nothing (avoid type error)
    post.comments.replace_more(limit=0)
    log_count = 0

    # log the comments on the parent post and report the number logged
    for comment in post.comments:
        if len(comment.body) > 150:     # min length is arbitrary atm
            log_count += 1
            log_comment(False, cur, parent_url, parent_upvotes, 
                        comment.score, date, game, comment.body)

    if log_count:
        rds_conn.commit()
        print(game, " - comments logged: ", log_count)


def main():
    # register with aws
    rds_conn, cur = connect_rds(True)

    # register with praw
    reddit = register_praw()

    # load posts into dataframe 
    fetch_cmd = """SELECT * FROM Posts
                   ORDER BY game"""

    df = pd.read_sql_query(fetch_cmd, rds_conn)



    # process the urls in the dataframe
    df.apply(process_url, args=(reddit, rds_conn, cur), axis=1)


if __name__ == "__main__":
    main()
