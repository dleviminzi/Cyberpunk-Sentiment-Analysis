import datetime as dt
import json
import sys
import pandas as pd
import calendar as cal
from datetime import datetime
from collections import defaultdict
from time import gmtime, sleep, strftime
from psaw import PushshiftAPI
sys.path.insert(1, '../utils')
from utilities import connect_rds, register_praw

class Scraper:
    def __init__(self, game_name, release_date, subreddit, date_interval):
        # initialize pushshift and aws rds connections
        self.reddit = PushshiftAPI(register_praw())
        self.rds_db, self.cursor = connect_rds(True)

        # initialize game details
        self.game_name = game_name
        self.release_date = release_date
        self.subreddit = subreddit

        # dictionary where we will hold post urls
        self.urls = {}

        self.timestamps = self.generate_dates(date_interval)


    def generate_dates(self, dt_intvl):
        ''' 
        This method will take the release date and generate time stamps for 
        the dt_intvl days prior to release and the dt_intvl days after release 
        '''

        spl_rel = self.release_date.split('-')

        # get release date in datetime form and set start/end of interval
        rel_dt = dt.date(int(spl_rel[0]), int(spl_rel[1]), int(spl_rel[2]))
        curr = rel_dt - dt.timedelta(dt_intvl)
        end = rel_dt + dt.timedelta(dt_intvl)

        stamps = []     # list of timestamps to be returned

        while curr <= end:      # iterate through appending stamps
            stamps.append(cal.timegm(curr.timetuple()))
            curr += dt.timedelta(1)
        
        return stamps


    ############################# POST SCRAPING ###############################
    def scrape_posts(self):
        '''
        Collects valid posts and places them into a dictionary that will be 
        used with reddit api to collect details on each url.
        '''
        log_count = 0   # tracking how many posts we're getting 

        for i in range(len(self.timestamps) - 1):
            after = self.timestamps[i]       # this translates to one day at a time
            before = self.timestamps[i+1]

            top_posts = self.reddit.search_submissions(after=after,
                                               before=before,
                                               filter=['url', 'score'],
                                               sort_type='score',
                                               subreddit=self.subreddit)

            for post in top_posts:
                if len(post.url) > 10 and post.url[8] == 'i':
                    continue        # image posts cause problems down the line
                elif "reddit" not in post.url:
                    continue

                date = dt.datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                url = post.url
                upvotes = post.score

                # storing entry into the dictionary
                self.urls[url] = [date, self.game_name, upvotes]

                log_count += 1
                sys.stdout.write("\rPosts logged: {} Current date: {}".format(log_count, date))

        # save in case we wish to resume at comment scraping
        df = pd.DataFrame.from_dict(self.urls)
        df.to_csv('./urls.csv', index=False)


    ############################ COMMENT SCRAPING #############################
    def log_comment(self, is_parent, parent_url, parent_upvotes, comment_upvotes, 
                    date, body, log_count):
        '''
        Takes a comment and inserts it into our database after performing some
        final checks and classifying it as the parent post.
        '''
        if body == "[deleted]" or body == "[removed]":
            return

        if is_parent:
            body = "PARENT: " + body

        self.cursor.execute("""INSERT INTO CyberPunkComments (
                                parent_url,parent_upvotes,comment_upvotes, 
                                _date,game,body)
                                values(%s, %s, %s, TIMESTAMP %s, %s, %s);""",
                                (parent_url,parent_upvotes,comment_upvotes,
                                date,self.game_name,body[:4560]))

        sys.stdout.write("\rComments logged: {} \n Current date: {}".format(log_count, date))


    def scrape_comments(self):
        '''
        Goes through our urls and sends the parent and all subsequent comments
        off to the logger.
        '''

        r = register_praw()

        log_count = 0
        for url in self.urls:
            p_upvotes = self.urls[url][2]       # parent submission details
            date = self.urls[url][0]

            try:
                post = r.submission(url=url)
            except:
                continue    # catch occasional odd urls 

            if post.selftext:       # logging the parent post
                log_count += 1
                self.log_comment(True, url, p_upvotes, post.score, 
                                 date, post.selftext, log_count)

                self.rds_db.commit()
            
            # replace "more comments" with nothing (avoid type error)
            post.comments.replace_more(limit=0)

            for comment in post.comments:
                if len(comment.body) > 75:
                    log_count += 1
                    self.log_comment(False, url, p_upvotes, comment.score, date,
                                     comment.body, log_count) 


if __name__ == "__main__":
    # optionally change to 
    s = Scraper("Cyberpunk 2077", "2020-12-10", "cyberpunkgame", 
                date_interval=30)

    s.scrape_posts() 
    s.scrape_comments()