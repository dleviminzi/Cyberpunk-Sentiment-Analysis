import praw
import json

def register_praw():
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

    return r