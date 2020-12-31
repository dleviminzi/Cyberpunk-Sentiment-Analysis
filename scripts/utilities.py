import psycopg2 as ps
import praw
import json

def register_praw():
    # pushshift upvote's are incorrect w/o praw (backend?)
    creds = {}
    with open('../reddit_creds.json') as f:
        creds = json.load(f)
        
    # register with reddit api
    r = praw.Reddit(client_id=creds['client_id'],
                        client_secret=creds['client_secret'],
                        user_agent=creds['user_agent'],
                        username=creds['username'],
                        password=creds['password'])

    return r

def connect_rds(give_cursor):
    # connect up to postgre in rds
    creds = {}
    with open('../rds_creds.json') as f:
        creds = json.load(f)

    rds_conn = ps.connect(host=creds['POSTGRES_ADDRESS'],
                          database=creds['POSTGRES_DBNAME'],
                          user=creds['POSTGRES_USERNAME'],
                          password=creds['POSTGRES_PASSWORD'],
                          port=creds['POSTGRES_PORT'])

    cur = rds_conn.cursor()

    if give_cursor:
        return rds_conn, cur
    else:
        return rds_conn