import streamlit as st
import pandas as pd
import altair as alt
import datetime as dt
from collections import defaultdict

df = pd.read_csv('./judged_cbp.csv')
df = df.sort_values('comment_upvotes', ascending=False)
df = df.reset_index()
df = df.drop(columns=['index', 'parent_upvotes','_date','pos_raw'])
df['pos_rting'] = df['pos_rting'] * 100

st.title("Sentiment analysis of the Cyberpunk 2077 subreddit before and after the worst release of all time")
st.markdown("""
github repo: https://github.com/dleviminzi/cyberpunk_sentiment_analysis
""")

st.header("Scrapping Posts and Comments from Reddit")

st.markdown("""
Before getting into things, let's discuss how the data was obtained and let's 
take a look at it. If you want to just see the code, go ahead and click on the 
expander below.
""")
with st.beta_expander('Scraper'):
    st.code("""
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
                    sys.stdout.write("\\rPosts logged: {} Current date: {}".format(log_count, date))

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

            self.cursor.execute(\"""INSERT INTO CyberPunkComments (
                                    parent_url,parent_upvotes,comment_upvotes, 
                                    _date,game,body)
                                    values(%s, %s, %s, TIMESTAMP %s, %s, %s);\""",
                                    (parent_url,parent_upvotes,comment_upvotes,
                                    date,self.game_name,body[:4560]))

            sys.stdout.write("\\rComments logged: {} \\n Current date: {}".format(log_count, date))


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
    """)
st.markdown('')
st.markdown("""
The scraper takes the name of the game CyberPunk 2077, the release date, an 
interval size, and a target subreddit. The interval indicates how many days 
before and after release you are interested in scraping data for. For each of those
days, psaw is used to attempt to gather the top posts of the day. Then for each 
top post, praw is used to gather the top comments. Note, that we can only "attempt"
to gather the top posts for a day because psaw only stores snapshots. This means 
that the number of upvotes it reports is not entirely reliable. 
""")
st.markdown("""
Below you can take a look at plot of comments per day. Two things are clear:

1. As expected, the amount of activity in the subreddit was higher during the 
days surrounding the release.

2. The scraper struggled to find data after ~12/20.
""")
comment_count = {}
for day in sorted(df['day'].unique()):
    comment_count[day] = len(df[df['day'] == day])

df1 = pd.DataFrame.from_dict(comment_count, orient='index')

# grouping days into quantiles
df1 = df1.reset_index()
df1 = df1.rename(columns={'index':'date', 0: 'comment_count'})
twofive = df1.quantile(q=0.25)[0]
sevfive = df1.quantile(q=0.75)[0]

def mark_quantile(row):
    if row['comment_count'] > twofive and row['comment_count'] < sevfive:
        return 'middle'
    elif row['comment_count'] < twofive:
        return 'lower'
    else: 
        return 'upper'

df1['quantile'] = df1.apply(mark_quantile, axis=1)

comments_per_day = alt.Chart(df1, title="Comments per day").mark_point().encode(
    x='date',
    y='comment_count',
    color=alt.Color('quantile', scale=alt.Scale(scheme='set1'))
).properties(width=700, height=300)

st.write(comments_per_day)
st.markdown("""
At first, I thought that one of the APIs must have enacted a limit on my requests.
However, even after re-running the scraper on more concentrated intervals, this 
trend remained. If the drop had only lasted through christmas, I might've suggested
that people spent less time on reddit during the holidays. Yet, the drop persisted 
beyond holidays and personally I much prefer reddit to spending time with my family (a joke!). 
I will continue to investigate, but at the moment I can only attribute it to some 
misunderstanding on my part or failure on the part of the API. In either case,
for now those days will not be considered. 
""")
df = df[df['day'] < '2020-12-21']

st.header("Cleaning the Text")
st.markdown("""
We must clean the text before passing it to our model. To do this, we remove the 
"PARENT" flag, links, and bot comments. Additionally, redditors who receive lots
of upvotes tend to append an edit to their post that says how grateful they are. 
In the case where the comment itself was bashing the game, this can lead to some
uncertainty during inference. Thus if a comment is found to contain "edit: blah...", 
we will truncate it to only include the writing before that. Finally, this model 
can only handle inputs of 512 tokens are less, so some excessively long comments
will have to be truncated. For comments that are too long, I opted to take the slice
starting from the end because I imagine conclusions are more informative than introductions.
""")

with st.beta_expander('Example of raw and cleaned text'):
    col_dirty, col_clean = st.beta_columns(2)

    with col_dirty:
        st.markdown("##### Raw Text")
        st.markdown(df['body'][1]) 
    with col_clean:
        st.markdown("##### Cleaned Text")
        st.markdown(df['clean_body'][1])


st.header("Most Upvoted Comments")
st.markdown("""
Let's take a look at the comments(pre-cleaning) that received the most 
upvotes during the interval we pulled. They are funny and tragic.
""")
num = st.slider("Number of top comments to display (max=100)", 0, 100, 3)
st.write(df.drop(columns=['prnt', 'clean_body']).head(num))
st.write("""
We can also get a nice visualization of when these comments were posted. As 
expected, the mysterious drop in data is pretty clear here again.
""")
c = alt.Chart(df, title="When top comments occur").mark_circle().encode(
    x='day', 
    y='pos_rting', 
    size='comment_upvotes', 
    color='comment_upvotes', 
    tooltip=['day', 'pos_rting', 'comment_upvotes']).properties(width=700, height=300)

st.write(c)

st.header("Positivity Rating")
st.markdown("""
To be brief, the positivity rating was generated by training the HuggingFace Bert
for sequence classification model on a bunch of steam reviews. There are more details
in the section titled "model" below. When using the model for inference, we need
to tokenize the input data and then recover the logits returned after passing it 
into the model. At that point, we can just use the softmax function ($\sigma(\\vec{\\textbf{z}})_i = \\frac{e^{z_i}}{\\sum_{j=1}^K e^{z_j}}$)
to recover probabilities.
""")
st.markdown("""
Because the task only has two classes (positive or negative) we can arbitrarily 
choose which rating we will use. In this case, I select the positivity probability.
This means that we are looking at the probability that the comment has positive
sentiment. So, high rating $\implies$ positive sentiment. 
""")

st.markdown("## Analyzing Positivity Rating")
st.markdown("""
### Average Positivity Score Per Day (a simple approach)
The simplest way to see how the positivity changed around launch is to take the 
average positivity score for each day. However, we must be careful because our 
classifier is not always all that confident. We can use a threshold that way we 
only count comments for which the probability of positive/negative sentiment 
is above some number. For instance if we choose a threshold of 60% and a comment
has a positivity rating of 55%, it will not be included in the graph. You can 
select a threshold below and see how it impacts the visualization.
""")
thresh = st.slider("Select positivity threshold ", 0, 99, 95)
df_a = df[df['pos_rting'] >= thresh]
df_b = df[df['pos_rting'] <= 100 - thresh]
df_trimmed = pd.concat([df_a, df_b])
st.markdown("Original number of comments: {}".format(len(df)))
st.markdown("Remaining comments: {}".format(len(df_trimmed)))
st.write("""
Number of removed comments: {}""".format(len(df) - len(df_trimmed)))


main_ = df_trimmed.groupby('day').mean()

b = alt.Chart(main_.reset_index()).mark_line(point=True).encode(
    x='day',
    y='pos_rting'
).properties(width=700, height=400)
col1, col2 = st.beta_columns(2)
with col1:
    date = st.date_input("""Date""", value=dt.datetime(2020, 12, 11), 
                        min_value=dt.datetime(2020, 11, 18),
                        max_value=dt.datetime(2021, 1, 6))

with col2:
    st.markdown("""
    Positivity rating
    
    {}""".format(round(main_['pos_rting'].loc[str(date)], 2)))
st.write(b)

st.markdown("""
## Weighting by Number of Upvotes
The method above ignores one of our best indicators: the number of upvotes the 
comment received. As it stands, a positivity rating of 95% with 5 upvotes is 
just as relevant as a positivity rating of 5% with 1000 upvotes. This doesn't seem very fair.
""")
st.markdown("")
st.markdown("""
To try to address this we can weight each comment by the number of upvotes that 
it received. For each day, the positivity rating of each comment is counted the 
number of time that it was upvoted. Doing this we get the following plot.
""")
thresh2 = st.slider("Select positivity threshold  ", 0, 99, 95)
df_a2 = df[df['pos_rting'] >= thresh2]
df_b2 = df[df['pos_rting'] <= 100 - thresh2]
df_trimmed2 = pd.concat([df_a2, df_b2])
rting_log = defaultdict(list)

def rting(row):
    for i in range(row['comment_upvotes']):
        rting_log[row['day']].append(row['pos_rting'])
        if i == 10000:
            break


df_trimmed2.apply(rting, axis=1)
rting_log.keys()
rtings = {}

for day in rting_log.keys():
    av_rting = 0
    for rting in rting_log[day]:
        av_rting += rting
    rtings[day] = av_rting/len(rting_log[day])

df_2 = pd.DataFrame.from_dict(rtings,orient='index')
df_2 = df_2.reset_index()
df_2 = df_2.rename(columns={'index': 'day', 0: 'avg_rting'})
d = alt.Chart(df_2).mark_line(point=True).encode(
    x='day',
    y='avg_rting'
).properties(width=700, height=400)
st.write(d)

st.markdown("""
The results from using weighted comments show that the drop in positivity 
rating on release day is much more dramatic than previously depicted. However, 
there is a somewhat unexpected rebound after the initial drop. I suspect that 
this is a result of people joking about all the bugs in the game after the initial 
wave of rage had ended. 
""")
st.markdown("")
st.markdown("""
Maybe more to come...
""")

st.header("Code for the Model")
st.markdown("""
For sentiment analysis, I used Hugging Face's BERT for sequence classification.
Essentially, this means that instead of training a model from scratch I just fine 
tuned it on my more constrained dataset. In the expanders below you can take a 
look at the code.
""")
with st.beta_expander('Model'):
    st.code("""
    from transformers import BertForSequenceClassification
    import torch.nn as nn

    class BERT(nn.Module):
        # implementation is made extremely easy by huggingface
        def __init__(self):
            super(BERT, self).__init__()

            options_name = "bert-base-uncased"
            self.encoder = BertForSequenceClassification.from_pretrained(options_name)

        def forward(self, text, label, training=True):
            if training:
                loss, prediction = self.encoder(text, labels=label)[:2]
                return loss, prediction
            else:
                prediction = self.encoder(text, labels=label)[:2]
                return prediction
    """)


with st.beta_expander('Trainer'):
    st.code("""
    import pkbar
    import torch

    def model_loss(model, sentiment, review):
        sentiment = sentiment.type(torch.cuda.LongTensor)           
        review = review.type(torch.cuda.LongTensor)  
        
        # loss is computed by BertForSequenceClassification
        loss, _ = model(review, sentiment)      # discarding pred here

        return loss


    def save_checkpoint(model, optimizer, epoch, valid_loss, train_loss, final=False):
        beep = "chkpt"
        if final:
            beep = "final"

        torch.save(
            {'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'epoch' : epoch,
            'valid_loss' : valid_loss}, 
            './checkpoints/model_large_{}.pt'.format(beep))

        print("\\nSaved checkpoint... Validation loss reported: {}".format(valid_loss))


    def train(model, optimizer, t_loader, v_loader, num_epochs=5, best_loss = float("Inf")):
        valid_loss = 0.0

        model.train()
        for epoch in range(num_epochs):
            kbar = pkbar.Kbar(target=len(t_loader), epoch=epoch, 
                            num_epochs=num_epochs, width=10, always_stateful=False)

            for p_t, ((review, sentiment), _) in enumerate(t_loader):
                optimizer.zero_grad()       # zero grads 
                loss = model_loss(model, sentiment, review)     # calc loss
                loss.backward()         # update grads
                optimizer.step()        # update parameters

                # update progress bar
                train_loss = loss.item()
                kbar.update(p_t, values=[("training loss", train_loss)])

                # validation 
                if p_t != 0 and p_t % 17085 == 0:        # => 5 times per epoch
                    print("\\n\\n Performing validation cycle.\\n")

                    model.eval()    # pause training

                    with torch.no_grad():
                        for p_v, ((review, sentiment), _) in enumerate(v_loader):
                            loss = model_loss(model, sentiment, review)
                            valid_loss += loss.item()
                    
                    avg_valid_loss = valid_loss/len(v_loader)

                    if best_loss > avg_valid_loss:
                        best_loss = avg_valid_loss

                        # save checkpoint
                        save_checkpoint(model, optimizer, epoch, avg_valid_loss, train_loss)

                    # reset validation loss
                    valid_loss = 0.0

        save_checkpoint(model, optimizer, 0, 0, 0, True)
    """)
