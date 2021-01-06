import pandas as pd
import sys
import re
from sklearn.model_selection import train_test_split

def truncate(row):
    trunc = row['review'][:512]
    return trunc

if __name__ == "__main__":
    df = pd.read_csv('./steamset.csv')
    df = df[['review', 'sentiment']]

    # chop and clean
    df['review'] = df.apply(truncate, axis=1)
    df.review.str.replace('[^a-zA-Z0-9]', ' ')

    # train, test, validate
    train, test = train_test_split(df, test_size=0.15)
    train, validate = train_test_split(train, test_size=0.12)   # ~10% of initial ds

    # output to csvs
    train = train.reset_index()
    test = test.reset_index()
    validate = validate.reset_index()

    train = train[['review', 'sentiment']]
    test = test[['review', 'sentiment']]
    validate = validate[['review', 'sentiment']]

    train.to_csv('./train.csv', index=False)
    test.to_csv('./test.csv', index=False)
    validate.to_csv('./validate.csv', index=False)











        