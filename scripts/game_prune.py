import pandas as pd
import numpy as np
import datetime

def format_date(row):
    try:
        month = datetime.datetime.strptime(row["Date"][:3], "%b").month
        day = row["Date"].split(' ')[1]
        day = int(day.split(',')[0])
        year = int(row["Date"].split(' ')[2])        
        row["Date"] = "{}, {}, {}".format(year, month, day)
    except:
        print("HERE")    # dump these rows
        print(row)


def dump_df(csv_path, publisher_name, ticker):
    df = pd.read_csv(csv_path)
    df.replace('nan', np.nan, regex=True)
    df.Title = df.Title.ffill()
    df.Date = df.Date.ffill()
    df = df.groupby('Title').agg({'Date': 'first', 'Platforms': lambda x: ', '.join(x), 'Developers': 'first'}).reset_index() 
    df.apply(format_date, axis=1)
    df = df[df.Date.str[:4]=="2020"]
    df = df[df["Platforms"].str.contains('PlayStation|Xbox|Nintendo')].reset_index().drop(['index'], axis=1)
    df.insert(loc=0, column="Ticker", value=ticker)
    df.insert(loc=0, column="Publisher", value=publisher_name)
    return df


def main():
    # activision
    atvi = pd.read_csv("./raw_data/games/ATVI.csv")
    atvi.insert(loc=0, column="Ticker", value="ATVI")
    atvi.insert(loc=0, column="Publisher", value="Activision")

    # ubisoft
    ubi = dump_df("./raw_data/games/UBI.csv", "Ubisoft", "UBSFY")

    # cd projekt red
    cdpr = pd.read_csv("./raw_data/games/CDPR.csv")
    cdpr.insert(loc=0, column="Ticker", value="OTGLF")
    cdpr.insert(loc=0, column="Publisher", value="CD Projekt")

    # take-two 2k
    tti2k = dump_df("./raw_data/games/2K.csv", "2K Games", "TTWO")

    #take-two rockstar
    ttirs = dump_df("./raw_data/games/RSTAR.csv", "Rockstar Games", "TTWO")

    # electronic arts
    ea = dump_df("./raw_data/games/EA.csv", "Electronic Arts", "EA")

    # nintendo
    nint = dump_df("./raw_data/games/NINT.csv", "Nintendo", "NTDOY")

    game_list = [atvi, ubi, cdpr, tti2k, ttirs, ea, nint]
    games = pd.concat(game_list)
    games.to_csv("games2.csv")


if __name__ == "__main__":
    main()