import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import numpy as np
from scipy import stats
import math 
import matplotlib.pyplot as plt
from mplsoccer import PyPizza, Bumpy
from highlight_text import fig_text
import streamlit as st
from unidecode import unidecode 
from datetime import datetime
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

page_config = st.set_page_config(page_title="Player Overview", page_icon=":soccer:", layout="centered", initial_sidebar_state="auto",
                                 menu_items={
                                        "Get Help": "mailto:remiawosanya8@gmail.com",
                                        "Report a bug": "mailto:remiawosanya8@gmail.com"
                                            })

tab1, tab2 = st.tabs(["Percentile Rank", "Similar Players"])

def get_url():
    url = "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"
    data = requests.get(url)
    soup = BeautifulSoup(data.text, "lxml")

    section = soup.select("div.section_wrapper")[0]

    links = section.find_all("a") 
    links = [l.get("href") for l in links]
    links = [l for l in links if "/players/" in l]
    urls = [f"https://fbref.com{l}" for l in links]
    # print(urls) 
    return [url, urls]

def player_summary_stats():
    urls = get_url()[1]
    data = requests.get(urls[0])
    df = pd.read_html(data.text, match = "Player Standard Stats")[0]
    df.columns = df.columns.droplevel()
    df.reset_index(drop=True, inplace=True)
    df.columns = [x.lower() for x in df.columns]

    # drop table divider rows
    df = df.loc[df["rk"] != "Rk"]
    df = df.drop("pos", axis=1).join(df["pos"].str.split(",", expand=True).stack().reset_index(drop=True, level=1).rename("pos"))
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    # print(df.dtypes)

    cols = pd.Series(df.columns)
    # print(cols[cols.duplicated()].unique())

    for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    # print(df.columns)

    df.rename(columns = {"gls.1":"gls/90", "ast.1":"ast/90", "g+a.1":"g+a/90", "g-pk.1":"g-pk/90",
                        "g+a-pk":"g+a-pk/90", "xg.1":"xg/90", "xag.1":"xag/90",
                        "xg+xag":"xg+xag/90", "npxg.1":"npxg/90", "npxg+xag.1":"npxg+xag/90"}, inplace=True)
    # print(df.head())
    
    time.sleep(1)
    return df

def player_shooting_stats():
    urls = get_url()[1]
    data = requests.get(urls[3])
    df = pd.read_html(data.text, match = "Player Shooting")[0]
    df.columns = df.columns.droplevel()
    df.reset_index(drop=True, inplace=True)
    df.columns = [x.lower() for x in df.columns]

    # drop table divider rows
    df = df.loc[df["rk"] != "Rk"]
    df = df.drop("pos", axis=1).join(df["pos"].str.split(",", expand=True).stack().reset_index(drop=True, level=1).rename("pos"))
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: pd.to_numeric(x,errors="ignore"))
    # print(df.dtypes)

    cols = pd.Series(df.columns)
    # print(cols[cols.duplicated()].unique())

    time.sleep(1)
    return df

def player_passing_stats():
    urls = get_url()[1]
    data = requests.get(urls[4])
    df = pd.read_html(data.text, match = "Player Passing")[0]
    df.columns = df.columns.droplevel()
    df.reset_index(drop=True, inplace=True)
    df.columns = [x.lower() for x in df.columns]

    # drop table divider rows
    df = df.loc[df["rk"] != "Rk"]
    df = df.drop("pos", axis=1).join(df["pos"].str.split(",", expand=True).stack().reset_index(drop=True, level=1).rename("pos"))
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    # print(df.dtypes)

    cols = pd.Series(df.columns)
    # print(cols[cols.duplicated()].unique())

    for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    # print(df.columns)

    df.rename(columns = {"cmp":"total_cmp", "att":"total_att", "cmp%":"total_cmp_pct",
                                        "cmp.1":"short_cmp", "att.1":"short_att", "cmp%.1":"short_cmp_pct",
                                        "cmp.2":"med_cmp", "att.2":"med_att", "cmp%.2":"med_cmp_pct",
                                        "cmp.3":"long_cmp", "att.3":"long_att", "cmp%.3":"long_cmp_pct"}, inplace=True)
    # print(df.head())
    
    time.sleep(1)
    return df

def player_gca_stats():
    urls = get_url()[1]
    data = requests.get(urls[6])
    df = pd.read_html(data.text, match = "Player Goal and Shot Creation")[0]
    df.columns = df.columns.droplevel()
    df.reset_index(drop=True, inplace=True)
    df.columns = [x.lower() for x in df.columns]

    # drop table divider rows
    df = df.loc[df["rk"] != "Rk"]
    df = df.drop("pos", axis=1).join(df["pos"].str.split(",", expand=True).stack().reset_index(drop=True, level=1).rename("pos"))
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    # print(df.dtypes)

    cols = pd.Series(df.columns)
    # print(cols[cols.duplicated()].unique())

    for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    # print(df.columns)

    df.rename(columns = {"passlive":"sca_passlive", "passdead":"sca_passdead", "to":"sca_takeons",
                         "sh":"sca_shots", "fld":"sca_fld", "def":"sca_def",
                         "passlive.1":"gca_passlive", "passdead.1":"gca_passdead", "to.1":"gca_takeons",
                         "sh.1":"gca_shots", "fld.1":"gca_fld", "def.1":"gca_def"}, inplace=True)
    # print(df.head())
    
    time.sleep(1)
    return df

def player_def_stats():
    urls = get_url()[1]
    data = requests.get(urls[7])
    df = pd.read_html(data.text, match = "Player Defensive Actions")[0]
    df.columns = df.columns.droplevel()
    df.reset_index(drop=True, inplace=True)
    df.columns = [x.lower() for x in df.columns]

    # drop table divider rows
    df = df.loc[df["rk"] != "Rk"]
    df = df.drop("pos", axis=1).join(df["pos"].str.split(",", expand=True).stack().reset_index(drop=True, level=1).rename("pos"))
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    # print(df.dtypes)

    cols = pd.Series(df.columns)
    # print(cols[cols.duplicated()].unique())

    for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    # print(df.columns)

    df.rename(columns = {"tkl":"total_tkl", "tkl.1":"drib_tkld", "att":"tkl_att",
                         "past":"drib_past", "sh":"shots_blocked", "pass":"pass_blocked"}, inplace=True)
    # print(df.head())
    
    time.sleep(1)
    return df

def player_poss_stats():
    urls = get_url()[1]
    data = requests.get(urls[8])
    df = pd.read_html(data.text, match = "Player Possession")[0]
    df.columns = df.columns.droplevel()
    df.reset_index(drop=True, inplace=True)
    df.columns = [x.lower() for x in df.columns]

    # drop table divider rows
    df = df.loc[df["rk"] != "Rk"]
    df = df.drop("pos", axis=1).join(df["pos"].str.split(",", expand=True).stack().reset_index(drop=True, level=1).rename("pos"))
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    # print(df.dtypes)

    cols = pd.Series(df.columns)
    # print(cols[cols.duplicated()].unique())
    
    time.sleep(1)
    return df

def player_misc_stats():
    urls = get_url()[1]
    data = requests.get(urls[10])
    df = pd.read_html(data.text, match = "Player Miscellaneous Stats")[0]
    df.columns = df.columns.droplevel()
    df.reset_index(drop=True, inplace=True)
    df.columns = [x.lower() for x in df.columns]

    # drop table divider rows
    df = df.loc[df["rk"] != "Rk"]
    df = df.drop("pos", axis=1).join(df["pos"].str.split(",", expand=True).stack().reset_index(drop=True, level=1).rename("pos"))
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    # print(df.dtypes)

    cols = pd.Series(df.columns)

    df.rename(columns = {"won":"aerial_duels_won", "lost":"aerial_duels_lost", "won%":"aerial_duels_pct"}, inplace=True)
    # print(df.head())
    
    time.sleep(1)
    return df

summ = ["player", "pos", "squad", "comp", "min", "90s", "gls", "ast", "g+a", "xg", "npxg", "xag", "prgc", "prgp", "prgr"]
shoot = ["sh", "sot", "sh/90", "g/sh", "g/sot", "dist", "npxg/sh", "g-xg", "np:g-xg"]
passing = ["total_att", "total_cmp_pct", "totdist", "prgdist", "short_att", "short_cmp_pct", "long_att", "long_cmp_pct",
           "xa", "a-xag", "kp", "1/3", "ppa"]
gca = ["sca", "sca90", "sca_takeons", "sca_fld", "sca_def", "gca", "gca90", "gca_takeons"]
defense = ["total_tkl", "tklw", "def 3rd", "mid 3rd", "att 3rd", "tkl_att", "tkl%", "lost", "blocks", "shots_blocked",
           "pass_blocked", "tkl+int", "clr", "err"]
possession = ["touches", "def 3rd", "mid 3rd", "att 3rd", "att pen", "att", "succ%", "tkld%", "carries",
              "1/3", "cpa", "mis", "dis"]

# master dataframe
@st.cache_data
def dataframes():
    summ_df = player_summary_stats()
    shoot_df = player_shooting_stats()
    png_df = player_passing_stats()
    gca_df = player_gca_stats()
    def_df = player_def_stats()
    poss_df = player_poss_stats()

    summary = summ_df[summ]
    shooting = shoot_df[shoot]
    pas = png_df[passing]
    gca_ = gca_df[gca]
    def_ = def_df[defense]
    poss = poss_df[possession]

    df = pd.concat([summary, shooting, pas, gca_, def_, poss], axis=1)
    df["player"] = df["player"].astype("str")
    df["player"] = df["player"].apply(unidecode)
    df["comp"] = df["comp"].str.replace("eng", "")
    df["comp"] = df["comp"].str.replace("fr", "")
    df["comp"] = df["comp"].str.replace("it", "")
    df["comp"] = df["comp"].str.replace("de", "")
    df["comp"] = df["comp"].str.replace("es", "")

    cols = pd.Series(df.columns)
    # print(cols[cols.duplicated()].unique())

    for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    # print(df.columns)
    df.rename(columns = {"gls":"goals", "ast":"assists", "prgc":"prg carries", "prgp":"prg passes",
                        "prgr":"prg receptions", "sh":"shots", "sot":"shots on target", "sh/90":"shots/90",
                        "g/sh":"goals/shots", "dist":"shot dist", "total_att":"pass att", "total_cmp_pct":"pass completion%",
                        "totdist":"pass dist", "prgdist":"prg pass dist", "short_att":"short pass att",
                        "short_cmp_pct":"short pass%", "long_att":"long pass att", "long_cmp_pct":"long pass%",
                        "kp":"key passes", "1/3":"final 1/3 pass", "ppa":"pass penalty area", "sca":"shot creating act",
                        "sca90":"shot creating act/90", "gca":"goal creating act", "gca/90":"goal creating act/90",
                        "total_tkl":"total tkl", "tklw":"tkl won", "def 3rd":"tkl def 3rd", "mid 3rd":"tkl mid 3rd",
                        "att 3rd":"tkl att 3rd", "tkl_att":"tkl att", "lost":"tkl lost", "shots_blocked":"shots blk",
                        "pass_blocked":"pass blk", "def 3rd.1":"touch def 3rd", "mid 3rd.1":"touch mid 3rd",
                        "att 3rd.1":"touch att 3rd", "att pen":"touch att pen", "att":"takeon att", "succ%":"takeon%",
                        "1/3.1":"carries final 1/3", "cpa":"carries penalty area", "mis":"miscontrol", "dis":"disposs"}, inplace=True)

    return df

df = dataframes()

# sentence transformer model
@st.cache_data
def sentence_transformer(p):
        # Define the model we want to use (it'll download itself)
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        sentences = list(df["player"])

        # vector embeddings created from dataset
        embeddings = model.encode(sentences)

        # query vector embedding
        query_embedding = model.encode(p)

        # define our distance metric
        def cosine_similarity(a, b):
            return np.dot(a, b)/(norm(a)*norm(b))

        # run semantic similarity search
        for e, s in zip(embeddings, sentences):
            if cosine_similarity(e, query_embedding) > 0.8:
                result = s
                # print(s, " -> similarity score = ", cosine_similarity(e, query_embedding))

        return result

with tab1:
    # title
    title = st.title("Pizza Plot", anchor=None)

    # sidebar
    note = st.sidebar.warning("The selected player must be playing in one of the big 5 European leagues and have played more than 700 minutes.")

    position = ["DF", "MF", "FW"]
    position_select = st.sidebar.radio(label="Select a position:", options=position, index=2)

    options = df.columns[6:]
    default = ["goals", "assists", "g+a", "xg", "npxg", "xag", "prg carries", "prg passes", "prg receptions"]
    multi = st.sidebar.multiselect(label="Select a set of parameters:", options=options, default=default)

    text_input = st.sidebar.text_input(label="Player:", help="Include both first and last name",
                                    placeholder="Enter Player Name", value="Kylian Mbappe")
    input = sentence_transformer(text_input)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    date = st.sidebar.write(f"**Latest data as of {dt_string}**")

    # pizza plot
    def basic_pizza():
        df = dataframes()
        df = df.loc[(df["pos"] == position_select) & (df["min"] > 700)]

        params = multi
        player = df.loc[df["player"] == input].reset_index(drop=True)
        squad = player["squad"][0]
        player = player[params] 
        player = list(player.loc[0])

        values = []
        for i in range(len(params)):
                values.append(math.floor(stats.percentileofscore(df[params[i]], player[i], nan_policy="omit")))
        # print(values)

        slice_colors = ["#CDC0B0"] * len(params)
        text_colors = ["#000000"] * len(params)
        font_title = {"family": "serif",
                    "color":  "black",
                    "weight": "bold",
                    "size": 16}
        font = {"family": "serif",
                "color":  "black",
                "weight": "normal"}

        # instantiate PyPizza class
        baker = PyPizza(
            params=params,                  # list of parameters
            background_color="#F5F5DC",     # background color
            straight_line_color="#F5F5DC",  # color for straight lines
            straight_line_lw=3,             # linewidth for straight lines
            last_circle_lw=3,               # linewidth of last circle
            other_circle_lw=1,              # linewidth for other circles
            inner_circle_size=10            # size of inner circle
            )

        # plot pizza
        fig, ax = baker.make_pizza(
            values,                          # list of values
            figsize=(7, 7),                  # adjust figsize according to your need
            color_blank_space="same",        # use same color to fill blank space
            slice_colors=slice_colors,       # color for individual slices
            value_colors=text_colors,        # color for the value-text
            value_bck_colors=slice_colors,   # color for the blank spaces
            blank_alpha=0.4,                 # alpha for blank-space colors
            kwargs_slices=dict(
                edgecolor="#F5F5DC", zorder=1, linewidth=1
            ),                            
            kwargs_params=dict(
                color="black", fontsize=11, va="center", fontdict=font
            ),                             
            kwargs_values=dict(
                color="#2C5C1E", fontsize=7, zorder=3, fontdict=font,
                bbox=dict(
                    edgecolor="#2C5C1E", facecolor="none",
                    boxstyle="circle,pad=0.4", lw=1
                )
            )                               
        )
        
        # add title
        fig.text(
            0.515, 0.975, f"{input} - {squad} 22/23", size=16,
            ha="center", color="#000000", fontdict=font_title
        )

        # add subtitle
        fig.text(
            0.515, 0.933,
            f"Percentile Rank vs Top 5 League {position_select}",
            size=13,
            ha="center", color="#000000", fontdict=font
        )
        
        # add credits
        CREDIT_1 = "Data: FBref.com"
        CREDIT_2 = "Twitter: @statsandstuff7"

        fig.text(
            0.99, 0.02, f"{CREDIT_1}\n{CREDIT_2}", size=7,
            fontdict=font, color="#000000",
            ha="right"
        )
        
        plt.show()

    players = df["player"]
    player = df.loc[(df["player"] == input) & (df["min"] > 700)].reset_index(drop=True)
    p = player["pos"]

    if (input not in list(players)) or (position_select not in list(p)):
        st.error("Check position and/or spelling is correct")
    else:
        with st.spinner(text="Refreshing Data"):
            time.sleep(1)
            st.success("Done!")

        # st.balloons()
        # st.snow()
        pizza = st.pyplot(basic_pizza())
        pizza = st.set_option("deprecation.showPyplotGlobalUse", False)

with tab2:
    # dataframe
    def show_df():
        df = dataframes()
        df = df.loc[(df["pos"] == position_select) & (df["min"] > 700)]

        params = multi
        df1 = df[params]
        df1 = ((df1.rank(axis=0, numeric_only=True, pct=True))*100).round(decimals=1)
        df1["aps%"] = df1.mean(numeric_only=True, axis=1).round(decimals=1)
        df["aps%"] = df1["aps%"]

        p = df.loc[df["player"] == input]
        aps = float(p["aps%"])
        lower = aps - 3
        upper = aps + 3
        similar = df.loc[(df["aps%"] > lower) & (df["aps%"] < upper)].sort_values("aps%", ascending=False)
        new = similar[["player", "pos", "squad", "comp", "min", "aps%"]]
        df = df[params]
        new[params] = df

        return new
    
    title = st.title("Similar Players", anchor=None)
    med_link = "https://medium.com/@remiawosanya8/the-average-percentile-score-a-measure-of-overall-performance-5e983d1a1b86"
    aps_text = st.success(f'Similar players are selected on the basis of the Average Percentile Score (aps%) of selected parameters. For more information read here:  \n{med_link}')
    table = show_df()
    # select = st.selectbox(label="Sort by:", options=multi, index=0)
    filter = st.number_input("Top N:", max_value=len(table), value=len(table), key=1)
    table = table.sort_values("aps%", ascending=False).reset_index(drop=True, inplace=False).head(filter)
    show_table = st.write(table)
    url = "https://fbref.com/"
    link = st.write(f"Data can be found at: {url}")

