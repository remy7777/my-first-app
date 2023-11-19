# Libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from unidecode import unidecode
import streamlit as st

main_page = "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"

# Functions
def get_table_url(url):
    data = requests.get(url)
    soup = BeautifulSoup(data.text, "lxml")
    section = soup.select("div.section_wrapper")[0]
    links = [f"https://fbref.com{l.get('href')}" for l in section.find_all("a") if "/players/" in l.get('href')]
    return links

def get_player_stats(url_index, match_phrase, rename_dict):
    urls = get_table_url(main_page)
    data = requests.get(urls[url_index])
    df = pd.read_html(data.text, match=match_phrase)[0]
    df.columns = df.columns.droplevel()
    df.reset_index(drop=True, inplace=True)
    df.columns = [x.lower() for x in df.columns]
    
    # drop table divider rows
    df = df.loc[df["rk"] != "Rk"]
    df = df.drop("pos", axis=1).join(df["pos"].str.split(",", expand=True).stack().reset_index(drop=True, level=1).rename("pos"))
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))

    cols = pd.Series(df.columns)
    duplicate_columns = cols[cols.duplicated()].unique()

    for dup in duplicate_columns: 
        cols[cols[cols == dup].index.values.tolist()] = [f"{dup}.{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols

    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
    
    time.sleep(1)
    return df

player_summary_stats = get_player_stats(0, "Player Standard Stats", {
    "gls.1": "gls/90",
    "ast.1": "ast/90",
    "g+a.1": "g+a/90",
    "g-pk.1": "g-pk/90",
    "g+a-pk": "g+a-pk/90",
    "xg.1": "xg/90",
    "xag.1": "xag/90",
    "xg+xag": "xg+xag/90",
    "npxg.1": "npxg/90",
    "npxg+xag.1": "npxg+xag/90"
    })

player_shooting_stats = get_player_stats(3, "Player Shooting", None)

player_passing_stats = get_player_stats(4, "Player Passing", {
    "cmp": "total_cmp",
    "att": "total_att",
    "cmp%": "total_cmp_pct",
    "cmp.1": "short_cmp",
    "att.1": "short_att",
    "cmp%.1": "short_cmp_pct",
    "cmp.2": "med_cmp",
    "att.2": "med_att",
    "cmp%.2": "med_cmp_pct",
    "cmp.3": "long_cmp",
    "att.3": "long_att",
    "cmp%.3": "long_cmp_pct",
})

player_gca_stats = get_player_stats(6, "Player Goal and Shot Creation", {
    "passlive": "sca_passlive",
    "passdead": "sca_passdead",
    "to": "sca_takeons",
    "sh": "sca_shots",
    "fld": "sca_fld",
    "def": "sca_def",
    "passlive.1": "gca_passlive",
    "passdead.1": "gca_passdead",
    "to.1": "gca_takeons",
    "sh.1": "gca_shots",
    "fld.1": "gca_fld",
    "def.1": "gca_def"
})

player_def_stats = get_player_stats(7, "Player Defensive Actions", {
    "tkl": "total_tkl",
    "tkl.1": "drib_tkld",
    "att": "tkl_att",
    "past": "drib_past",
    "sh": "shots_blocked",
    "pass": "pass_blocked"
})

player_poss_stats = get_player_stats(8, "Player Possession", None)

summ_cols = ["player", "pos", "squad", "comp", "min", "90s", "gls", "ast", "g+a", "xg", "npxg", "xag", "prgc", "prgp", "prgr"]
shoot_cols = ["sh", "sot","g/sh", "dist", "npxg/sh", "g-xg", "np:g-xg"]
passing_cols = ["total_att", "total_cmp_pct", "prgdist", "short_att", "short_cmp_pct", "long_att", "long_cmp_pct",
                "xa", "a-xag", "kp", "1/3", "ppa"]
gca_cols = ["sca", "sca_takeons", "gca", "gca_takeons"]
def_cols = ["total_tkl", "tklw", "def 3rd", "mid 3rd", "att 3rd", "tkl_att", "tkl%", "blocks", "int", "clr", "err"]
poss_cols = ["touches", "def 3rd", "mid 3rd", "att 3rd", "att pen", "att", "succ%", "tkld%", "1/3", "cpa", "mis", "dis"]

@st.cache_data()
def joint_df():
    summary = player_summary_stats[summ_cols].reset_index(drop=True)
    shooting = player_shooting_stats[shoot_cols].reset_index(drop=True)
    passing = player_passing_stats[passing_cols].reset_index(drop=True)
    gca = player_gca_stats[gca_cols].reset_index(drop=True)
    defense = player_def_stats[def_cols].reset_index(drop=True)
    possession = player_poss_stats[poss_cols].reset_index(drop=True)

    df = pd.concat([
        summary,
        shooting, 
        passing,
        gca,
        defense,
        possession
        ], axis=1).reset_index(drop=True)
    
    df["player"] = df["player"].astype("str").apply(unidecode)
    df["comp"] = df["comp"].replace({"eng": "", "fr": "", "it": "", "de": "", "es": ""}, regex=True)

    cols = pd.Series(df.columns)

    for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols

    df.rename(columns={
        "gls":"goals", "ast":"assists", "prgc":"prg carries", "prgp":"prg passes",
        "sh":"shots", "sot":"shots on target", "g/sh":"goals/shot",
        "dist":"shot distance", "total att":"pass attempts", "total_cmp_pct":"pass %",
        "prgdist":"prgpass dist", "short_att":"short passes", "short_cmp_pct":"short pass%", 
        "long_att":"long pass att", "long_cmp_pct":"long pass%", "kp":"key passes", 
        "1/3":"final 1/3 pass", "ppa":"penalty area pass", "sca":"shot creating act", "gca":"goal creating act",
        "total_tkl":"tackles", "tklw":"tackles won", "def 3rd":"tkl def 3rd", "mid 3rd":"tkl mid 3rd",
        "att 3rd":"tkl att 3rd", "tkl_att":"tackle att", "def 3rd.1":"def 3rd touches", "mid 3rd.1":"mid 3rd touches",
        "att 3rd.1":"att 3rd touches", "att pen":"att pen touches", "att":"take-on att", "succ%":"takeon%",
        "1/3.1":"final 1/3 carries", "cpa":"penalty area carries", "mis":"miscontrol", "dis":"disposs"}, inplace=True)
    
    df[df.columns[6:]] = df[df.columns[6:]].div(df["90s"], axis=0).round(decimals=1)
    df.columns = [x.upper() for x in df.columns]
    
    return df



