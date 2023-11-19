from scipy import stats
import math 
import matplotlib.pyplot as plt
from mplsoccer import PyPizza
from highlight_text import fig_text

def pizza_plot(data, position, metrics, player_name):
        df = data
        median_minutes = df["MIN"].median()
        df = df.loc[(df["POS"] == position) & (df["MIN"] >= median_minutes)]

        params = metrics
        player = df.loc[df["PLAYER"] == player_name].reset_index(drop=True)
        squad = player["SQUAD"][0]
        player = player[params] 
        player = list(player.loc[0])

        values = []
        for i in range(len(params)):
                values.append(math.floor(stats.percentileofscore(df[params[i]], player[i], nan_policy="omit")))

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
            0.515, 0.975, f"{player_name} - {squad} 23/24", size=16,
            ha="center", color="#000000", fontdict=font_title
        )

        # add subtitle
        fig.text(
            0.515, 0.933,
            f"Percentile Rank p90 vs Top 5 League {position}",
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




