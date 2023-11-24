# Libraries
#region  ----------------------------------------- #
import streamlit as st
from datetime import datetime
import time 
import sys

sys.path.insert(0, "functions/")
from fbref import joint_df
from sentence_transformer_model import sentence_transformer
from pizza_plot import pizza_plot
#endregion ---------------------------------------- #

# Page Configuration
#region  ----------------------------------------- #
page_config = st.set_page_config(page_title="Player Stats", page_icon=":soccer:", layout="centered", initial_sidebar_state="auto",
                                 menu_items={
                                        "Get Help": "mailto:remiawosanya8@gmail.com",
                                        "Report a bug": "mailto:remiawosanya8@gmail.com"
                                            })
#endregion ---------------------------------------- #

df = joint_df()
list_of_players = list(df["PLAYER"])
data_source = "https://fbref.com/"
med_link = "https://medium.com/@remiawosanya8/the-average-percentile-score-a-measure-of-overall-performance-5e983d1a1b86"

# Sidebar
#region  ----------------------------------------- #
st.sidebar.warning("Selected player must be playing in a big 5 European league and have played more than the median number of minutes.")
position_select = st.sidebar.radio("Select a position:", ["DF", "MF", "FW"], index=2)
options = df.columns[6:]
multi = st.sidebar.multiselect("Select parameters:", options=options, default=["GOALS", "ASSISTS", "XG", "PRG CARRIES", "PRG PASSES", "PASS %", "TOUCHES"])
text_input = st.sidebar.text_input("Player:", placeholder="Enter Player Name", value="Kylian Mbappe")
input = sentence_transformer(list_of_players, text_input)

# Display last update time
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
last_update_time = st.sidebar.write("Last Data Update: " + dt_string)
#endregion ---------------------------------------- #

# Tabs
#region  ----------------------------------------- #
tab1, tab2 = st.tabs(["Percentile Rank", "Average Percentile Score"])

with tab1:
    # Title
    st.title("Pizza Plot")

    # Display source link
    st.write(f"Data from: [FBref]({data_source})")
    
    # Check player and position validity
    median_minutes = df["MIN"].median()
    player_df = df.loc[(df["PLAYER"] == input) & (df["MIN"] >= median_minutes)].reset_index(drop=True)
    position_comparison = list(player_df["POS"])
    if (input not in list_of_players) or (position_select not in position_comparison):
        st.error("Check position and/or spelling is correct")
    else:
        # Display pizza plot
        with st.spinner(text="Refreshing Data"):
            time.sleep(1)
            #st.success("Done!")

        pizza = st.pyplot(pizza_plot(data=df, position=position_select, metrics=multi, player_name=input))
        pizza = st.set_option("deprecation.showPyplotGlobalUse", False)


with tab2:
    # Title
    st.title("Similar Players (APS)")

    # Display APS explanation
    st.success(f"Similar players are selected based on the Average Percentile Score (APS) of selected parameters. "
               f"The APS differential is the number of percentage points +/- the selected player's APS")
    
    # Similar players table
    dff = df.loc[(df["POS"] == position_select) & (df["MIN"] > 700)]
    params = multi
    dff = ((dff[params].rank(axis=0, numeric_only=True, pct=True)) * 100).round(decimals=1)
    dff["APS %"] = dff.mean(numeric_only=True, axis=1).round(decimals=1)
    df["APS %"] = dff["APS %"]
    player_aps = df.loc[df["PLAYER"] == input]
    aps = float(player_aps["APS %"].iloc[0])

    # Adjust the APS bounds based on user input
    aps_slider = st.slider("Similarity (APS % Differential)", min_value=1, max_value=10, value=3)
    aps_upper_bound = aps + aps_slider
    aps_lower_bound = aps - aps_slider

    # Filter similar players based on adjusted APS bounds
    similar = df.loc[(df["APS %"] >= aps_lower_bound) & (df["APS %"] <= aps_upper_bound)].sort_values("APS %", ascending=False)
    similar = similar[["PLAYER", "POS", "SQUAD", "COMP", "MIN", "APS %"] + params].sort_values("APS %", ascending=False)

    # Display the filtered DataFrame
    similar = similar.reset_index(drop=True)
    st.dataframe(similar)

    medium = st.write(f'For more information on the APS read here:  \n{med_link}')
#endregion ---------------------------------------- #
