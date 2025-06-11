import os
import json
import zipfile
from datetime import datetime

import requests
import streamlit as st
import pandas as pd
import altair as alt

from st_aggrid import AgGrid, GridOptionsBuilder


## Constants #########################################################################
POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'OF', 'SP', 'RP']
ACCENT = "teal"
## End Constants #####################################################################


@st.cache_data
def load_data(today: str) -> None:
    """
    Function to be run at the initialization of the dashboard.

    Downloads all of the relevant CSV data files from GitHub, where they are stored as build
    artifact for a build that runs daily and scrapes the relevant statistics.

    Expects GitHub PAT with proper permissions to be available as an environment variable under
    'GITHUB_PAT'.

    :param str date: The date we want the data for, in YYYY-mm-dd format, which will usually be 
                     today. Added as a parameter for caching purposes.

    :raises ValueError: Raises a ValueError if an artifact with today's timestamp is not found.
    """

    url = "https://api.github.com/repos/hockey-stats/chart-plotting/actions/artifacts"
    payload = {}
    headers = {
        'Authorization': f'Bearer {os.environ["GITHUB_PAT"]}'
    }
    output_filename = 'data.zip'

    # Returns a list of every available artifact for the repo
    response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
    response_body = json.loads(response.text)

    for artifact in response_body['artifacts']:
        if artifact['name'] == 'dashboard-fa-data':
            artifact_creation_date = artifact['created_at'].split('T')[0]
            if today == artifact_creation_date:
                download_url = artifact['archive_download_url']
                break
                # Breaks when we find an artifact with the correct name and today's date
    else:
        # Raise an error if no such artifact as found
        raise ValueError(f"Data for {today} not found, exiting....")

    # Downloads the artifact as a zip file...
    dl_response = requests.request("GET", download_url, headers=headers, data=payload, timeout=20)
    with open(output_filename, 'wb') as fo:
        fo.write(dl_response.content)

    # ... and unzips
    with zipfile.ZipFile(output_filename, 'r') as zip_ref:
        zip_ref.extractall('data')

    # The unzipped contents will be one DataFrame for each position
    for pos in POSITIONS:
        # Read the raw CSV into a DataFrame
        df = pd.read_csv(f'data/fantasy_data_{pos}.csv')

        # Determine the number of players on team, for rows to freeze
        t_df = df[df['on_team']]
        num_freeze = len(set(t_df['Name']))
        st.session_state[f"{pos}_rows_freeze"] = num_freeze

    print(f"Data loaded for {artifact_creation_date}")


########################################################################################
## Main Script #########################################################################
########################################################################################

# Get data for today's date.
load_data(datetime.today().strftime('%Y-%m-%d'))

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 70px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

chosen_position = st.sidebar.selectbox(
    label="Position:",
    options=POSITIONS
)

chosen_term = st.sidebar.radio(
    label="Chosen term:",
    options=['Last Week', 'Last Month', 'Full Season'],
    index=0
)

df = pd.read_csv(f"data/fantasy_data_{chosen_position}.csv", index_col=False)

plot_df = df[df['term'] == chosen_term.split(' ')[-1].lower()]

table_df = plot_df[['Name', 'Position(s)', 'ABs', 'AVG', 'HRs', 'RBIs', 'SBs', 'wRC+', 'xwOBA']]


l_column, r_column = st.columns(2)

with l_column:
   # st.dataframe(
   #     data=table_df,
   #     hide_index=True
   # )
    grid_options = {
        'defaultColDef': {
            'resizable': True
        },
        'columnDefs': [
            {'headerName': 'Name', 'field': 'Name', 'pinned': 'left'},
        ]
    }
    AgGrid(table_df, gridOptions=grid_options)

with r_column:
    chart = (
        alt.Chart(plot_df)
        .mark_circle()
        .encode(
            x='xwOBA',
            y='wRC+',
            color='on_team',
            tooltip=['Name']
        )
    )

    event = st.altair_chart(chart)


########################################################################################
## End Main Script #####################################################################
########################################################################################


