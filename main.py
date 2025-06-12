import os
import json
import zipfile
from datetime import datetime

import requests
import streamlit as st
import pandas as pd
import altair as alt

from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, StAggridTheme


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

st.set_page_config(layout='wide')

st.markdown(
    """
    <style>
       [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 30px;
           max-width: 30px;
       }
    </style>
    """,
    unsafe_allow_html=True,
)

l_column, r_column = st.columns([0.6, 0.4])

with l_column:
    chosen_position = st.selectbox(
        label="Position:",
        options=POSITIONS,
    )

with r_column:
    chosen_term = st.radio(
        label="Chosen term:",
        options=['Last Week', 'Last Month', 'Full Season'],
        index=0,
        horizontal=True
    )

col_width = 60


df = pd.read_csv(f"data/fantasy_data_{chosen_position}.csv", index_col=False)

plot_df = df[df['term'] == chosen_term.split(' ')[-1].lower()]

table_df = plot_df[['Name', 'ABs', 'AVG', 'HRs', 'RBIs', 'SBs', 'wRC+', 'xwOBA', 'on_team']]

table_df['Name'] = table_df.apply(lambda row: \
                                  f"{row['Name'][0]}. {' '.join(row['Name'].split(' ')[1:])}",
                                  axis=1)
columnDefs = [
    {'field': col,
     'headerName': col,
     'width': col_width,
     'sortable': True}
     for col in list(table_df.columns) if col != 'on_team'
]

columnDefs[0]['width'] = 90

min_x = df['xwOBA'].min()
max_x = df['xwOBA'].max()

min_y = df['wRC+'].min()
max_y = df['wRC+'].max()

with r_column:
    chart = (
        alt.Chart(plot_df,
                padding={'left': 20, 'top': 20, 'right': 20, 'bottom': 20},
                width=300)
        .mark_circle(
            size=150
        )
        .encode(
            color=alt.Color('on_team').scale(scheme='dark2', reverse=True),
            tooltip=['Name', 'Team', 'wRC+', 'xwOBA'],
            x=alt.X('xwOBA', scale=alt.Scale(domain=[0.2, 0.45])),
            y=alt.Y('wRC+', scale=alt.Scale(domain=[50, 200])),
            text='Name'
        )
    )

    st.altair_chart(chart)

with l_column:
    cellStyle = JsCode(
        r"""
        function(cellClassParams) {
            if (cellClassParams.data.on_team) {
                return {'background-color': '#a6761d'}
            }
            return {};
        }
        """)

    css={'.ag-header-group-cell-label.ag-sticky-label': {'flex-direction': 'column', 'margin': 'auto',
                                                     'font-size': '30pt'}}

    grid_builder = GridOptionsBuilder.from_dataframe(table_df)
    grid_options = grid_builder.build()

    grid_options['defaultColDef']['cellStyle'] = cellStyle
    grid_options['defaultColDef']['autoHeight'] = True
    grid_options['columnDefs'] = columnDefs

    AgGrid(table_df, gridOptions=grid_options, allow_unsafe_jscode=True,
           custom_css=css)


########################################################################################
## End Main Script #####################################################################
########################################################################################


