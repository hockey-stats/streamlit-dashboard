import os
import json
import zipfile
from datetime import datetime, timedelta
import requests
import streamlit as st
import pandas as pd
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode


## Constants #########################################################################
BATTING_POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'OF', 'All Batters']
PITCHING_POSITIONS = ['SP', 'RP', 'All Pitchers']
ALL_POSITIONS = BATTING_POSITIONS + PITCHING_POSITIONS
ACCENT = "teal"

# Define minimum thresholds for players to meet to be displayed over a given term
THRESHOLDS = {
    "All Batters": {
        "week": 15,
        "month": 50,
        "season": 100
    },
    "SP": {
        "week": 6,
        "month": 10,
        "season": 80
    },
    "RP": {
        "week": 1,
        "month": 10,
        "season": 25
    },
    "All Pitchers": {
        "week": 1,
        "month": 10,
        "season": 25
    }
}
## End Constants #####################################################################

st.set_page_config(layout='wide')


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
            print(artifact['created_at'])
            artifact_creation_date = artifact['created_at'].split('T')[0]
            if today == artifact_creation_date:
                download_url = artifact['archive_download_url']
                break
                # Breaks when we find an artifact with the correct name and today's date
    else:
        # Raise an error if no such artifact as found
        raise ValueError(f"Data for {today} not found, exiting....")

    print(f"Found artifact at {download_url}")
    print("Downloading...")

    # Downloads the artifact as a zip file...
    dl_response = requests.request("GET", download_url, headers=headers, data=payload, timeout=20)
    with open(output_filename, 'wb') as fo:
        fo.write(dl_response.content)

    print("Download complete")

    # ... and unzips
    with zipfile.ZipFile(output_filename, 'r') as zip_ref:
        zip_ref.extractall('data')

    print(os.listdir('data'))

    print(f"Data loaded for {artifact_creation_date}")


########################################################################################
## Main Script #########################################################################
########################################################################################

# Get data for today's date.
today = datetime.today()
# If checking before 7am UTC, use yesterday's data instead, since data hasn't been updated yet
if today.hour <= 7:
    today -= timedelta(days=1)
load_data(today.strftime('%Y-%m-%d'))

# Set title
st.markdown(
    """
    # Interesting Free Agents
    """
)

# Get two columns for our page
l_column, r_column = st.columns([0.53, 0.47])

# Add the position selector to left column...
with l_column:
    chosen_position = st.selectbox(
        label="Position:",
        options=ALL_POSITIONS,
    )

# ... and term selector on the right
with r_column:
    chosen_term = st.radio(
        label="Chosen term:",
        options=['Last Week', 'Last Month', 'Full Season'],
        index=0,
        horizontal=True
    )

# Load the correct CSV for chosen position
if chosen_position in BATTING_POSITIONS:
    df = pd.read_csv("data/batter_data.csv")
    if chosen_position != 'All Batters':
        df = df[df['Position(s)'].str.contains(chosen_position)]
    # Rename HardHit% and present in full percentages
    df['HH%'] = df.apply(lambda row: row['HardHit%'] * 100, axis=1)

# For the pitchers DataFrames, scale K-BB% up to be a raw percentage
elif chosen_position in PITCHING_POSITIONS:
    df = pd.read_csv("data/pitcher_data.csv")
    if chosen_position != 'All Pitchers':
        df = df[df['Position(s)'].str.contains(chosen_position)]
    df['K-BB%'] = df.apply(lambda row: row['K-BB%'] * 100, axis=1)



########################################################################################
##  Begin Table ########################################################################
########################################################################################

# Table only wants data from the chosen term
term = chosen_term.split(' ')[-1].lower()
table_df = df[df['term'] == term]

# Apply minimum thresholds for players not on our team
if chosen_position in BATTING_POSITIONS:
    table_df = table_df[(table_df['ABs'] >= THRESHOLDS['All Batters'][term]) | (table_df['on_team'])]
else:
    table_df = table_df[(table_df['IP'] >= THRESHOLDS[chosen_position][term]) | (table_df['on_team'])]

display_number = 25 if 'All' in chosen_position else 15

table_df = table_df.sort_values(by=['on_team', 'Rank'], ascending=[False, True]).head(display_number)

# Don't want to include every single column from the DataFrame. Choose specific columns
# based on whether we're dealing with hitters or pitchers
if chosen_position in BATTING_POSITIONS:
    table_df = table_df[['Name', 'ABs', 'AVG', 'HRs', 'RBIs', 'Runs', 'SBs', 'wRC+', 'xwOBA',
                         'HH%', 'Rank', 'on_team']]
else:
    table_df = table_df[['Name', 'IP', 'ERA', 'WHIP', 'Ks', 'Ws', 'SVs', 'Stuff+', 'xERA',
                         'K-BB%', 'Rank', 'on_team']]

# Formats name, e.g. Bo Bichette -> B. Bichette
table_df['Name'] = table_df.apply(lambda row: \
                                  f"{row['Name'][0]}. {' '.join(row['Name'].split(' ')[1:])}",
                                  axis=1)

# Define certain columns which can be smaller by default
small_cols = ['ABs', 'IPs', 'HRs', 'RBIs', 'Runs', 'SBs', 'Ks', 'Ws', 'SVs', 'Rank']

# Define column options for each column we want to include
columnDefs = [
    {
    'field': col,
    'headerName': col,
    'type': 'rightAligned',
    'width': 13 if col in small_cols else 40,
    'height': 20,
    'sortable': True,
    'sortingOrder': ['desc', 'asc', None]
    } for col in list(table_df.columns) if col != 'on_team'
]

# Format the decimal numbers for certain metrics
for colDef in columnDefs:
    if colDef['field'] in {'AVG', 'xwOBA'}:
        colDef['type'] = ['numericColumn', 'customNumericFormat']
        colDef['precision'] = 3
    elif colDef['field'] in {'ERA', 'WHIP', 'xERA'}:
        colDef['type'] = ['numericColumn', 'customNumericFormat']
        colDef['precision'] = 2
    elif colDef['field'] in {'K-BB%', 'HardHit%'}:
        colDef['type'] = ['numericColumn', 'customNumericFormat']
        colDef['precision'] = 1

# Set the name column (always the first one) to be left-aligned
columnDefs[0]['type'] = 'leftAligned'
columnDefs[0]['width'] = 70

# Second column (either ABs or IPs) and last columcan also be smaller
columnDefs[1]['width'] = 10

with l_column:
    # Define CSS rule to color the rows for every player on our team.
    cellStyle = JsCode(
        r"""
        function(cellClassParams) {
            if (cellClassParams.data.on_team) {
                return {'background-color': '#a6761d'}
            }
            return {};
        }
        """)

    # Define the font size for the table
    css = {
            ".ag-row": {"font-size": "12pt"},
            ".ag-header": {"font-size": "12pt"}
        }

    grid_builder = GridOptionsBuilder.from_dataframe(table_df)
    grid_options = grid_builder.build()

    # Add the cell style rule to each column
    grid_options['defaultColDef']['cellStyle'] = cellStyle
    # Set height/width of columns automatically
    grid_options['defaultColDef']['autoHeight'] = True
    grid_options['defaultColDef']['autoWidth'] = True

    grid_options['columnDefs'] = columnDefs

    # Add the table to our dashboard
    AgGrid(table_df, gridOptions=grid_options, allow_unsafe_jscode=True,
           fit_columns_on_grid_load=True, custom_css=css,
           height=485)

########################################################################################
##  End Table ##########################################################################
########################################################################################

########################################################################################
##  Begin Plot #########################################################################
########################################################################################

# Define different x/y values and domains based on whether we're looking at hitters
# or pitchers. Also define a column, `size_encoding`, that will be used to determine
# the size of the markers on the plot.
if chosen_position in BATTING_POSITIONS:
    x_val = 'xwOBA'
    y_val = 'wRC+'
    size_encoding = 'HH%'
    x_domain = [0.2, 0.45]
    y_domain = [50, 200]
else:
    x_val = 'xERA'
    y_val = 'Stuff+'
    size_encoding = 'K-BB%'
    x_domain = [5.5, 1.5]
    y_domain = [80, 135]

# Create a specific DF for the plot object
plot_df = df[df['term'] == term]

# Apply minimum thresholds
if chosen_position in BATTING_POSITIONS:
    plot_df = plot_df[(plot_df['ABs'] >= THRESHOLDS['All Batters'][term]) | (plot_df['on_team'])]
    plot_df['label_size'] = [1.4 for _ in range(len(plot_df))]
else:
    plot_df = plot_df[(plot_df['IP'] >= THRESHOLDS[chosen_position][term]) | (plot_df['on_team'])]
    plot_df['label_size'] = [1 for _ in range(len(plot_df))]

display_number = 25 if 'All' in chosen_position else 15

plot_df = plot_df.sort_values(by=['on_team', 'Rank'], ascending=[False, True]).head(display_number)

# Create a last name column to use for chart labels
plot_df['last_name'] = plot_df.apply(lambda row: ' '.join(row['Name'].split(' ')[1:]), axis=1)

with r_column:
    # Creates a scatter plot of players for each position, highlighting players on our team
    chart = (
        alt.Chart(plot_df,
                  width=300,
                  height=600)
        .mark_circle()
        .encode(
            color=alt.Color('on_team').scale(scheme='dark2', reverse=True, domain=[False, True]),
            size=size_encoding,
            tooltip=['Name', 'Team', 'Rank', y_val, x_val, size_encoding, 'Position(s)'],
            x=alt.X(x_val, scale=alt.Scale(domain=x_domain)),
            y=alt.Y(y_val, scale=alt.Scale(domain=y_domain)),
            text='Name')
    )

    # Create the corresponding labels for each player to add to the plot
    labels = chart.mark_text(
        align='left',
        dx=9,
        dy=9,
        fontSize=10,
        limit=100
    ).encode(
        text='last_name',
        size='label_size'
    )
    st.altair_chart(chart + labels)

########################################################################################
##  End Plot ###########################################################################
########################################################################################

########################################################################################
## End Main Script #####################################################################
########################################################################################
