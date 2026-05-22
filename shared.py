import os
import json
import zipfile
import requests
import streamlit as st
from datetime import datetime, timedelta


@st.cache_data
def load_data(today: str) -> None:
    """
    Function to be run at the initialization of the dashboard.
    """
    url = "https://api.github.com/repos/hockey-stats/streamlit-dashboard/actions/artifacts"
    payload = {}
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_PAT']}"}
    output_filename = "data.zip"

    response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
    response_body = json.loads(response.text)

    download_url = None
    for artifact in response_body["artifacts"]:
        if artifact["name"] == "dashboard-fa-data":
            artifact_creation_date = artifact["created_at"].split("T")[0]
            if today == artifact_creation_date:
                download_url = artifact["archive_download_url"]
                break
    else:
        # If today's data is not found, try to find the most recent one instead of crashing
        # Or just use the existing data if available.
        # For this dashboard, it seems it expects today's data.
        if os.path.exists("data"):
            print(f"Data for {today} not found, using existing local data.")
            return
        raise ValueError(f"Data for {today} not found and no local data available.")

    if download_url:
        print(f"Found artifact at {download_url}")
        dl_response = requests.request(
            "GET", download_url, headers=headers, data=payload, timeout=20
        )
        with open(output_filename, "wb") as fo:
            fo.write(dl_response.content)

        with zipfile.ZipFile(output_filename, "r") as zip_ref:
            zip_ref.extractall("data")


def get_today_date():
    today = datetime.today()
    # If checking before 7am UTC, use yesterday's data instead
    if today.hour <= 7:
        today -= timedelta(days=1)
    return today
