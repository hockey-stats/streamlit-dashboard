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


import time


def trigger_workflow() -> bool:
    """
    Triggers the GitHub Action workflow.
    """
    url = "https://api.github.com/repos/hockey-stats/streamlit-dashboard/actions/workflows/update_dashboard_data.yml/dispatches"
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_PAT')}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {"ref": "main"}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        return response.status_code == 204
    except Exception as e:
        print(f"Error triggering workflow: {e}")
        return False


def get_latest_workflow_run():
    """
    Retrieves the latest workflow run for update_dashboard_data.yml
    """
    url = "https://api.github.com/repos/hockey-stats/streamlit-dashboard/actions/workflows/update_dashboard_data.yml/runs"
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_PAT')}",
        "Accept": "application/vnd.github.v3+json",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            runs = response.json().get("workflow_runs", [])
            if runs:
                return runs[0]
        return None
    except Exception:
        return None


def get_run_status(run_id: int):
    """
    Checks the status of a specific workflow run.
    """
    url = f"https://api.github.com/repos/hockey-stats/streamlit-dashboard/actions/runs/{run_id}"
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_PAT')}",
        "Accept": "application/vnd.github.v3+json",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("status"), data.get("conclusion")
        return None, None
    except Exception:
        return None, None


def get_today_date():
    today = datetime.today()
    # If checking before 7am UTC, use yesterday's data instead
    if today.hour <= 7:
        today -= timedelta(days=1)
    return today
