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
    Fetches the latest data artifact from GitHub Actions.
    """
    url = "https://api.github.com/repos/hockey-stats/streamlit-dashboard/actions/artifacts"
    payload = {}

    pat = os.environ.get("GITHUB_PAT")
    if not pat:
        print("GITHUB_PAT environment variable not found.")
        if os.path.exists("data"):
            return
        raise ValueError("GITHUB_PAT not set and no local data available.")

    headers = {"Authorization": f"Bearer {pat}"}
    output_filename = "data.zip"

    try:
        response = requests.request(
            "GET", url, headers=headers, data=payload, timeout=10
        )
        response.raise_for_status()
        response_body = response.json()
    except Exception as e:
        print(f"Error fetching artifacts: {e}")
        if os.path.exists("data"):
            return
        raise e

    download_url = None

    # 1. Look for 'public-stats-data' (the new one)
    # The list is usually sorted by most recent first
    artifacts = response_body.get("artifacts", [])
    print(f"Total artifacts found: {len(artifacts)}")

    for artifact in artifacts:
        if artifact["name"] == "public-stats-data":
            download_url = artifact["archive_download_url"]
            print(
                f"Found latest public-stats-data artifact created at {artifact['created_at']}"
            )
            break

    # 2. Fallback to old name if not found
    if not download_url:
        for artifact in response_body.get("artifacts", []):
            if artifact["name"] == "dashboard-fa-data":
                download_url = artifact["archive_download_url"]
                print(
                    f"Found fallback dashboard-fa-data artifact created at {artifact['created_at']}"
                )
                break

    if not download_url:
        if os.path.exists("data"):
            print(f"No artifacts found, using existing local data.")
            return
        raise ValueError(f"No data artifacts found and no local data available.")

    if download_url:
        print(f"Downloading artifact from {download_url}")
        dl_response = requests.request(
            "GET", download_url, headers=headers, data=payload, timeout=20
        )
        with open(output_filename, "wb") as fo:
            fo.write(dl_response.content)

        with zipfile.ZipFile(output_filename, "r") as zip_ref:
            zip_ref.extractall("data")
        print("Data extraction successful.")


import time


def trigger_workflow() -> bool:
    """
    Triggers the GitHub Action workflow.
    """
    url = "https://api.github.com/repos/hockey-stats/streamlit-dashboard/actions/workflows/update_public_stats.yml/dispatches"
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
    Retrieves the latest workflow run for update_public_stats.yml
    """
    url = "https://api.github.com/repos/hockey-stats/streamlit-dashboard/actions/workflows/update_public_stats.yml/runs"
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
