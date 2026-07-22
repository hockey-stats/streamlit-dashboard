import streamlit as st

import shared
import time


# --- SETTINGS ---
st.set_page_config(page_title="Fantasy Baseball Stats", page_icon="⚾", layout="wide")

# Define pages
pages = [
    st.Page("views/free_agent_summary.py", title="Free Agent Summary", icon="📋"),
    st.Page("views/league_weekly_stats.py", title="League Weekly Stats", icon="📈"),
    st.Page("views/probable_pitchers.py", title="Probable Pitchers", icon="⚾"),
]

# Initialize navigation
pg = st.navigation(pages, position="hidden")

# Render header and navigation
st.markdown("### ⚾ Fantasy Baseball Dashboard")

nav_col, refresh_col = st.columns([5, 1])

with nav_col:
    selected_page_title = st.segmented_control(
        "Navigation",
        options=[p.title for p in pages],
        default=pg.title,
        label_visibility="collapsed",
    )

with refresh_col:
    do_refresh = st.button(
        "🔄 Refresh",
        help="Trigger GitHub Actions and wait for data update",
        use_container_width=True,
    )

if do_refresh:
    with st.status("Triggering update...") as status:
        if shared.trigger_workflow():
            status.update(
                label="Workflow triggered. Waiting for run to start...", state="running"
            )
            # Poll for the new run to appear
            run_id = None
            for _ in range(12):  # Try for 60 seconds
                time.sleep(5)
                latest_run = shared.get_latest_workflow_run()
                if latest_run and latest_run.get("status") in [
                    "queued",
                    "in_progress",
                    "requested",
                ]:
                    run_id = latest_run.get("id")
                    break

            if run_id:
                status.update(
                    label=f"Workflow in progress (Run #{run_id})...", state="running"
                )
                while True:
                    run_status, conclusion = shared.get_run_status(run_id)
                    if run_status == "completed":
                        if conclusion == "success":
                            status.update(
                                label="Update complete! Refreshing dashboard...",
                                state="complete",
                            )
                            st.cache_data.clear()
                            time.sleep(2)
                            st.rerun()
                        else:
                            status.update(
                                label=f"Workflow failed: {conclusion}", state="error"
                            )
                            break
                    time.sleep(10)
            else:
                status.update(
                    label="Timed out waiting for workflow to start. Check GitHub Actions.",
                    state="error",
                )
        else:
            status.update(
                label="Failed to trigger workflow. Check GITHUB_PAT permissions.",
                state="error",
            )

# Update session state and switch page if selection changed
if selected_page_title and selected_page_title != pg.title:
    # Find the matching page object
    target_page = [p for p in pages if p.title == selected_page_title][0]
    st.switch_page(target_page)

# Run the selected page
pg.run()