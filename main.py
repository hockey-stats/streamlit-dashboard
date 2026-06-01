import streamlit as st

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

# Render horizontal navigation menu
st.markdown("### ⚾ Fantasy Baseball Dashboard")

selected_page_title = st.segmented_control(
    "Navigation",
    options=[p.title for p in pages],
    default=pg.title,
    label_visibility="collapsed",
)

# Update session state and switch page if selection changed
if selected_page_title and selected_page_title != pg.title:
    # Find the matching page object
    target_page = [p for p in pages if p.title == selected_page_title][0]
    st.switch_page(target_page)

# Run the selected page
pg.run()
