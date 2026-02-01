"""
arXiv Researcher Dashboard
==========================

Interactive Streamlit dashboard for exploring research data.
"""

import argparse
import sys


def run_dashboard() -> None:
    """Launch the Streamlit dashboard."""
    parser = argparse.ArgumentParser(
        prog="arxiv-dashboard",
        description="Launch the arXiv Researcher interactive dashboard"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="arxiv_data",
        help="Directory containing analysis data"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to run the dashboard on"
    )

    args = parser.parse_args()

    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("Error: streamlit is not installed.")
        print("Install it with: pip install arxiv-researcher[dashboard]")
        sys.exit(1)

    # Get the path to the main dashboard file
    import os
    dashboard_file = os.path.join(
        os.path.dirname(__file__),
        "_dashboard_app.py"
    )

    # If the dashboard app doesn't exist, use the standalone one
    if not os.path.exists(dashboard_file):
        # Try the root dashboard.py
        root_dashboard = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "dashboard.py"
        )
        if os.path.exists(root_dashboard):
            dashboard_file = root_dashboard
        else:
            print("Error: Dashboard file not found.")
            sys.exit(1)

    sys.argv = [
        "streamlit", "run",
        dashboard_file,
        "--server.port", str(args.port),
        "--",
        "--data", args.data
    ]

    stcli.main()


if __name__ == "__main__":
    run_dashboard()
