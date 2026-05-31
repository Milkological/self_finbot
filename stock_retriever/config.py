# config.py
# Responsible for loading Reddit API credentials from a .env file and
# exposing them to the rest of the application via a single function.
#
# WHY use a .env file?
#   Hardcoding secrets in source code is a security risk (OWASP A02: Cryptographic Failures).
#   Storing them in environment variables (loaded from a local .env file) keeps them
#   out of version control and off any shared or public systems.

import os
from dotenv import load_dotenv

# Load the .env file from the current working directory into the process environment.
# This call is safe to make at import time; it does nothing if .env doesn't exist.
# Variables already in the environment are NOT overwritten by load_dotenv().
load_dotenv()


def get_reddit_credentials() -> dict:
    """
    Reads Reddit API credentials from environment variables and returns them
    as a dict. Raises EnvironmentError with a helpful message if any variable
    is missing so the user knows exactly what to fix.

    Returns:
        A dict with keys: "client_id", "client_secret", "user_agent"

    Raises:
        EnvironmentError: If one or more required env variables are not set.
    """
    # Read each credential from the environment.
    # os.getenv returns None (not an error) if the variable is missing.
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    # Collect the names of any missing variables so we can report all of them at once
    # rather than surfacing errors one at a time.
    missing = [
        name for name, value in {
            "REDDIT_CLIENT_ID": client_id,
            "REDDIT_CLIENT_SECRET": client_secret,
            "REDDIT_USER_AGENT": user_agent,
        }.items()
        if not value
    ]

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}.\n"
            "Copy .env.example to .env and fill in your Reddit API credentials."
        )

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "user_agent": user_agent,
    }
