"""Test environment: API tests run without the .env login credentials."""

import os

os.environ["AUTH_USERNAME"] = ""
os.environ["AUTH_PASSWORD"] = ""