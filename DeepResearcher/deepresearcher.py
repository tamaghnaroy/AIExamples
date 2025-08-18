#!/usr/bin/env python

from deepresearcher.cli.main import entry_point
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    entry_point()