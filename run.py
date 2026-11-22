#!/usr/bin/env python3
"""Convenience launcher: python run.py [--port 8000]"""
import sys

from refgap.web import main

if __name__ == "__main__":
    sys.exit(main())
