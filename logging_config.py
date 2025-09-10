#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging Configuration Module

Forces root logger initialization to ensure INFO logs are visible
even under Uvicorn/Gunicorn suppression.
"""

import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Setup root logger with forced output to stdout.
    This ensures logging works consistently across different environments
    including Uvicorn/Gunicorn which may suppress logging.
    
    Args:
        level: Logging level (default: INFO)
    """
    # Get root logger and clear all existing handlers
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    
    # Create new stdout handler with consistent formatting
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add handler to root and set level
    root.addHandler(handler)
    root.setLevel(level)
    
    # Ensure propagation is enabled
    root.propagate = True
    
    # Log successful initialization
    root.info("Logging configuration initialized successfully")