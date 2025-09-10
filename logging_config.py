#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging Configuration Module (Singleton)

Forces root logger initialization to ensure INFO logs are visible
even under Uvicorn/Gunicorn suppression.
"""

import logging
import sys
import threading

# Singleton state
_logging_initialized = False
_init_lock = threading.Lock()


def setup_logging(level=logging.INFO):
    """
    Setup root logger with forced output to stdout (singleton pattern).
    This ensures logging works consistently across different environments
    including Uvicorn/Gunicorn which may suppress logging.
    
    Only executes once per process to prevent duplicate initialization.
    
    Args:
        level: Logging level (default: INFO)
    """
    global _logging_initialized
    
    with _init_lock:
        if _logging_initialized:
            # Already initialized - skip silently
            return
        
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
        
        # Mark as initialized
        _logging_initialized = True
        
        # Log successful initialization
        root.info("Logging configuration initialized successfully (singleton)")


def is_logging_initialized():
    """
    Check if logging has been initialized.
    
    Returns:
        bool: True if logging has been initialized
    """
    return _logging_initialized