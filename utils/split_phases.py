#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split-level Processing Phases
Each split is broken into granular phases to ensure UI never blocks
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class SplitPhase(Enum):
    """Granular phases for processing a single split"""
    GEMINI_CALL = "gemini_call"          # Call Gemini API
    JSON_PARSE = "json_parse"            # Parse JSON response
    POSTPROCESS = "postprocess"          # Post-process data
    VALIDATION = "validation"            # Validate data
    COMPLETED = "completed"              # Split processing complete
    FAILED = "failed"                    # Split processing failed


@dataclass
class SplitProcessingState:
    """State for processing a single split across multiple phases"""

    # Split identification
    split_index: int = 0
    split_path: str = ""
    page_start: int = 0
    page_end: int = 0

    # Current phase
    phase: SplitPhase = SplitPhase.GEMINI_CALL

    # Intermediate data (stored between phases)
    gemini_response: Optional[Any] = None  # Raw Gemini API response
    parsed_json: Optional[List[Dict]] = None  # Parsed JSON data
    processed_data: Optional[List[Dict]] = None  # Post-processed data
    validated_data: Optional[List[Dict]] = None  # Final validated data

    # Error tracking
    error: Optional[str] = None

    # Phase attempt counter (for timeout detection)
    phase_attempts: int = 0
    max_phase_attempts: int = 3  # If same phase repeats 3 times, abort

    def advance_phase(self):
        """Advance to the next phase"""
        phase_order = [
            SplitPhase.GEMINI_CALL,
            SplitPhase.JSON_PARSE,
            SplitPhase.POSTPROCESS,
            SplitPhase.VALIDATION,
            SplitPhase.COMPLETED
        ]

        current_idx = phase_order.index(self.phase)
        if current_idx < len(phase_order) - 1:
            self.phase = phase_order[current_idx + 1]
            self.phase_attempts = 0  # Reset counter on phase change
            logger.debug(f"Split {self.split_index}: advanced to {self.phase.value}")
        else:
            self.phase = SplitPhase.COMPLETED
            logger.info(f"Split {self.split_index}: all phases completed")

    def mark_failed(self, error: str):
        """Mark this split as failed"""
        self.phase = SplitPhase.FAILED
        self.error = error
        logger.error(f"Split {self.split_index} failed: {error}")

    def increment_attempt(self) -> bool:
        """
        Increment phase attempt counter

        Returns:
            bool: True if should continue, False if max attempts reached
        """
        self.phase_attempts += 1
        if self.phase_attempts >= self.max_phase_attempts:
            logger.warning(
                f"Split {self.split_index} phase {self.phase.value} "
                f"reached max attempts ({self.max_phase_attempts})"
            )
            return False
        return True

    def get_final_data(self) -> List[Dict]:
        """Get the final processed data"""
        return self.validated_data or []

    def is_complete(self) -> bool:
        """Check if split processing is complete (success or failure)"""
        return self.phase in [SplitPhase.COMPLETED, SplitPhase.FAILED]
