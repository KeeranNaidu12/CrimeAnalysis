"""
Orchestrator wrapper for API
Re-exports functions from your existing model_orchestrator.py
"""

import sys
from pathlib import Path

# Add the parent directory to path to import your existing module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from your existing file
from model_orchestrator import (
    load_all_models,
    predict_for_date_range,
    CRIME_DISPLAY_NAMES,
    MODELS_DIR,
    MODEL_FILES,
    create_neighbourhood_mappings,
    extract_neighbourhood_id,
    clean_neighbourhood_name,
    find_neighbourhood_by_input,
    to_timestamp,
    Colours
)

# Re-export for API use
__all__ = [
    'load_all_models',
    'predict_for_date_range',
    'CRIME_DISPLAY_NAMES',
    'MODELS_DIR',
    'MODEL_FILES',
    'create_neighbourhood_mappings',
    'extract_neighbourhood_id',
    'clean_neighbourhood_name',
    'find_neighbourhood_by_input',
    'to_timestamp'
]