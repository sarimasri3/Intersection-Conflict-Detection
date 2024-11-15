# src/__init__.py

"""
Initialization file for the src package.
"""

from .conflict_detection import (
    Vehicle,
    parse_intersection_layout,
    parse_vehicles,
    detect_conflicts,
    paths_cross,
    arrival_time_close,
    is_vehicle_on_right,
    apply_priority_rules,
    compute_waiting_times,
)