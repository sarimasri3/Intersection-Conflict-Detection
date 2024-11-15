# src/utils.py

"""
Utility Functions Module

This module contains utility functions for parsing vehicle scenarios into
readable text descriptions and formatting analysis results.

Author: Your Name
Date: YYYY-MM-DD
"""

import json


def parse_scenario_to_string(scenario_string):
    """
    Converts a vehicle scenario JSON string into a well-formatted text description.

    Args:
        scenario_string (str): JSON string of the vehicle scenario.

    Returns:
        str: Formatted text description of the scenario.
    """
    scenario_data = json.loads(scenario_string)
    vehicles = scenario_data.get("vehicles_scenario", [])
    scenario_description = []

    for vehicle in vehicles:
        vehicle_id = vehicle.get("vehicle_id", "Unknown")
        lane = vehicle.get("lane", "Unknown")
        speed = vehicle.get("speed", "Unknown")
        distance = vehicle.get("distance_to_intersection", "Unknown")
        direction = vehicle.get("direction", "Unknown")
        destination = vehicle.get("destination", "Unknown")

        vehicle_description = (
            f"Vehicle {vehicle_id} is in lane {lane}, approaching from the {direction}, "
            f"traveling at {speed:.2f} km/h, and is {distance:.2f} meters away from the intersection, "
            f"heading towards {destination}."
        )
        scenario_description.append(vehicle_description)

    readable_string = "\n".join(scenario_description)
    return readable_string


def parse_analysis_to_string(row):
    """
    Parses the analysis results into a readable string.

    Args:
        row (pd.Series): A row from the dataset containing conflict analysis results.

    Returns:
        str: Formatted analysis of conflicts and decisions.
    """
    # If no conflict, return "No"
    if row['is_conflict'] == 'no':
        return "No conflict detected."

    # Conflict detected
    conflict_status = "Conflict detected."

    # Conflicts Overview
    conflicts_summary = f"Number of conflicts: {row['number_of_conflicts']}"

    # Involved vehicles in conflicts
    conflict_vehicles = row['conflict_vehicles']
    involved_vehicles = ""
    if conflict_vehicles:
        involved_pairs = [
            f"Vehicle {v['vehicle1_id']} and Vehicle {v['vehicle2_id']}"
            for v in conflict_vehicles
        ]
        involved_vehicles = "Involved vehicles: " + ", ".join(involved_pairs) + "."
    else:
        involved_vehicles = "No vehicles involved in conflicts."

    # Actions & Decisions
    decisions = row['decisions']
    decisions_summary = "\n".join(decisions) if decisions else "No decisions made."

    # Priority Assignment
    priority_order = row['priority_order']
    if priority_order:
        priority_list = [
            f"Vehicle {vehicle_id}: Priority {priority}"
            for vehicle_id, priority in priority_order.items()
            if priority is not None
        ]
        priority_summary = "Priority Order:\n" + "\n".join(priority_list)
    else:
        priority_summary = "No priority information available."

    # Vehicle Waiting Times
    waiting_times = row['waiting_times']
    if waiting_times:
        waiting_list = [
            f"Vehicle {vehicle_id}: {waiting_time} second(s)"
            for vehicle_id, waiting_time in waiting_times.items()
        ]
        waiting_summary = "Waiting Times:\n" + "\n".join(waiting_list)
    else:
        waiting_summary = "No waiting times recorded."

    # Combine all parts into a full report
    report = (
        f"{conflict_status}\n"
        f"{conflicts_summary}\n"
        f"{involved_vehicles}\n"
        f"Actions & Decisions:\n{decisions_summary}\n"
        f"{priority_summary}\n"
        f"{waiting_summary}"
    )
    return report
