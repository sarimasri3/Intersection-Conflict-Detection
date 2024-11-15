# src/data_generation.py

"""
Data Generation Module

This module contains functions to generate random vehicle scenarios and datasets
for testing the conflict detection system.

Author: Your Name
Date: YYYY-MM-DD
"""

import json
import pandas as pd
import random
import math
from .conflict_detection import (
    parse_vehicles,
    detect_conflicts,
    parse_intersection_layout,
)

def generate_vehicle_scenario(num_vehicles, intersection_layout, fixed_vehicle_count=True):
    """
    Generates a random vehicle scenario.

    Args:
        num_vehicles (int): Number of vehicles in the scenario.
        intersection_layout (dict): The intersection layout.
        fixed_vehicle_count (bool): If True, use num_vehicles; else, randomly choose between 2 and num_vehicles.

    Returns:
        dict: A vehicle scenario containing a list of vehicles.
    """
    if not fixed_vehicle_count:
        num_vehicles = random.randint(2, num_vehicles)

    vehicles = []
    vehicle_ids = set()
    for _ in range(num_vehicles):
        # Generate unique vehicle ID
        while True:
            vehicle_id = f"V{random.randint(1000, 9999)}"
            if vehicle_id not in vehicle_ids:
                vehicle_ids.add(vehicle_id)
                break

        # Choose a random direction
        direction = random.choice(['north', 'east', 'south', 'west'])

        # Get possible lanes for the chosen direction
        direction_lanes = list(intersection_layout[direction].keys())
        lane = random.choice(direction_lanes)

        # Get possible destinations for the chosen direction and lane
        possible_destinations = intersection_layout[direction][lane]
        destination = random.choice(possible_destinations)

        # Generate random speed and distance
        speed = random.uniform(20, 80)  # Speed between 20 km/h and 80 km/h
        distance_to_intersection = random.uniform(50, 500)  # Distance between 50 m and 500 m

        vehicles.append({
            "vehicle_id": vehicle_id,
            "lane": lane,
            "speed": speed,
            "distance_to_intersection": distance_to_intersection,
            "direction": direction,
            "destination": destination
        })

    scenario = {"vehicles_scenario": vehicles}
    return scenario


def generate_dataset(total_records=50000, num_vehicles=5, fixed_vehicle_count=True):
    """
    Generates a dataset containing vehicle scenario data.

    Args:
        total_records (int): Total number of records to generate.
        num_vehicles (int): Maximum number of vehicles in each scenario.
        fixed_vehicle_count (bool): If True, use num_vehicles; else, randomly choose between 2 and num_vehicles.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the dataset.
    """
    data = []
    conflict_counts = {'yes': 0, 'no': 0}

    # Load the intersection layout
    intersection_layout_json = '''
    {
        "intersection_layout": {
            "north": {
                "1": ["F", "H"],
                "2": ["E", "D", "C"]
            },
            "east": {
                "3": ["H", "B"],
                "4": ["G", "E", "F"]
            },
            "south": {
                "5": ["B", "D"],
                "6": ["A", "G", "H"]
            },
            "west": {
                "7": ["D", "F"],
                "8": ["B", "C", "A"]
            }
        }
    }
    '''
    intersection_layout_data = json.loads(intersection_layout_json)
    intersection_layout = parse_intersection_layout(intersection_layout_data)

    while len(data) < total_records:
        scenario = generate_vehicle_scenario(num_vehicles, intersection_layout, fixed_vehicle_count)
        try:
            vehicles = parse_vehicles(scenario, intersection_layout)
        except ValueError as e:
            continue  # Skip scenarios with invalid data

        conflicts = detect_conflicts(vehicles)

        is_conflict = 'yes' if conflicts else 'no'

        # Balance the dataset
        if conflict_counts[is_conflict] >= total_records / 2:
            continue  # Skip to balance the dataset
        conflict_counts[is_conflict] += 1

        number_of_conflicts = len(conflicts)
        places_of_conflicts = ['intersection' for _ in conflicts]  # All conflicts are at the intersection

        # Extract conflict_vehicles and decisions
        conflict_vehicles = []
        decisions = []
        all_conflict_vehicle_ids = set()
        for conflict in conflicts:
            conflict_vehicle_ids = set([conflict['vehicle1_id'], conflict['vehicle2_id']])
            all_conflict_vehicle_ids.update(conflict_vehicle_ids)
            conflict_vehicles.append({
                'vehicle1_id': conflict['vehicle1_id'],
                'vehicle2_id': conflict['vehicle2_id']
            })
            decisions.append(conflict['decision'])

        # Now, for all vehicles involved in conflicts, recompute priority orders and waiting times
        # Build a list of vehicles involved in conflicts
        conflicting_vehicles = [v for v in vehicles if v.vehicle_id in all_conflict_vehicle_ids]

        # Apply priority rules to all conflicting vehicles
        overall_priority_order = {}
        overall_waiting_times = {}

        if conflicting_vehicles:
            # Sort vehicles based on their time to intersection
            sorted_vehicles = sorted(conflicting_vehicles, key=lambda v: v.time_to_intersection)

            # Assign priorities based on arrival times
            for idx, vehicle in enumerate(sorted_vehicles):
                overall_priority_order[vehicle.vehicle_id] = idx + 1  # Priority 1 is highest

            # Compute waiting times
            traversal_time = 2  # Time to clear the intersection in seconds
            vehicle_arrival_times = {v.vehicle_id: v.time_to_intersection for v in conflicting_vehicles}
            for idx, vehicle in enumerate(sorted_vehicles):
                if idx == 0:
                    # First vehicle doesn't wait
                    overall_waiting_times[vehicle.vehicle_id] = 0
                else:
                    # Wait until the previous vehicle has cleared the intersection
                    prev_vehicle = sorted_vehicles[idx - 1]
                    required_arrival_time = vehicle_arrival_times[prev_vehicle.vehicle_id] + traversal_time
                    wait_time = max(0, required_arrival_time - vehicle_arrival_times[vehicle.vehicle_id])
                    overall_waiting_times[vehicle.vehicle_id] = math.ceil(wait_time)
                    # Update the vehicle's arrival time after waiting
                    vehicle_arrival_times[vehicle.vehicle_id] += wait_time

        # For vehicles not involved in conflicts, set priority and waiting time to default values
        non_conflicting_vehicles = [v for v in vehicles if v.vehicle_id not in all_conflict_vehicle_ids]
        for vehicle in non_conflicting_vehicles:
            overall_priority_order[vehicle.vehicle_id] = None  # No priority needed
            overall_waiting_times[vehicle.vehicle_id] = 0  # No waiting time

        record = {
            'scenario': json.dumps(scenario),
            'is_conflict': is_conflict,
            'number_of_conflicts': number_of_conflicts,
            'places_of_conflicts': places_of_conflicts,
            'conflict_vehicles': conflict_vehicles,
            'decisions': decisions,
            'priority_order': overall_priority_order,
            'waiting_times': overall_waiting_times
        }
        data.append(record)

    dataset = pd.DataFrame(data)
    return dataset

