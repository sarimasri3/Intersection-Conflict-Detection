# src/conflict_detection.py

"""
Conflict Detection Module

This module contains classes and functions to detect potential conflicts
between vehicles approaching an intersection based on their trajectories,
arrival times, and movement types.

Author: Your Name
Date: YYYY-MM-DD
"""

import json
import math
import warnings

# Mapping of opposite directions
OPPOSITE_DIRECTIONS = {
    'north': 'south',
    'east': 'west',
    'south': 'north',
    'west': 'east'
}

# Enable or disable logging for debugging
log = False


def parse_intersection_layout(data):
    """
    Parses the intersection layout from the given data.

    Args:
        data (dict): Data containing the intersection layout.

    Returns:
        dict: Parsed intersection layout.
    """
    return data['intersection_layout']


class Vehicle:
    """
    Represents a vehicle approaching an intersection.

    Attributes:
        vehicle_id (str): Unique identifier for the vehicle.
        lane (str): Lane number.
        speed (float): Speed in km/h.
        distance_to_intersection (float): Distance to intersection in meters.
        direction (str): Direction of approach ('north', 'east', 'south', 'west').
        destination (str): Destination road.
        time_to_intersection (float): Time to reach the intersection in seconds.
        movement_type (str): Type of movement ('straight', 'left', 'right', or 'unknown').
    """

    VALID_DIRECTIONS = ['north', 'east', 'south', 'west']

    def __init__(
        self,
        vehicle_id,
        lane,
        speed,
        distance_to_intersection,
        direction,
        destination,
        intersection_layout
    ):
        """
        Initializes a Vehicle instance.

        Args:
            vehicle_id (str): Unique identifier for the vehicle.
            lane (str): Lane number.
            speed (float): Speed in km/h.
            distance_to_intersection (float): Distance to intersection in meters.
            direction (str): Direction of approach.
            destination (str): Destination road.
            intersection_layout (dict): Layout of the intersection.
        """
        self.vehicle_id = vehicle_id
        self.lane = str(lane)
        self.speed = speed  # in km/h
        self.distance_to_intersection = distance_to_intersection  # in meters
        self.direction = direction.lower()
        self.destination = destination
        self.validate_inputs()
        self.time_to_intersection = self.compute_time_to_intersection()
        self.movement_type = self.get_movement_type(intersection_layout)
        # If logging is enabled, print vehicle information after initialization
        if log:
            print(f"Initialized Vehicle {self.vehicle_id}: lane={self.lane}, speed={self.speed}, "
                  f"distance_to_intersection={self.distance_to_intersection}, direction={self.direction}, "
                  f"destination={self.destination}, movement_type={self.movement_type}, "
                  f"time_to_intersection={self.time_to_intersection:.2f}s")

    def validate_inputs(self):
        """
        Validates the inputs for the vehicle.
        """
        if self.speed < 0:
            raise ValueError(f"Vehicle {self.vehicle_id} has negative speed.")
        if self.distance_to_intersection < 0:
            raise ValueError(f"Vehicle {self.vehicle_id} has negative distance to intersection.")
        if self.direction not in self.VALID_DIRECTIONS:
            raise ValueError(f"Vehicle {self.vehicle_id} has invalid direction '{self.direction}'.")
        if not self.vehicle_id:
            raise ValueError("Vehicle ID cannot be empty.")

    def compute_time_to_intersection(self):
        """
        Computes the time for the vehicle to reach the intersection.

        Returns:
            float: Time to intersection in seconds.
        """
        speed_m_per_s = (self.speed * 1000) / 3600  # Convert km/h to m/s
        if speed_m_per_s == 0:
            return float('inf')  # Infinite time if speed is zero
        time = self.distance_to_intersection / speed_m_per_s
        return time

    def get_movement_type(self, intersection_layout):
        """
        Determines the movement type (straight, left, right) based on the intersection layout.

        Args:
            intersection_layout (dict): Layout of the intersection.

        Returns:
            str: Movement type ('straight', 'left', 'right', or 'unknown').
        """
        direction = self.direction
        lane = self.lane
        destination = self.destination

        lane_destinations = intersection_layout.get(direction, {}).get(lane, [])
        if not lane_destinations:
            warnings.warn(
                f"Vehicle {self.vehicle_id} is in an unknown lane '{lane}' for direction '{direction}'.",
                category=UserWarning
            )
            return 'unknown'
        if destination not in lane_destinations:
            warnings.warn(
                f"Destination '{destination}' not accessible from lane '{lane}' for direction '{direction}'.",
                category=UserWarning
            )
            return 'unknown'

        index = lane_destinations.index(destination)
        if lane in ['1', '3', '5', '7']:
            if index == 0:
                movement_type = 'right'
            elif index == 1:
                movement_type = 'straight'
            elif index == 2:
                movement_type = 'left'
            else:
                movement_type = 'unknown'
        elif lane in ['2', '4', '6', '8']:
            movement_type = 'left'  # These lanes are dedicated left-turn lanes
        else:
            movement_type = 'unknown'

        if movement_type == 'unknown':
            warnings.warn(
                f"Vehicle {self.vehicle_id} has unknown movement type.",
                category=UserWarning
            )
        return movement_type


def parse_vehicles(data, intersection_layout):
    """
    Parses vehicle data from the given scenario.

    Args:
        data (dict): Vehicle scenario data.
        intersection_layout (dict): Layout of the intersection.

    Returns:
        list of Vehicle: List of Vehicle objects.
    """
    vehicles = []
    vehicle_ids = set()
    for vehicle_data in data['vehicles_scenario']:
        vehicle_id = vehicle_data['vehicle_id']
        if vehicle_id in vehicle_ids:
            raise ValueError(f"Duplicate vehicle ID detected: {vehicle_id}")
        vehicle_ids.add(vehicle_id)
        vehicle = Vehicle(
            vehicle_id=vehicle_id,
            lane=vehicle_data['lane'],
            speed=vehicle_data['speed'],
            distance_to_intersection=vehicle_data['distance_to_intersection'],
            direction=vehicle_data['direction'],
            destination=vehicle_data['destination'],
            intersection_layout=intersection_layout
        )
        vehicles.append(vehicle)
    return vehicles


def paths_cross(vehicle1, vehicle2):
    """
    Determines if the paths of two vehicles cross.

    Args:
        vehicle1 (Vehicle): First vehicle.
        vehicle2 (Vehicle): Second vehicle.

    Returns:
        bool: True if paths cross, False otherwise.
    """
    if log:
        print(f"Checking if paths cross between Vehicle {vehicle1.vehicle_id} and Vehicle {vehicle2.vehicle_id}")
    if 'unknown' in [vehicle1.movement_type, vehicle2.movement_type]:
        if log:
            print(f"At least one vehicle has unknown movement type: {vehicle1.movement_type}, {vehicle2.movement_type}")
        return False
    if vehicle1.vehicle_id == vehicle2.vehicle_id:
        if log:
            print("Comparing the same vehicle.")
        return False

    # Same direction
    if vehicle1.direction == vehicle2.direction:
        if log:
            print(f"Vehicles are coming from the same direction: {vehicle1.direction}")
        return False

    # Vehicles going straight from opposite directions do not conflict
    if vehicle1.movement_type == 'straight' and vehicle2.movement_type == 'straight' and \
       OPPOSITE_DIRECTIONS[vehicle1.direction] == vehicle2.direction:
        if log:
            print(f"Vehicles are going straight from opposite directions: {vehicle1.direction} and {vehicle2.direction}")
        return False

    # Opposite left turns do not conflict
    if vehicle1.movement_type == 'left' and vehicle2.movement_type == 'left' and \
       OPPOSITE_DIRECTIONS[vehicle1.direction] == vehicle2.direction:
        if log:
            print(f"Vehicles are making left turns from opposite directions: {vehicle1.direction} and {vehicle2.direction}")
        return False

    # Right turns from opposite directions do not conflict
    if vehicle1.movement_type == 'right' and vehicle2.movement_type == 'right' and \
       OPPOSITE_DIRECTIONS[vehicle1.direction] == vehicle2.direction:
        if log:
            print(f"Vehicles are making right turns from opposite directions: {vehicle1.direction} and {vehicle2.direction}")
        return False

    # Right turns from adjacent directions do not conflict
    if vehicle1.movement_type == 'right' and vehicle2.movement_type == 'right' and \
       vehicle1.direction != vehicle2.direction and \
       OPPOSITE_DIRECTIONS[vehicle1.direction] != vehicle2.direction:
        if log:
            print(f"Vehicles are making right turns from adjacent directions: {vehicle1.direction} and {vehicle2.direction}")
        return False

    # Vehicles going straight from perpendicular directions conflict
    if vehicle1.movement_type == 'straight' and vehicle2.movement_type == 'straight' and \
       (vehicle1.direction != vehicle2.direction) and \
       (OPPOSITE_DIRECTIONS[vehicle1.direction] != vehicle2.direction):
        if log:
            print(f"Vehicles are going straight from perpendicular directions: {vehicle1.direction} and {vehicle2.direction}")
        return True

    # Left turn conflicts
    if vehicle1.movement_type == 'left' or vehicle2.movement_type == 'left':
        if log:
            print(f"At least one vehicle is turning left: {vehicle1.movement_type}, {vehicle2.movement_type}")
        return True

    # Right turn vs straight from adjacent directions conflict
    if (vehicle1.movement_type == 'right' and vehicle2.movement_type == 'straight' and \
        (vehicle1.direction != vehicle2.direction) and \
        (OPPOSITE_DIRECTIONS[vehicle1.direction] != vehicle2.direction)) or \
       (vehicle2.movement_type == 'right' and vehicle1.movement_type == 'straight' and \
        (vehicle1.direction != vehicle2.direction) and \
        (OPPOSITE_DIRECTIONS[vehicle2.direction] != vehicle1.direction)):
        if log:
            print("One vehicle is turning right and the other is going straight from adjacent directions.")
        return True

    # For all other cases, assume paths do not cross
    if log:
        print(f"Vehicles do not conflict: {vehicle1.vehicle_id} and {vehicle2.vehicle_id}")
    return False


def arrival_time_close(vehicle1, vehicle2, threshold=4.0):
    """
    Checks if the arrival times of two vehicles are within a certain threshold.

    Args:
        vehicle1 (Vehicle): First vehicle.
        vehicle2 (Vehicle): Second vehicle.
        threshold (float): Time difference threshold in seconds.

    Returns:
        bool: True if arrival times are within the threshold, False otherwise.
    """
    if vehicle1.time_to_intersection == float('inf') or vehicle2.time_to_intersection == float('inf'):
        if log:
            print(f"At least one vehicle has infinite time to intersection: {vehicle1.time_to_intersection}, {vehicle2.time_to_intersection}")
        return False
    time_diff = abs(vehicle1.time_to_intersection - vehicle2.time_to_intersection)
    if log:
        print(f"Time difference between Vehicle {vehicle1.vehicle_id} and Vehicle {vehicle2.vehicle_id}: {time_diff:.2f}s")
    return time_diff <= threshold


def is_vehicle_on_right(vehicle1, vehicle2):
    """
    Determines if vehicle2 is on the right of vehicle1 based on their directions.

    Args:
        vehicle1 (Vehicle): First vehicle.
        vehicle2 (Vehicle): Second vehicle.

    Returns:
        bool: True if vehicle2 is on the right of vehicle1, False otherwise.
    """
    direction_order = ['north', 'east', 'south', 'west']
    idx1 = direction_order.index(vehicle1.direction)
    idx2 = direction_order.index(vehicle2.direction)
    result = (idx2 - idx1) % 4 == 1
    if log:
        print(f"Vehicle {vehicle2.vehicle_id} is {'on the right of' if result else 'not on the right of'} Vehicle {vehicle1.vehicle_id}")
    return result


def apply_priority_rules(vehicle1, vehicle2):
    """
    Applies priority rules to determine which vehicle must yield.

    Args:
        vehicle1 (Vehicle): First vehicle.
        vehicle2 (Vehicle): Second vehicle.

    Returns:
        tuple: (decision message, vehicle priorities dictionary)
    """
    if log:
        print(f"Applying priority rules between Vehicle {vehicle1.vehicle_id} and Vehicle {vehicle2.vehicle_id}")
    time_difference = abs(vehicle1.time_to_intersection - vehicle2.time_to_intersection)
    if log:
        print(f"Time difference: {time_difference:.2f}s")
    priority = {}
    if time_difference <= 1.0:
        if log:
            print("Vehicles arrive within 1 second of each other")
        # 1. Straight over turn
        if vehicle1.movement_type == 'straight' and vehicle2.movement_type != 'straight':
            if log:
                print(f"Vehicle {vehicle1.vehicle_id} is going straight, Vehicle {vehicle2.vehicle_id} is turning")
            decision = f"Potential conflict: Vehicle {vehicle2.vehicle_id} must yield to Vehicle {vehicle1.vehicle_id}"
            priority = {vehicle1.vehicle_id: 1, vehicle2.vehicle_id: 2}
        elif vehicle2.movement_type == 'straight' and vehicle1.movement_type != 'straight':
            if log:
                print(f"Vehicle {vehicle2.vehicle_id} is going straight, Vehicle {vehicle1.vehicle_id} is turning")
            decision = f"Potential conflict: Vehicle {vehicle1.vehicle_id} must yield to Vehicle {vehicle2.vehicle_id}"
            priority = {vehicle2.vehicle_id: 1, vehicle1.vehicle_id: 2}
        # 2. Right turn over left turn
        elif vehicle1.movement_type == 'right' and vehicle2.movement_type == 'left':
            if log:
                print(f"Vehicle {vehicle1.vehicle_id} is turning right, Vehicle {vehicle2.vehicle_id} is turning left")
            decision = f"Potential conflict: Vehicle {vehicle2.vehicle_id} must yield to Vehicle {vehicle1.vehicle_id}"
            priority = {vehicle1.vehicle_id: 1, vehicle2.vehicle_id: 2}
        elif vehicle2.movement_type == 'right' and vehicle1.movement_type == 'left':
            if log:
                print(f"Vehicle {vehicle2.vehicle_id} is turning right, Vehicle {vehicle1.vehicle_id} is turning left")
            decision = f"Potential conflict: Vehicle {vehicle1.vehicle_id} must yield to Vehicle {vehicle2.vehicle_id}"
            priority = {vehicle2.vehicle_id: 1, vehicle1.vehicle_id: 2}
        # 3. Right-hand rule
        else:
            if is_vehicle_on_right(vehicle1, vehicle2):
                if log:
                    print(f"Vehicle {vehicle2.vehicle_id} is on the right of Vehicle {vehicle1.vehicle_id}")
                decision = f"Potential conflict: Vehicle {vehicle1.vehicle_id} must yield to Vehicle {vehicle2.vehicle_id}"
                priority = {vehicle2.vehicle_id: 1, vehicle1.vehicle_id: 2}
            else:
                if log:
                    print(f"Vehicle {vehicle1.vehicle_id} is on the right of Vehicle {vehicle2.vehicle_id}")
                decision = f"Potential conflict: Vehicle {vehicle2.vehicle_id} must yield to Vehicle {vehicle1.vehicle_id}"
                priority = {vehicle1.vehicle_id: 1, vehicle2.vehicle_id: 2}
    else:
        # Vehicle that arrives later must yield
        if vehicle1.time_to_intersection > vehicle2.time_to_intersection:
            if log:
                print(f"Vehicle {vehicle1.vehicle_id} arrives later than Vehicle {vehicle2.vehicle_id}")
            decision = f"Potential conflict: Vehicle {vehicle1.vehicle_id} must yield to Vehicle {vehicle2.vehicle_id}"
            priority = {vehicle2.vehicle_id: 1, vehicle1.vehicle_id: 2}
        else:
            if log:
                print(f"Vehicle {vehicle2.vehicle_id} arrives later than Vehicle {vehicle1.vehicle_id}")
            decision = f"Potential conflict: Vehicle {vehicle2.vehicle_id} must yield to Vehicle {vehicle1.vehicle_id}"
            priority = {vehicle1.vehicle_id: 1, vehicle2.vehicle_id: 2}
    if log:
        print(f"Decision: {decision}")
    return decision, priority


def compute_waiting_times(vehicles, priorities):
    """
    Computes the waiting time for each vehicle based on priority and arrival times.

    Args:
        vehicles (list of Vehicle): List of Vehicle objects.
        priorities (dict): Dictionary mapping vehicle IDs to their priority levels.

    Returns:
        dict: Dictionary mapping vehicle IDs to their waiting times.
    """
    # The vehicle with higher priority (lower number) proceeds first.
    # Waiting time is the difference in arrival times if lower priority vehicle arrives earlier.
    waiting_times = {}
    for vehicle_id, priority in priorities.items():
        vehicle = next((v for v in vehicles if v.vehicle_id == vehicle_id), None)
        if vehicle is None:
            continue
        # Vehicles with priority 1 have zero waiting time
        if priority == 1:
            waiting_times[vehicle_id] = 0
        else:
            # Find the vehicle(s) with higher priority
            higher_priority_vehicles = [v_id for v_id, p in priorities.items() if p < priority]
            max_wait = 0
            for hp_vehicle_id in higher_priority_vehicles:
                hp_vehicle = next((v for v in vehicles if v.vehicle_id == hp_vehicle_id), None)
                if hp_vehicle:
                    # Calculate the additional waiting time needed
                    traversal_time = 2  # Assume it takes 2 seconds to clear the intersection
                    wait_time = max(0, (hp_vehicle.time_to_intersection + traversal_time) - vehicle.time_to_intersection)
                    max_wait = max(max_wait, wait_time)
            waiting_times[vehicle_id] = math.ceil(max_wait)
    return waiting_times


def detect_conflicts(vehicles):
    """
    Detects conflicts between vehicles approaching an intersection.

    Args:
        vehicles (list of Vehicle): List of Vehicle objects.

    Returns:
        list of dict: List of conflicts detected. Each conflict is a dictionary containing:
            - 'vehicle1_id': ID of the first vehicle.
            - 'vehicle2_id': ID of the second vehicle.
            - 'decision': Conflict decision message.
            - 'place': Place of the conflict ('intersection').
            - 'priority_order': Dictionary of vehicle IDs to their priority levels.
            - 'waiting_times': Dictionary of vehicle IDs to their waiting times.
    """
    conflicts = []
    n = len(vehicles)
    for i in range(n):
        for j in range(i + 1, n):
            vehicle1 = vehicles[i]
            vehicle2 = vehicles[j]
            if log:
                print(f"\nEvaluating vehicles {vehicle1.vehicle_id} and {vehicle2.vehicle_id}")
            if paths_cross(vehicle1, vehicle2):
                if arrival_time_close(vehicle1, vehicle2):
                    decision, priority = apply_priority_rules(vehicle1, vehicle2)
                    waiting_times = compute_waiting_times([vehicle1, vehicle2], priority)
                    conflicts.append({
                        'vehicle1_id': vehicle1.vehicle_id,
                        'vehicle2_id': vehicle2.vehicle_id,
                        'decision': decision,
                        'place': 'intersection',
                        'priority_order': priority,
                        'waiting_times': waiting_times
                    })
                else:
                    if log:
                        print("Vehicles do not arrive close in time; no conflict.")
            else:
                if log:
                    print("Vehicles paths do not cross; no conflict.")

    return conflicts


def output_conflicts(conflicts):
    """
    Outputs the conflicts detected.

    Args:
        conflicts (list of dict): List of conflict dictionaries.
    """
    for conflict in conflicts:
        if log:
            print(conflict['decision'])
        pass
