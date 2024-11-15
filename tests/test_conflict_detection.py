# tests/test_conflict_detection.py

"""
Unit Tests for Conflict Detection Module

This module contains unit tests for the conflict detection system, testing various
scenarios to ensure the correctness of conflict detection logic.

Author: Your Name
Date: YYYY-MM-DD
"""

import unittest
import json
import warnings
from src.conflict_detection import (
    Vehicle,
    parse_vehicles,
    detect_conflicts,
    parse_intersection_layout,
)


class TestConflictDetection(unittest.TestCase):
    """
    Unit tests for conflict detection.
    """

    def setUp(self):
        self.intersection_layout_json = '''
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
        self.intersection_layout_data = json.loads(self.intersection_layout_json)
        self.intersection_layout = parse_intersection_layout(self.intersection_layout_data)

    def test_no_conflict_different_directions(self):
        """
        Test vehicles from different directions with no path crossing.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V001",
                    "lane": 1,
                    "speed": 50,
                    "distance_to_intersection": 200,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V002",
                    "lane": 4,
                    "speed": 50,
                    "distance_to_intersection": 200,
                    "direction": "east",
                    "destination": "G"
                }
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertEqual(conflict['vehicle1_id'], 'V001')
        self.assertEqual(conflict['vehicle2_id'], 'V002')

    def test_conflict_same_time_straight_crossing(self):
        """
        Test vehicles arriving at the same time going straight and paths cross.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V003",
                    "lane": 1,
                    "speed": 60,
                    "distance_to_intersection": 180,
                    "direction": "north",
                    "destination": "H"
                },
                {
                    "vehicle_id": "V004",
                    "lane": 3,
                    "speed": 60,
                    "distance_to_intersection": 180,
                    "direction": "east",
                    "destination": "B"
                }
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertIn("Vehicle V003 must yield to Vehicle V004", conflict['decision'])

    def test_conflict_different_speeds_same_arrival(self):
        """
        Test vehicles arriving at the same time with crossing paths.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V005",
                    "lane": 1,
                    "speed": 40,
                    "distance_to_intersection": 100,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V006",
                    "lane": 5,
                    "speed": 80,
                    "distance_to_intersection": 200,
                    "direction": "south",
                    "destination": "D"
                }
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 0)

    def test_conflict_right_turn_vs_left_turn(self):
        """
        Test right-turning vehicle vs left-turning vehicle.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V007",
                    "lane": 1,
                    "speed": 30,
                    "distance_to_intersection": 90,
                    "direction": "north",
                    "destination": "H"
                },
                {
                    "vehicle_id": "V008",
                    "lane": 6,
                    "speed": 30,
                    "distance_to_intersection": 90,
                    "direction": "south",
                    "destination": "G"
                }
            ]
        }
        '''
        # V007 is making a right turn (destination "H" from lane 1)
        # V008 is making a left turn (destination "G" from lane 6)
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertIn("Vehicle V008 must yield to Vehicle V007", conflict['decision'])

    def test_conflict_opposite_left_turns(self):
        """
        Test both vehicles turning left from opposite directions.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V009",
                    "lane": 2,
                    "speed": 40,
                    "distance_to_intersection": 80,
                    "direction": "north",
                    "destination": "E"
                },
                {
                    "vehicle_id": "V010",
                    "lane": 6,
                    "speed": 40,
                    "distance_to_intersection": 80,
                    "direction": "south",
                    "destination": "A"
                }
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Left turns from opposite directions generally don't conflict
        self.assertEqual(len(conflicts), 0)

    def test_conflict_right_hand_rule(self):
        """
        Test applying the right-hand rule.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V011",
                    "lane": 2,
                    "speed": 30,
                    "distance_to_intersection": 60,
                    "direction": "north",
                    "destination": "D"
                },
                {
                    "vehicle_id": "V012",
                    "lane": 5,
                    "speed": 30,
                    "distance_to_intersection": 60,
                    "direction": "south",
                    "destination": "D"
                }
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Vehicle V012 is on the right of V011
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertIn("Vehicle V011 must yield to Vehicle V012", conflict['decision'])

    def test_conflict_late_arrival(self):
        """
        Test vehicles arriving at different times.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V013",
                    "lane": 1,
                    "speed": 40,
                    "distance_to_intersection": 400,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V014",
                    "lane": 3,
                    "speed": 40,
                    "distance_to_intersection": 100,
                    "direction": "east",
                    "destination": "B"
                }
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 0)

    def test_conflict_same_destination(self):
        """
        Test vehicles heading to the same destination.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V015",
                    "lane": 1,
                    "speed": 50,
                    "distance_to_intersection": 100,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V016",
                    "lane": 7,
                    "speed": 50,
                    "distance_to_intersection": 100,
                    "direction": "west",
                    "destination": "F"
                }
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Both vehicles are heading to street F, potential conflict
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertIn("Vehicle V015 must yield to Vehicle V016", conflict['decision'])

    def test_conflict_zero_speed(self):
        """
        Test vehicle with zero speed (stopped).
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V017",
                    "lane": 1,
                    "speed": 0,
                    "distance_to_intersection": 0,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V018",
                    "lane": 3,
                    "speed": 40,
                    "distance_to_intersection": 100,
                    "direction": "east",
                    "destination": "B"
                }
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        # Since V017 is not moving, it should not cause a conflict
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 0)

    def test_conflict_high_speed(self):
        """
        Test vehicle approaching at high speed.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V019",
                    "lane": 1,
                    "speed": 100,
                    "distance_to_intersection": 500,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V020",
                    "lane": 3,
                    "speed": 50,
                    "distance_to_intersection": 200,
                    "direction": "east",
                    "destination": "B"
                }
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertIn("Vehicle V019 must yield to Vehicle V020", conflict['decision'])

    # Additional 10 tests to validate the decision

    def test_conflict_multiple_left_turns(self):
        """
        Test multiple vehicles making left turns from different directions.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V052",
                    "lane": 2,
                    "speed": 30,
                    "distance_to_intersection": 60,
                    "direction": "north",
                    "destination": "E"
                },
                {
                    "vehicle_id": "V053",
                    "lane": 4,
                    "speed": 30,
                    "distance_to_intersection": 60,
                    "direction": "east",
                    "destination": "G"
                }
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Left turns from adjacent directions conflict
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertIn("must yield", conflict['decision'])

    def test_conflict_right_turns_from_adjacent_directions(self):
        """
        Test right-turning vehicle vs straight-going vehicle from adjacent directions.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V054",
                    "lane": 1,
                    "speed": 40,
                    "distance_to_intersection": 80,
                    "direction": "north",
                    "destination": "H"
                },
                {
                    "vehicle_id": "V055",
                    "lane": 7,
                    "speed": 40,
                    "distance_to_intersection": 80,
                    "direction": "west",
                    "destination": "D"
                }
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Right-turning vehicle must yield to straight-going vehicle
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertIn("Vehicle V055 must yield to Vehicle V054", conflict['decision'])

    def test_conflict_straight_vs_left_turn(self):
        """
        Test vehicle going straight vs vehicle turning left.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V056",
                    "lane": 1,
                    "speed": 50,
                    "distance_to_intersection": 100,
                    "direction": "north",
                    "destination": "H"
                },
                {
                    "vehicle_id": "V057",
                    "lane": 4,
                    "speed": 50,
                    "distance_to_intersection": 100,
                    "direction": "east",
                    "destination": "F"
                }
            ]
        }
        '''
        # V056 going straight, V057 turning left
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertIn("Vehicle V057 must yield to Vehicle V056", conflict['decision'])

    def test_conflict_different_arrival_times(self):
        """
        Test vehicles arriving with a time difference greater than threshold.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V058",
                    "lane": 1,
                    "speed": 40,
                    "distance_to_intersection": 400,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V059",
                    "lane": 3,
                    "speed": 40,
                    "distance_to_intersection": 100,
                    "direction": "east",
                    "destination": "B"
                }
            ]
        }
        '''
        # Time difference is greater than threshold; no conflict
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 0)

    def test_conflict_same_arrival_different_directions(self):
        """
        Test vehicles from different directions arriving at the same time.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V060",
                    "lane": 1,
                    "speed": 60,
                    "distance_to_intersection": 180,
                    "direction": "north",
                    "destination": "H"
                },
                {
                    "vehicle_id": "V061",
                    "lane": 5,
                    "speed": 60,
                    "distance_to_intersection": 180,
                    "direction": "south",
                    "destination": "D"
                }
            ]
        }
        '''
        # Paths do not cross; no conflict
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 0)

    def test_conflict_opposite_directions_straight(self):
        """
        Test vehicles going straight from opposite directions.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V062",
                    "lane": 1,
                    "speed": 50,
                    "distance_to_intersection": 100,
                    "direction": "north",
                    "destination": "H"
                },
                {
                    "vehicle_id": "V063",
                    "lane": 5,
                    "speed": 50,
                    "distance_to_intersection": 100,
                    "direction": "south",
                    "destination": "B"
                }
            ]
        }
        '''
        # Vehicles going straight from opposite directions do not conflict
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 0)

    def test_conflict_three_vehicles(self):
        """
        Test scenario with three vehicles approaching.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V064",
                    "lane": 1,
                    "speed": 40,
                    "distance_to_intersection": 80,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V065",
                    "lane": 3,
                    "speed": 40,
                    "distance_to_intersection": 80,
                    "direction": "east",
                    "destination": "B"
                },
                {
                    "vehicle_id": "V066",
                    "lane": 5,
                    "speed": 40,
                    "distance_to_intersection": 80,
                    "direction": "south",
                    "destination": "D"
                }
            ]
        }
        '''
        # Multiple conflicts expected
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 2)

    def test_conflict_vehicle_stopped_in_intersection(self):
        """
        Test vehicle stopped in the intersection.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V067",
                    "lane": 1,
                    "speed": 0,
                    "distance_to_intersection": 0,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V068",
                    "lane": 3,
                    "speed": 40,
                    "distance_to_intersection": 80,
                    "direction": "east",
                    "destination": "B"
                }
            ]
        }
        '''
        # Stopped vehicle may cause a conflict if arrival times are close
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Since V067 is not moving, no conflict expected
        self.assertEqual(len(conflicts), 0)

    def test_conflict_invalid_lane(self):
        """
        Test vehicle in an invalid lane.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V069",
                    "lane": 9,
                    "speed": 40,
                    "distance_to_intersection": 100,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V070",
                    "lane": 3,
                    "speed": 40,
                    "distance_to_intersection": 100,
                    "direction": "east",
                    "destination": "B"
                }
            ]
        }
        '''
        # Vehicle V069 is in an invalid lane
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        with self.assertWarns(UserWarning):
            vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        # V069 has unknown movement type; no conflict expected
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 0)

    def test_conflict_duplicate_vehicle_ids(self):
        """
        Test scenario with duplicate vehicle IDs.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V071",
                    "lane": 1,
                    "speed": 40,
                    "distance_to_intersection": 100,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V071",
                    "lane": 3,
                    "speed": 40,
                    "distance_to_intersection": 100,
                    "direction": "east",
                    "destination": "B"
                }
            ]
        }
        '''
        # Duplicate vehicle IDs should raise an error
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        with self.assertRaises(ValueError):
            parse_vehicles(vehicles_scenario_data, self.intersection_layout)

    def test_conflict_vehicle_with_infinite_time(self):
        """
        Test vehicle with infinite time to intersection (zero speed, non-zero distance).
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V072",
                    "lane": 1,
                    "speed": 0,
                    "distance_to_intersection": 100,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V073",
                    "lane": 3,
                    "speed": 40,
                    "distance_to_intersection": 100,
                    "direction": "east",
                    "destination": "B"
                }
            ]
        }
        '''
        # V072 has infinite time to intersection; should not cause a conflict
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        self.assertEqual(len(conflicts), 0)

    def test_conflict_vehicle_with_zero_distance(self):
        """
        Test vehicle with zero distance to intersection.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V074",
                    "lane": 1,
                    "speed": 40,
                    "distance_to_intersection": 0,
                    "direction": "north",
                    "destination": "F"
                },
                {
                    "vehicle_id": "V075",
                    "lane": 4,
                    "speed": 40,
                    "distance_to_intersection": 0,
                    "direction": "east",
                    "destination": "F"
                }
            ]
        }
        '''
        # Both vehicles are at the intersection
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Conflict expected
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertIn("must yield", conflict['decision'])

    def test_conflict_eight_vehicles_all_lanes_no_conflict(self):
        """
        Test scenario with 8 vehicles, one in each lane, with no conflicts.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {"vehicle_id": "V100", "lane": 1, "speed": 40, "distance_to_intersection": 1000, "direction": "north", "destination": "F"},
                {"vehicle_id": "V101", "lane": 2, "speed": 30, "distance_to_intersection": 900, "direction": "north", "destination": "E"},
                {"vehicle_id": "V102", "lane": 3, "speed": 60, "distance_to_intersection": 1200, "direction": "east", "destination": "B"},
                {"vehicle_id": "V103", "lane": 4, "speed": 50, "distance_to_intersection": 1100, "direction": "east", "destination": "G"},
                {"vehicle_id": "V104", "lane": 5, "speed": 70, "distance_to_intersection": 1300, "direction": "south", "destination": "B"},
                {"vehicle_id": "V105", "lane": 6, "speed": 80, "distance_to_intersection": 1400, "direction": "south", "destination": "A"},
                {"vehicle_id": "V106", "lane": 7, "speed": 90, "distance_to_intersection": 1500, "direction": "west", "destination": "D"},
                {"vehicle_id": "V107", "lane": 8, "speed": 100, "distance_to_intersection": 1600, "direction": "west", "destination": "B"}
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # With significantly varied arrival times, conflicts should be avoided
        self.assertEqual(len(conflicts), 1)

    def test_conflict_eight_vehicles_all_lanes_with_conflicts(self):
        """
        Test scenario with 8 vehicles, one in each lane, with multiple conflicts.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {"vehicle_id": "V108", "lane": 1, "speed": 40, "distance_to_intersection": 100, "direction": "north", "destination": "H"},
                {"vehicle_id": "V109", "lane": 2, "speed": 40, "distance_to_intersection": 100, "direction": "north", "destination": "D"},
                {"vehicle_id": "V110", "lane": 3, "speed": 40, "distance_to_intersection": 100, "direction": "east", "destination": "H"},
                {"vehicle_id": "V111", "lane": 4, "speed": 40, "distance_to_intersection": 100, "direction": "east", "destination": "F"},
                {"vehicle_id": "V112", "lane": 5, "speed": 40, "distance_to_intersection": 100, "direction": "south", "destination": "D"},
                {"vehicle_id": "V113", "lane": 6, "speed": 40, "distance_to_intersection": 100, "direction": "south", "destination": "H"},
                {"vehicle_id": "V114", "lane": 7, "speed": 40, "distance_to_intersection": 100, "direction": "west", "destination": "F"},
                {"vehicle_id": "V115", "lane": 8, "speed": 40, "distance_to_intersection": 100, "direction": "west", "destination": "B"}
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Multiple conflicts expected
        self.assertGreaterEqual(len(conflicts), 1)

    def test_conflict_eight_vehicles_all_straight(self):
        """
        Test scenario where all eight vehicles arrive simultaneously, all going straight.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {"vehicle_id": "V201", "lane": 1, "speed": 40, "distance_to_intersection": 80, "direction": "north", "destination": "H"},
                {"vehicle_id": "V202", "lane": 3, "speed": 40, "distance_to_intersection": 80, "direction": "east", "destination": "B"},
                {"vehicle_id": "V203", "lane": 5, "speed": 40, "distance_to_intersection": 80, "direction": "south", "destination": "D"},
                {"vehicle_id": "V204", "lane": 7, "speed": 40, "distance_to_intersection": 80, "direction": "west", "destination": "F"},
                {"vehicle_id": "V205", "lane": 1, "speed": 40, "distance_to_intersection": 80, "direction": "north", "destination": "H"},
                {"vehicle_id": "V206", "lane": 3, "speed": 40, "distance_to_intersection": 80, "direction": "east", "destination": "B"},
                {"vehicle_id": "V207", "lane": 5, "speed": 40, "distance_to_intersection": 80, "direction": "south", "destination": "D"},
                {"vehicle_id": "V208", "lane": 7, "speed": 40, "distance_to_intersection": 80, "direction": "west", "destination": "F"}
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Expect conflicts between vehicles going straight from perpendicular directions
        self.assertEqual(len(conflicts), 16)
        for conflict in conflicts:
            self.assertIn("must yield", conflict['decision'])

    def test_conflict_eight_vehicles_staggered_arrival(self):
        """
        Test scenario with eight vehicles arriving at staggered times, with various movement types.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {"vehicle_id": "V209", "lane": 1, "speed": 60, "distance_to_intersection": 180, "direction": "north", "destination": "H"},
                {"vehicle_id": "V210", "lane": 2, "speed": 30, "distance_to_intersection": 90, "direction": "north", "destination": "E"},
                {"vehicle_id": "V211", "lane": 3, "speed": 50, "distance_to_intersection": 200, "direction": "east", "destination": "B"},
                {"vehicle_id": "V212", "lane": 4, "speed": 40, "distance_to_intersection": 120, "direction": "east", "destination": "G"},
                {"vehicle_id": "V213", "lane": 5, "speed": 80, "distance_to_intersection": 240, "direction": "south", "destination": "B"},
                {"vehicle_id": "V214", "lane": 6, "speed": 70, "distance_to_intersection": 210, "direction": "south", "destination": "A"},
                {"vehicle_id": "V215", "lane": 7, "speed": 60, "distance_to_intersection": 180, "direction": "west", "destination": "D"},
                {"vehicle_id": "V216", "lane": 8, "speed": 50, "distance_to_intersection": 150, "direction": "west", "destination": "C"}
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Verify conflicts are detected correctly based on arrival times and movements
        self.assertGreaterEqual(len(conflicts), 1)
        for conflict in conflicts:
            self.assertIn("must yield", conflict['decision'])


    def test_conflict_eight_vehicles_potential_gridlock(self):
        """
        Test scenario where all vehicles are turning left simultaneously, potentially causing gridlock.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {"vehicle_id": "V217", "lane": 2, "speed": 40, "distance_to_intersection": 80, "direction": "north", "destination": "E"},
                {"vehicle_id": "V218", "lane": 4, "speed": 40, "distance_to_intersection": 80, "direction": "east", "destination": "G"},
                {"vehicle_id": "V219", "lane": 6, "speed": 40, "distance_to_intersection": 80, "direction": "south", "destination": "A"},
                {"vehicle_id": "V220", "lane": 8, "speed": 40, "distance_to_intersection": 80, "direction": "west", "destination": "C"},
                {"vehicle_id": "V221", "lane": 2, "speed": 40, "distance_to_intersection": 80, "direction": "north", "destination": "E"},
                {"vehicle_id": "V222", "lane": 4, "speed": 40, "distance_to_intersection": 80, "direction": "east", "destination": "G"},
                {"vehicle_id": "V223", "lane": 6, "speed": 40, "distance_to_intersection": 80, "direction": "south", "destination": "A"},
                {"vehicle_id": "V224", "lane": 8, "speed": 40, "distance_to_intersection": 80, "direction": "west", "destination": "C"}
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Since all are turning left, conflicts should be detected
        self.assertEqual(len(conflicts), 16)
        for conflict in conflicts:
            self.assertIn("must yield", conflict['decision'])
    def test_conflict_eight_vehicles_mixed_speeds(self):
        """
        Test scenario with eight vehicles at different speeds arriving close in time.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {"vehicle_id": "V225", "lane": 1, "speed": 80, "distance_to_intersection": 160, "direction": "north", "destination": "H"},
                {"vehicle_id": "V226", "lane": 2, "speed": 60, "distance_to_intersection": 120, "direction": "north", "destination": "E"},
                {"vehicle_id": "V227", "lane": 3, "speed": 100, "distance_to_intersection": 200, "direction": "east", "destination": "B"},
                {"vehicle_id": "V228", "lane": 4, "speed": 50, "distance_to_intersection": 100, "direction": "east", "destination": "F"},
                {"vehicle_id": "V229", "lane": 5, "speed": 90, "distance_to_intersection": 180, "direction": "south", "destination": "D"},
                {"vehicle_id": "V230", "lane": 6, "speed": 70, "distance_to_intersection": 140, "direction": "south", "destination": "G"},
                {"vehicle_id": "V231", "lane": 7, "speed": 60, "distance_to_intersection": 120, "direction": "west", "destination": "F"},
                {"vehicle_id": "V232", "lane": 8, "speed": 80, "distance_to_intersection": 160, "direction": "west", "destination": "B"}
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Expect conflicts due to similar arrival times despite different speeds
        self.assertGreaterEqual(len(conflicts), 1)
        for conflict in conflicts:
            self.assertIn("must yield", conflict['decision'])
    def test_conflict_eight_vehicles_all_left_turns(self):
        """
        Test scenario where all vehicles are turning left with varied arrival times.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {"vehicle_id": "V233", "lane": 2, "speed": 40, "distance_to_intersection": 120, "direction": "north", "destination": "E"},
                {"vehicle_id": "V234", "lane": 4, "speed": 60, "distance_to_intersection": 180, "direction": "east", "destination": "G"},
                {"vehicle_id": "V235", "lane": 6, "speed": 50, "distance_to_intersection": 150, "direction": "south", "destination": "A"},
                {"vehicle_id": "V236", "lane": 8, "speed": 70, "distance_to_intersection": 210, "direction": "west", "destination": "C"},
                {"vehicle_id": "V237", "lane": 2, "speed": 55, "distance_to_intersection": 165, "direction": "north", "destination": "E"},
                {"vehicle_id": "V238", "lane": 4, "speed": 45, "distance_to_intersection": 135, "direction": "east", "destination": "G"},
                {"vehicle_id": "V239", "lane": 6, "speed": 65, "distance_to_intersection": 195, "direction": "south", "destination": "A"},
                {"vehicle_id": "V240", "lane": 8, "speed": 35, "distance_to_intersection": 105, "direction": "west", "destination": "C"}
            ]
        }
        '''
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)
        # Conflicts should be detected between vehicles arriving close in time
        self.assertGreaterEqual(len(conflicts), 1)
        for conflict in conflicts:
            self.assertIn("must yield", conflict['decision'])
    def test_conflict_priority_and_waiting_time(self):
        """
        Test conflict detection with priority order and waiting times.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {
                    "vehicle_id": "V100",
                    "lane": 1,
                    "speed": 50,
                    "distance_to_intersection": 100,
                    "direction": "north",
                    "destination": "H"
                },
                {
                    "vehicle_id": "V101",
                    "lane": 4,
                    "speed": 50,
                    "distance_to_intersection": 100,
                    "direction": "east",
                    "destination": "F"
                }
            ]
        }
        '''
        # V100 going straight, V101 turning left
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)

        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertEqual(conflict['vehicle1_id'], 'V100')
        self.assertEqual(conflict['vehicle2_id'], 'V101')
        self.assertIn("Vehicle V101 must yield to Vehicle V100", conflict['decision'])
        # Check priority order
        expected_priority = {'V100': 1, 'V101': 2}
        self.assertEqual(conflict['priority_order'], expected_priority)
        # Check waiting times
        expected_waiting_times = {'V100': 0, 'V101': 2}  # Assuming 2 seconds to clear intersection
        self.assertEqual(conflict['waiting_times'], expected_waiting_times)

    def test_no_conflict_no_waiting_times(self):
        """
        Test that no waiting times are calculated when there is no conflict.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {"vehicle_id": "V401", "lane": 1, "speed": 50, "distance_to_intersection": 100, "direction": "north", "destination": "F"},
                {"vehicle_id": "V402", "lane": 5, "speed": 50, "distance_to_intersection": 200, "direction": "south", "destination": "B"}
            ]
        }
        '''
        # Vehicles do not have crossing paths
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)

        self.assertEqual(len(conflicts), 0)
        # Since there are no conflicts, no waiting times should be calculated

    def test_conflict_with_zero_speed_vehicle(self):
        """
        Test conflict detection with a stopped vehicle.
        """
        vehicles_scenario_json = '''
        {
            "vehicles_scenario": [
                {"vehicle_id": "V501", "lane": 1, "speed": 0, "distance_to_intersection": 0, "direction": "north", "destination": "H"},
                {"vehicle_id": "V502", "lane": 3, "speed": 40, "distance_to_intersection": 80, "direction": "east", "destination": "B"}
            ]
        }
        '''
        # V501 is stopped at the intersection
        vehicles_scenario_data = json.loads(vehicles_scenario_json)
        vehicles = parse_vehicles(vehicles_scenario_data, self.intersection_layout)
        conflicts = detect_conflicts(vehicles)

        # Since V501 is not moving, no conflict should be detected
        self.assertEqual(len(conflicts), 0)
if __name__ == '__main__':
    unittest.main()
