import json
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import argparse
import os

@dataclass
class CommentaryEvent:
    timestamp: float
    duration: float
    event_type: str
    priority: int
    data: str

class TimestampFilter:
    def __init__(self, config: Dict = None):
        self.config = {
            'elimination_duration': 10,
            'player_count_duration': 10,
            'storm_duration': 10,
            'joke_duration': 10,
            'min_gap': 2,
            'min_joke_gap': 15.0,
            **(config or {})
        }
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _load_json(self, file_path: str) -> any:
        """Load JSON file safely with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in file: {file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            return []

    def _save_json(self, data: any, file_path: str) -> bool:
        """Save JSON file safely with error handling"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Successfully saved {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save {file_path}: {e}")
            return False

    def _check_overlap(self, time1: float, duration1: float, time2: float, duration2: float) -> bool:
        """Check if two time periods overlap including minimum gap"""
        end1 = time1 + duration1 + self.config['min_gap']
        end2 = time2 + duration2 + self.config['min_gap']
        return not (end1 <= time2 or end2 <= time1)

    def _create_events_list(self, eliminations_file: str, storm_file: str, 
                          player_count_file: str) -> List[CommentaryEvent]:
        """Create a list of all commentary events with their priorities"""
        events = []
        
        # Load all data with validation
        eliminations = self._load_json(eliminations_file)
        if not eliminations:
            self.logger.warning("No elimination events found or invalid file")
            
        storm_times = self._load_json(storm_file)
        if not storm_times:
            self.logger.warning("No storm events found or invalid file")
            
        player_data = self._load_json(player_count_file)
        if not player_data:
            self.logger.warning("No player count data found or invalid file")

        # Add elimination events (highest priority)
        for timestamp in eliminations:
            events.append(CommentaryEvent(
                timestamp=float(timestamp),
                duration=self.config['elimination_duration'],
                event_type='elimination',
                priority=1,
                data=''
            ))

        # Add player count events
        for timestamp, count in player_data.items():
            events.append(CommentaryEvent(
                timestamp=float(timestamp),
                duration=self.config['player_count_duration'],
                event_type='player_count',
                priority=2,
                data=str(count)
            ))

        # Add storm events
        for timestamp in storm_times:
            events.append(CommentaryEvent(
                timestamp=float(timestamp),
                duration=self.config['storm_duration'],
                event_type='storm',
                priority=3,
                data=''
            ))

        # Sort by timestamp and then by priority
        return sorted(events, key=lambda x: (x.timestamp, x.priority))

    def _find_joke_opportunities(self, events: List[CommentaryEvent]) -> List[float]:
        """Find valid timestamps where jokes can be inserted"""
        if not events:
            return []

        opportunities = []
        
        # Check gap at the start of the video
        if events[0].timestamp > self.config['min_joke_gap']:
            opportunities.append(events[0].timestamp / 2)

        # Check gaps between events
        for i in range(len(events) - 1):
            current_end = events[i].timestamp + events[i].duration
            next_start = events[i + 1].timestamp
            gap = next_start - current_end
            
            if gap >= self.config['min_joke_gap']:
                opportunities.append(current_end + (gap / 2))

        return opportunities

    def filter_timestamps(self, eliminations_file: str, storm_file: str, 
                        player_count_file: str, output_dir: str = "filtered_output"):
        """Filter timestamps to prevent overlapping commentary while respecting priorities"""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Get all events
        self.logger.info("Loading and processing events...")
        events = self._create_events_list(eliminations_file, storm_file, player_count_file)
        
        if not events:
            self.logger.warning("No events found to process!")
            return {}

        # Filter out overlapping events based on priority
        filtered_events = []
        for event in events:
            # Check for overlap with already filtered events
            overlap = False
            for filtered_event in filtered_events:
                if self._check_overlap(
                    filtered_event.timestamp, 
                    filtered_event.duration,
                    event.timestamp, 
                    event.duration
                ):
                    # If current event has higher priority, replace the filtered event
                    if event.priority < filtered_event.priority:
                        filtered_events.remove(filtered_event)
                    else:
                        overlap = True
                    break
            
            if not overlap:
                filtered_events.append(event)

        self.logger.info(f"Filtered {len(events) - len(filtered_events)} overlapping events")

        # Find opportunities for jokes
        joke_timestamps = self._find_joke_opportunities(filtered_events)
        self.logger.info(f"Found {len(joke_timestamps)} opportunities for jokes")

        # Prepare filtered data for output
        filtered_data = {
            'eliminations': [],
            'storm': [],
            'player_count': {},
            'jokes': []
        }

        # Organize filtered events by type
        for event in filtered_events:
            if event.event_type == 'elimination':
                filtered_data['eliminations'].append(event.timestamp)
            elif event.event_type == 'storm':
                filtered_data['storm'].append(event.timestamp)
            elif event.event_type == 'player_count':
                filtered_data['player_count'][str(int(event.timestamp))] = event.data

        filtered_data['jokes'] = joke_timestamps

        # Save filtered data
        success = True
        for key, data in filtered_data.items():
            if not self._save_json(data, output_path / f"filtered_{key}.json"):
                success = False

        if success:
            self.logger.info("Successfully filtered and saved all event data")
            return filtered_data
        else:
            self.logger.error("Some errors occurred while saving filtered data")
            return {}

def main():
    parser = argparse.ArgumentParser(description='Filter timestamps for video commentary')
    parser.add_argument('--input-dir', default='.', help='Directory containing detection JSON files')
    parser.add_argument('--output-dir', default='filtered_output', help='Output directory for filtered files')
    args = parser.parse_args()

    try:
        filter = TimestampFilter()
        filter.filter_timestamps(
            eliminations_file=os.path.join(args.input_dir, 'eliminations.json'),
            storm_file=os.path.join(args.input_dir, 'storm_shrinking.json'),
            player_count_file=os.path.join(args.input_dir, 'player_count_data.json'),
            output_dir=args.output_dir
        )
        return 0
    except Exception as e:
        logging.error(f"Error filtering timestamps: {e}")
        return 1

if __name__ == "__main__":
    exit(main())