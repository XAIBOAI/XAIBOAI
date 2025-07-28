import cv2
import easyocr
from difflib import SequenceMatcher
import os
import logging
import json
import matplotlib.pyplot as plt
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class cs2commentor:
    def __init__(self, language: str = 'en', similarity_threshold: float = 0.3):
        """
        Initialize the CS2 commentator with OCR and similarity threshold.
        """
        self.reader = easyocr.Reader([language], gpu=True)
        self.similarity_threshold = similarity_threshold

        # Define region of interest (ROI) for OCR
        self.elimination_roi = {"top": 714,
        "bottom": 786,
        "left": 660,
        "right": 1132}
        self.molotof_roi = {"top": 337,
        "bottom": 369,
        "left": 1606,
        "right": 1638}  # Example ROI for "MOLOTOF"
        self.flash_roi = {"top": 334,
        "bottom": 367,
        "left": 1719,
        "right": 1760}  # ROI for flash bang
        self.camping_roi = {"top": 334,
        "bottom": 653,
        "left": 1233,
        "right": 1240}  # ROI for camping

    def validate_frame(self, frame) -> bool:
        """ Validate if frame is good for processing. """
        return frame is not None and frame.size > 0 and frame.shape[0] > max(self.elimination_roi['bottom'], self.storm_roi['bottom']) and frame.shape[1] > max(self.elimination_roi['right'], self.storm_roi['right'])

    def detect_eliminated(self, frame, timestamp: int) -> bool:
        """ Detect if 'ELIMINATED' appears in the frame. """
        cropped_img = frame[self.elimination_roi['top']:self.elimination_roi['bottom'], self.elimination_roi['left']:self.elimination_roi['right']]
        ocr_result = self.reader.readtext(cropped_img)
        for _, text, _ in ocr_result:
            similarity = SequenceMatcher(None, "eliminated", text.lower()).ratio()
            if similarity > self.similarity_threshold:
                # Log and print when 'eliminated' is detected
                logger.info(f"'Eliminated' detected at timestamp: {timestamp} seconds")
                print(f"'Eliminated' detected at timestamp: {timestamp} seconds")
                return True
        return False

    def detect_MOLOTOF_FIRE (self, frame, timestamp: int) -> bool:
        """ Detect if molotof fire appears in the frame using orange color detection. """
        cropped_img = frame[self.storm_roi['top']:self.storm_roi['bottom'],
                      self.storm_roi['left']:self.storm_roi['right']]

        # Convert the ROI to HSV color space
        hsv_frame = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        # Define the HSV range for the orange color (#E725E9)
        # Convert the hex color #E725E9 to HSV
        target_color = np.uint8([[[231, 37, 233]]])  # RGB color for #E725E9
        hsv_target_color = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)[0][0]

        # Define the range for the orange color
        lower_orange = np.array([hsv_target_color[0] - 10, 100, 100])  # Lower bound for hue, adjust the range if needed
        upper_orange = np.array([hsv_target_color[0] + 10, 255, 255])  # Upper bound for hue

        # Create a mask that detects the orange color in the molotof ROI
        mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

        # Check if there are any porange pixels in the mask
        if np.any(mask > 0):
            # Log and print when molotof fire is detected
            logger.info(f"molotof fire warning detected at timestamp: {timestamp} seconds")
            print(f"molotof fire warning detected at timestamp: {timestamp} seconds")
            return True
        return False

    def analyze_molotof_fire(self, fire_timestamps: list) -> list:
        """
        Analyze the detected molotof fire timestamps and filter out isolated false positives.
        If timestamps occur in sequence (e.g., 173, 174, 175, ...), we consider it valid molotof.
        """
        if not molotof_timestamps:
            return []

        valid_molotof_timestamps = []
        sequence_start = None  # Variable to hold the start of a valid sequence

        for i in range(1, len(fire_timestamps)):
            if fire_timestamps[i] == fire_timestamps[i - 1] + 1:
                if sequence_start is None:
                    sequence_start = fire_timestamps[i - 1]  # Mark the start of a valid sequence

            else:
                # If the sequence breaks and we had a valid sequence
                if sequence_start is not None:
                    # Calculate when to mark "molotof fire" (5 seconds after sequence starts)
                    molotof_fire_start = sequence_start + 5
                    valid_molotof_timestamps.append(molotof_fire_start)
                    sequence_start = None  # Reset sequence_start for the next sequence

        # Handle the case where the last sequence reaches the end of the timestamps
        if sequence_start is not None:
            molotof_fire_start = sequence_start + 5
            valid_molotof_timestamps.append(molotof_fire_start)

        return valid_molotof_timestamps
    
    def detect_FLASH_BANG(self, frame, timestamp: int) -> bool:
    """ Detect if a flash bang appears in the frame using bright white color detection. """
    cropped_img = frame[self.storm_roi['top']:self.storm_roi['bottom'],
                        self.storm_roi['left']:self.storm_roi['right']]

    # Convert the ROI to HSV color space
    hsv_frame = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    # Define the HSV range for bright white flashbang effect
    # Flashbang is often pure bright white (#FFFFFF)
    lower_white = np.array([0, 0, 200])   # Lower bound for bright white
    upper_white = np.array([180, 30, 255])  # Upper bound for bright white

    # Create a mask that detects the white flash in the flashbang ROI
    mask = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Check if there are any white pixels in the mask
    if np.any(mask > 0):
        # Log and print when flashbang is detected
        logger.info(f"Flash bang detected at timestamp: {timestamp} seconds")
        print(f"Flash bang detected at timestamp: {timestamp} seconds")
        return True
    return False

def analyze_flash_bang(self, flash_timestamps: list) -> list:
    """
    Analyze the detected flash bang timestamps and filter out isolated false positives.
    If timestamps occur in sequence (e.g., 173, 174, 175, ...), we consider it a valid flash bang.
    """
    if not flash_timestamps:
        return []

    valid_flash_timestamps = []
    sequence_start = None  # Variable to hold the start of a valid sequence

    for i in range(1, len(flash_timestamps)):
        if flash_timestamps[i] == flash_timestamps[i - 1] + 1:
            if sequence_start is None:
                sequence_start = flash_timestamps[i - 1]  # Start of sequence
        else:
            if sequence_start is not None:
                flash_bang_time = sequence_start + 5
                valid_flash_timestamps.append(flash_bang_time)
                sequence_start = None

    if sequence_start is not None:
        flash_bang_time = sequence_start + 5
        valid_flash_timestamps.append(flash_bang_time)

    return valid_flash_timestamps


    def save_player_count_data(self, player_count_data: dict):
        """ Save the player count data to a JSON file. """
        with open('player_count_data.json', 'w') as json_file:
            json.dump(player_count_data, json_file, indent=4)
        logger.info("Player count data saved to player_count_data.json")

    def process_video(self, video_path: str, sample_rate: int = 1) -> (list, list):
        """ Process video to detect eliminations and storm shrinking warnings. """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return [], []

        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
        frame_interval = int(fps)  # Process one frame per second (1 frame per fps)
        eliminated_timestamps = []
        molotof_fire_timestamps = []
        frame_count = 0
        molotof_fire_sequences = []  # Store the detected timestamps for molotof fire
        player_count_data = {}

        # Set up the Matplotlib figure
        plt.ion()  # Enable interactive mode for continuous frame update
        fig, ax = plt.subplots()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate the timestamp in seconds for each frame
            timestamp = int(frame_count / fps)  # Ensure timestamp is always defined before usage

                          # Detect eliminations
                if self.validate_frame(frame) and self.detect_eliminated(frame, int(frame_count / fps)):
                    timestamp = int(frame_count / fps)
                    eliminated_timestamps.append(timestamp)

                # Detect storm shrinking warnings
                if self.validate_frame(frame) and self.detect_molotof_fire(frame, int(frame_count / fps)):
                    timestamp = int(frame_count / fps)
                    molotof_fire_timestamps.append(timestamp)

                # Convert BGR (OpenCV default) to RGB for Matplotlib
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame with ROIs using Matplotlib
                ax.clear()
                ax.imshow(rgb_frame)
                ax.set_title('Frame with ROIs')
                ax.axis('off')  # Turn off axis numbers/labels
                plt.pause(0.001)  # Short pause to create a video playback effect

            frame_count += 1

        cap.release()
        plt.ioff()  # Turn off interactive mode after the video ends
        plt.show()  # Keep the last frame open

        # Analyze molotof fire sequences
        valid_molotof_fire_timestamps = self.analyze_molotof_fire(molotof_fire_timestamps)

        # Save the player count data to a JSON file
        self.save_player_count_data(player_count_data)

        return eliminated_timestamps, valid_molotof_fire_timestamps


def main():
    # Example usage
    import socket

server = 'irc.chat.twitch.tv'
port = 6667
nickname = 'xaibo_ai'
token = 'oauth:y*************'  
channel = '#xaibo'

sock = socket.socket()
sock.connect((server, port))
sock.send(f"PASS {****}\n".encode('utf-8'))
sock.send(f"NICK {****}\n".encode('utf-8'))
sock.send(f"JOIN {****}\n".encode('utf-8'))

print(f"Connected to {channel}")

while True:
    resp = sock.recv(2048).decode('utf-8')
    if resp.startswith('PING'):
        sock.send("PONG :tmi.twitch.tv\r\n".encode('utf-8'))
    elif len(resp) > 0:
        print(resp)


    # Save eliminations to JSON
    if eliminated_timestamps:
        logger.info(f"Detected eliminations at: {eliminated_timestamps} seconds")
        with open('eliminations.json', 'w') as f:
            json.dump(eliminated_timestamps, f)
    else:
        logger.info("No eliminations detected in stream")

    # Save molotof fire warnings to JSON
    if molotof_fire_timestamps:
        logger.info(f"Detected storm shrinking warnings at: {molotof_fire_timestamps} seconds")
        with open('molotof_fire.json', 'w') as f:
            json.dump(molotof_fire_timestamps, f)
    else:
        logger.info("No molotof_fire warnings detected in stream")

if __name__ == "__main__":
    main()
