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

class FortniteCommentator:
    def __init__(self, language: str = 'en', similarity_threshold: float = 0.3):
        """
        Initialize the Fortnite commentator with OCR and similarity threshold.
        """
        self.reader = easyocr.Reader([language], gpu=True)
        self.similarity_threshold = similarity_threshold

        # Define region of interest (ROI) for OCR
        self.elimination_roi = {"top": 714,
        "bottom": 786,
        "left": 660,
        "right": 1132}
        self.storm_roi = {"top": 337,
        "bottom": 369,
        "left": 1606,
        "right": 1638}  # Example ROI for "Storm Shrinking"
        self.player_count_roi = {"top": 334,
        "bottom": 367,
        "left": 1719,
        "right": 1760}  # ROI for player count

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

    def detect_storm_shrinking(self, frame, timestamp: int) -> bool:
        """ Detect if storm shrinking warning appears in the frame using purple color detection. """
        cropped_img = frame[self.storm_roi['top']:self.storm_roi['bottom'],
                      self.storm_roi['left']:self.storm_roi['right']]

        # Convert the ROI to HSV color space
        hsv_frame = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        # Define the HSV range for the purple color (#E725E9)
        # Convert the hex color #E725E9 to HSV
        target_color = np.uint8([[[231, 37, 233]]])  # RGB color for #E725E9
        hsv_target_color = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)[0][0]

        # Define the range for the purple color
        lower_purple = np.array([hsv_target_color[0] - 10, 100, 100])  # Lower bound for hue, adjust the range if needed
        upper_purple = np.array([hsv_target_color[0] + 10, 255, 255])  # Upper bound for hue

        # Create a mask that detects the purple color in the storm shrinking ROI
        mask = cv2.inRange(hsv_frame, lower_purple, upper_purple)

        # Check if there are any purple pixels in the mask
        if np.any(mask > 0):
            # Log and print when storm shrinking is detected
            logger.info(f"Storm shrinking warning detected at timestamp: {timestamp} seconds")
            print(f"Storm shrinking warning detected at timestamp: {timestamp} seconds")
            return True
        return False

    def detect_player_count(self, frame) -> int:
        """ Detect the total player count using OCR in the specified ROI. """
        # Crop the frame to the player count ROI
        player_count_roi = frame[self.player_count_roi['top']:self.player_count_roi['bottom'],
                           self.player_count_roi['left']:self.player_count_roi['right']]

        # Convert the ROI to grayscale for better OCR accuracy (optional with easyocr, but sometimes helps)
        gray_roi = cv2.cvtColor(player_count_roi, cv2.COLOR_BGR2GRAY)

        # Use easyocr to extract text from the grayscale ROI
        ocr_result = self.reader.readtext(gray_roi)

        # Iterate through the OCR results to find a number
        for _, text, _ in ocr_result:
            # Extract only digits from the OCR result
            cleaned_text = ''.join(filter(str.isdigit, text))

            if cleaned_text.isdigit():
                return int(cleaned_text)

        # If no valid number is found, return None
        return None

    def analyze_storm_shrinking(self, shrinking_timestamps: list) -> list:
        """
        Analyze the detected storm shrinking timestamps and filter out isolated false positives.
        If timestamps occur in sequence (e.g., 173, 174, 175, ...), we consider it valid shrinking.
        """
        if not shrinking_timestamps:
            return []

        valid_storm_timestamps = []
        sequence_start = None  # Variable to hold the start of a valid sequence

        for i in range(1, len(shrinking_timestamps)):
            if shrinking_timestamps[i] == shrinking_timestamps[i - 1] + 1:
                if sequence_start is None:
                    sequence_start = shrinking_timestamps[i - 1]  # Mark the start of a valid sequence

            else:
                # If the sequence breaks and we had a valid sequence
                if sequence_start is not None:
                    # Calculate when to mark "storm shrinking" (5 seconds after sequence starts)
                    storm_shrink_start = sequence_start + 5
                    valid_storm_timestamps.append(storm_shrink_start)
                    sequence_start = None  # Reset sequence_start for the next sequence

        # Handle the case where the last sequence reaches the end of the timestamps
        if sequence_start is not None:
            storm_shrink_start = sequence_start + 5
            valid_storm_timestamps.append(storm_shrink_start)

        return valid_storm_timestamps

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
        storm_shrinking_timestamps = []
        frame_count = 0
        storm_shrinking_sequences = []  # Store the detected timestamps for storm shrinking
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

            # Only process one frame per second
            if frame_count % frame_interval == 0:
                # Draw the ROIs on the frame
                cv2.rectangle(frame, (self.elimination_roi['left'], self.elimination_roi['top']),
                              (self.elimination_roi['right'], self.elimination_roi['bottom']), (0, 255, 0), 2)
                cv2.rectangle(frame, (self.storm_roi['left'], self.storm_roi['top']),
                              (self.storm_roi['right'], self.storm_roi['bottom']), (255, 0, 0), 2)
                cv2.rectangle(frame, (self.player_count_roi['left'], self.player_count_roi['top']),
                              (self.player_count_roi['right'], self.player_count_roi['bottom']), (0, 0, 255), 2)

                # Detect eliminations
                if self.validate_frame(frame) and self.detect_eliminated(frame, int(frame_count / fps)):
                    timestamp = int(frame_count / fps)
                    eliminated_timestamps.append(timestamp)

                # Detect storm shrinking warnings
                if self.validate_frame(frame) and self.detect_storm_shrinking(frame, int(frame_count / fps)):
                    timestamp = int(frame_count / fps)
                    storm_shrinking_timestamps.append(timestamp)

                # Process player count every 30 seconds
                if timestamp % 40 == 0:
                    player_count = self.detect_player_count(frame)
                    if player_count is not None:
                        player_count_data[timestamp] = player_count


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

        # Analyze storm shrinking sequences
        valid_storm_shrinking_timestamps = self.analyze_storm_shrinking(storm_shrinking_timestamps)

        # Save the player count data to a JSON file
        self.save_player_count_data(player_count_data)

        return eliminated_timestamps, valid_storm_shrinking_timestamps


def main():
    # Example usage
    commentator = FortniteCommentator()
    input_video = "realvideo.mp4"
    eliminated_timestamps, storm_shrinking_timestamps = commentator.process_video(input_video)

    # Save eliminations to JSON
    if eliminated_timestamps:
        logger.info(f"Detected eliminations at: {eliminated_timestamps} seconds")
        with open('eliminations.json', 'w') as f:
            json.dump(eliminated_timestamps, f)
    else:
        logger.info("No eliminations detected in video")

    # Save storm shrinking warnings to JSON
    if storm_shrinking_timestamps:
        logger.info(f"Detected storm shrinking warnings at: {storm_shrinking_timestamps} seconds")
        with open('storm_shrinking.json', 'w') as f:
            json.dump(storm_shrinking_timestamps, f)
    else:
        logger.info("No storm shrinking warnings detected in video")

if __name__ == "__main__":
    main()
