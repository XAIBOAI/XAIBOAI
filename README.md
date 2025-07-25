# XAIBO

XAIBO is a powerful and flexible streaming platform that leverages artificial intelligence to provide enhanced streaming experiences.

## Features

- **AI-Powered Recommendations**: Get personalized content recommendations based on your viewing habits.
- **Real-Time Analytics**: Monitor and analyze streaming performance in real-time.
- **Multi-Platform Support**: Stream across various devices and platforms seamlessly.
- **High-Quality Streaming**: Enjoy high-definition streaming with minimal buffering.

## Installation

To install XAIBO, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/AIStreamer.git
    ```
2. Navigate to the project directory:
    ```bash
    cd AIStreamer
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the streaming service, run the following command:
```bash 
To start the streaming service, run the following commands in sequence:

1. Execute `select_roi.py` to select the region of interest:
    ```bash
    python select_roi.py <input_video>
    ```

2. Run `det.py` to perform detection:
    ```bash
    python det.py <input_video>
    ```

3. Execute `timestamp_filter.py` to filter timestamps:
    ```bash
    python timestamp_filter.py
    ```

4. Finally, run `vid_gen.py` to generate the video:
    ```bash
    python vid_gen.py <input_video> <output_video> <SPEECH.wav>
    ```
```

