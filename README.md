XAIBO is an AI-powered game analysis and commentary platform designed to review and comment on live CS2 matches in real-time or from stream recordings. With intelligent behavior tracking, event detection, and dynamic voice commentary, XAIBO provides entertaining and insightful breakdowns of every play.

Features
AI-Powered Commentary: Live review of CS2 gameplay with witty, smart, and sometimes savage remarks.

Event Detection: Detects kills, flashes, molotovs, triple kills, wins, and other CS2 highlights.

Stream Input Support: Connect directly to live or recorded Twitch/YouTube/OBS streams.

Customizable Voice Output: Choose from multiple voice styles and personalities.

Multiplatform Streaming: Supports output to Twitch, YouTube, or local save.

Installation
To install XAIBO, follow these steps:

bash
Kopiëren
Bewerken
git clone https://github.com/yourusername/xaibo.git
cd xaibo
pip install -r requirements.txt
Usage
To start reviewing a CS2 stream, follow these steps:

1. Set the stream source
Provide the URL or stream key (RTMP) of the CS2 game stream:

bash
Kopiëren
Bewerken
python connect_stream.py --source <stream_url_or_key>
Example:

bash
Kopiëren
Bewerken
python connect_stream.py --source "rtmp://twitch.tv/live/your_channel"
2. Start the real-time event detection and commentary engine
bash
Kopiëren
Bewerken
python xaibo_review.py --agent XAIBO --voice "cyber-funny" --stream_mode live
This will:

Connect to the stream

Detect key gameplay events (kills, flashes, wins, etc.)

Trigger voice commentary in real time or with minimal delay

3. Optional: Save a reviewed version with overlays
bash
Kopiëren
Bewerken
python save_reviewed_stream.py --source <stream_url> --output reviewed_game.mp4
Developer Notes
xaibo_review.py contains the main AI review engine (detection + commentary logic).

connect_stream.py handles the RTMP/stream capture and frame extraction.

voice_engine.py converts AI text commentary into speech using ElevenLabs or custom models.

event_detector/ includes all detection logic (kills, molotovs, flashed, etc.).

Coming Soon
Support for stream chat integration to react to funny or hyped moments.

Overlay engine for real-time meme inserts or stat breakdowns.

Replay highlight mode with cinematic slow-motion edits.
