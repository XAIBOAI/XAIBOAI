import os
import json
import random
import logging
import time
from typing import List, Dict
import moviepy as mp
import torch
from TTS.api import TTS
import argparse


class ProgressBar:
    def __init__(self, total, prefix='Progress:', length=50):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.start_time = time.time()
        self.current = 0

    def update(self, current):
        self.current = current
        self._print_progress()

    def _print_progress(self):
        elapsed_time = max(0.001, time.time() - self.start_time)  # Prevent division by zero
        percentage = (self.current / float(self.total)) * 100
        filled_length = int(self.length * self.current // self.total)
        bar = '█' * filled_length + '-' * (self.length - filled_length)

        # Calculate estimated remaining time
        try:
            items_per_second = self.current / elapsed_time
            remaining_items = self.total - self.current
            remaining_time = remaining_items / items_per_second if items_per_second > 0 else 0
        except:
            remaining_time = 0

        print(f'\r{self.prefix} |{bar}| {percentage:.1f}% Complete '
              f'(Elapsed: {elapsed_time:.1f}s, Remaining: {remaining_time:.1f}s)', end='')
        if self.current == self.total:
            print()


class VideoCommentaryGenerator:
    def __init__(self, config: Dict = None):
        self.config = {
            'game_audio_volume': 0.4,
            'commentary_volume': 1.6,
            'comments_file': 'comments.json',
            'jokes_file': 'jokes.json',
            'temp_dir': 'temp_audio',
            **(config or {})
        }

        print("\nInitializing VideoCommentaryGenerator...")
        progress = ProgressBar(4, prefix='Setup Progress')

        os.makedirs(self.config['temp_dir'], exist_ok=True)
        progress.update(1)

        self.tts_engine = self._init_tts()
        progress.update(2)

        self.comments = self._load_json(self.config['comments_file'])
        progress.update(3)

        self.jokes = self._load_json(self.config['jokes_file'])
        progress.update(4)

    def _init_tts(self) -> TTS:
        try:
            return TTS("tts_models/multilingual/multi-dataset/xtts_v2",
                       gpu=torch.cuda.is_available())
        except Exception as e:
            logging.error(f"Failed to initialize TTS: {e}")
            raise

    def _load_json(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load {file_path}: {e}")
            raise

    def _generate_audio(self, text: str, output_file: str, speaker_wav: str = None):
        try:
            self.tts_engine.tts_to_file(
                text=text,
                file_path=output_file,
                speaker_wav=speaker_wav,
                language="en"
            )
        except Exception as e:
            logging.error(f"Failed to generate audio: {e}")
            raise

    def _create_audio_clip(self, timestamp: float, comment: str,
                           prefix: str, index: int, speaker_wav: str = None) -> mp.AudioFileClip:
        """Create a single audio clip with the given comment at the specified timestamp"""
        output_file = os.path.join(self.config['temp_dir'], f"{prefix}_{index}.wav")
        self._generate_audio(comment, output_file, speaker_wav)
        return mp.AudioFileClip(output_file).with_start(float(timestamp))

    def add_commentary(self, video_path: str, output_path: str,
                       filtered_dir: str = "filtered_output",
                       speaker_wav: str = None):
        """Add commentary using pre-filtered timestamp files"""
        try:
            # Load filtered timestamps
            eliminations = self._load_json(os.path.join(filtered_dir, "filtered_eliminations.json"))
            storm_times = self._load_json(os.path.join(filtered_dir, "filtered_storm.json"))
            player_data = self._load_json(os.path.join(filtered_dir, "filtered_player_count.json"))
            joke_times = self._load_json(os.path.join(filtered_dir, "filtered_jokes.json"))

            audio_clips = []

            # Generate elimination clips
            for i, timestamp in enumerate(eliminations):
                comment = random.choice(self.comments['eliminations'])
                clip = self._create_audio_clip(timestamp, comment, 'elim', i, speaker_wav)
                audio_clips.append(clip)

            # Generate player count clips - using sequential comments
            player_comments = self.comments['player_count']
            for i, (timestamp, count) in enumerate(player_data.items()):
                # Use modulo to cycle through comments if we have more counts than comments
                comment = player_comments[i % len(player_comments)].replace('{players_left}', str(count))
                clip = self._create_audio_clip(float(timestamp), comment, 'player', i, speaker_wav)
                audio_clips.append(clip)

            # Generate storm clips
            for i, timestamp in enumerate(storm_times):
                comment = random.choice(self.comments['storm_shrinking'])
                clip = self._create_audio_clip(timestamp, comment, 'storm', i, speaker_wav)
                audio_clips.append(clip)

            # Generate joke clips
            for i, timestamp in enumerate(joke_times):
                joke = random.choice(self.jokes)
                clip = self._create_audio_clip(timestamp, joke, 'joke', i, speaker_wav)
                audio_clips.append(clip)

            # Combine with video
            video = mp.VideoFileClip(video_path)
            final_audio = mp.CompositeAudioClip([video.audio] + audio_clips)

            # Create final video
            final_video = video.with_audio(final_audio)
            final_video.write_videofile(
                output_path,
                codec='h264_nvenc',
                audio_codec='aac',
                ffmpeg_params=[
                    "-c:v", "h264_nvenc",
                    "-preset", "fast",
                    "-b:v", "10M",
                    "-maxrate", "15M",
                    "-pix_fmt", "yuv420p"
                ]
            )

        except Exception as e:
            logging.error(f"Failed to process video: {e}")
            raise

        finally:
            # Cleanup temporary files
            for file in os.listdir(self.config['temp_dir']):
                try:
                    os.remove(os.path.join(self.config['temp_dir'], file))
                except Exception as e:
                    logging.warning(f"Failed to remove temp file {file}: {e}")


def main():
    try:
        # Initialize generator
        generator = VideoCommentaryGenerator()

        # Example usage (replace with your file paths)
        input_video = "realvideo.mp4"
        output_video = "outputrealone.mp4"
        filtered_dir = "filtered_output"
        speech_file = "SPEECH.wav"

        # Process video (add your video processing method here)
        generator.add_commentary(input_video, output_video, filtered_dir, speech_file)

    except Exception as e:
        logging.error(f"Error in video processing: {e}")
        return 1
    return 0


if __name__ == '__main__':
    main()