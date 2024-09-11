import whisper
import moviepy.editor as mp
from gtts import gTTS
from googletrans import Translator
from audiomentations import Compose, TimeStretch
import os
from pydub import AudioSegment
import numpy as np

# Function to extract audio from the video
def extract_audio_from_video(video_path, audio_output_path):
    print(f"Extracting audio from video: {video_path}")
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path, codec='pcm_s16le')
    print(f"Audio extracted and saved to: {audio_output_path}")

# Function to translate the audio using Whisper
def translate_audio_whisper(audio_path, model_size="medium"):
    print(f"Transcribing and translating audio from: {audio_path}")
    # Load Whisper model on CPU
    model = whisper.load_model(model_size).to("cpu")  # Load model on CPU

    # Perform transcription and translation
    result = model.transcribe(audio_path, task="translate")
    
    # Print the transcribed text (in source language)
    print(f"Transcribed text from audio: {result['text']}")

    # Return the translated text
    return result['text']

# Function to translate text using Google Translator
def translate_text_google(text, dest_language="hi"):
    translator = Translator()
    translated = translator.translate(text, dest=dest_language)
    translated_text = translated.text
    print(f"Translated text: {translated_text}")
    return translated_text

# Function to convert translated text to speech using gTTS
def text_to_speech(translated_text, output_audio_path, language="hi"):
    print(f"Converting translated text to speech in language: {language}")
    mp3_path = "translated_audio.mp3"
    tts = gTTS(translated_text, lang=language)
    tts.save(mp3_path)
    
    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(output_audio_path, format="wav")
    os.remove(mp3_path)  # Clean up the temporary MP3 file
    
    print(f"Translated speech saved to: {output_audio_path}")

# Function to adjust the duration of the audio
def adjust_audio_duration(audio_path, target_duration, output_audio_path):
    print(f"Adjusting audio duration to match target duration: {target_duration} seconds")
    audio = AudioSegment.from_wav(audio_path)
    original_duration = len(audio) / 1000  # Duration in seconds
    
    if original_duration != target_duration:
        # Create time stretch augmentation
        time_stretch = Compose([TimeStretch(min_rate=1.0, max_rate=1.5)])
        audio_samples = np.array(audio.get_array_of_samples())
        augmented_samples = time_stretch(samples=audio_samples, sample_rate=audio.frame_rate)
        
        # Create new audio segment from augmented samples
        augmented_audio = AudioSegment(
            augmented_samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
        
        # Adjust the length by truncating or padding
        if len(augmented_audio) > target_duration * 1000:
            augmented_audio = augmented_audio[:int(target_duration * 1000)]
        else:
            silence_duration = int(target_duration * 1000) - len(augmented_audio)
            silence = AudioSegment.silent(duration=silence_duration)
            augmented_audio = augmented_audio + silence

        augmented_audio.export(output_audio_path, format="wav")
        print(f"Adjusted audio saved to: {output_audio_path}")
    else:
        # No adjustment needed
        os.rename(audio_path, output_audio_path)

# Function to replace the audio in the video with new audio
def replace_audio_in_video(video_path, new_audio_path, output_video_path):
    print(f"Replacing audio in video: {video_path}")
    video = mp.VideoFileClip(video_path)
    audio = mp.AudioFileClip(new_audio_path)
    new_video = video.set_audio(audio)
    new_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
    print(f"New video with replaced audio saved to: {output_video_path}")

# Main function to process the video
def translate_video(video_path, model_size="medium", output_language="hi"):
    # Define paths
    audio_path = "extracted_audio.wav"
    translated_audio_path = "translated_audio.wav"
    adjusted_audio_path = "adjusted_audio.wav"
    output_video_path = "translated_video.mp4"

    # Step 1: Extract audio from the input video
    extract_audio_from_video(video_path, audio_path)
    
    # Step 2: Translate the extracted audio using Whisper
    transcribed_text = translate_audio_whisper(audio_path, model_size)

    # Step 3: Translate the transcribed text to the desired language using Google Translator
    translated_text = translate_text_google(transcribed_text, dest_language=output_language)

    # Step 4: Convert the translated text into speech using TTS
    text_to_speech(translated_text, translated_audio_path, language=output_language)

    # Step 5: Adjust the duration of the translated audio to match the original video duration
    video = mp.VideoFileClip(video_path)
    target_duration = video.duration  # Duration in seconds
    adjust_audio_duration(translated_audio_path, target_duration, adjusted_audio_path)

    # Step 6: Replace the original audio in the video with the adjusted translated audio
    replace_audio_in_video(video_path, adjusted_audio_path, output_video_path)

    print(f"Translation complete. Output video saved as: {output_video_path}")

# Example usage
if __name__ == "__main__":
    input_video_path = "checkpoints/video.mp4"  # Replace with your input video path
    translate_video(input_video_path, model_size="medium", output_language="hi")  # 'hi' is Hindi. Change to any target language
