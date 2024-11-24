from aiohttp import web
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import see_processor
import os
from pydub import AudioSegment

processor = see_processor.SeeProcessor()

async def post_handler(request):
    try:
        data = await request.json()
        image_url = data['image_url']
        audio_url = data['audio_url']

        print(not audio_url is None, not image_url is None)

        convert_and_process(audio_url, image_url)

        return web.Response(text=str("received"), status=200)
    except ValueError as e:
        print(e)
        return web.Response(text=str(e), status=500)

def convert_and_process(audio_data_url, image_data_url, audio_output="audio.wav", image_output="image.jpeg"):
    try:
        # Convert audio (data:audio/webm;base64 to .wav)
        if audio_data_url.startswith("data:audio/webm;base64,"):
            base64_audio = audio_data_url.split(",")[1]
            audio_data = base64.b64decode(base64_audio)

            # Temporary .webm audio file
            temp_audio_file = "temp_audio.webm"
            with open(temp_audio_file, "wb") as audio_file:
                audio_file.write(audio_data)

            # Convert .webm to .wav using pydub
            audio = AudioSegment.from_file(temp_audio_file, format="webm")
            audio.export(audio_output, format="wav")
            audio_absolute_path = os.path.abspath(audio_output)
            print(f"Audio saved as {audio_absolute_path}.")
        else:
            raise ValueError("Invalid audio data URL format.")

        # Convert image (data:image/jpeg;base64 to .jpeg)
        if image_data_url.startswith("data:image/jpeg;base64,"):
            base64_image = image_data_url.split(",")[1]
            image_data = base64.b64decode(base64_image)

            # Save the JPEG image directly
            with open(image_output, "wb") as image_file:
                image_file.write(image_data)
            image_absolute_path = os.path.abspath(image_output)
            print(f"Image saved as {image_absolute_path}.")
        else:
            raise ValueError("Invalid image data URL format.")

        # Process the audio and image files (replace with your actual logic)
        process_files(image_output, audio_output)

    finally:
        # Clean up temporary files
        if os.path.exists(audio_output):
            os.remove(audio_output)
        if os.path.exists(image_output):
            os.remove(image_output)
        if os.path.exists("temp_audio.webm"):
            os.remove("temp_audio.webm")
        print("Temporary files deleted.")

def process_files(audio_file, image_file):
    print("Pass files into processor.")
    processor.generate(audio_file, image_file)

app = web.Application()
app.router.add_post("/api/detect", post_handler)
app.add_routes([web.static('/', "./build", show_index=True)])

if __name__ == '__main__':
    web.run_app(app)