from aiohttp import web
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import see_processor
import os
from pydub import AudioSegment

#processor = see_processor.SeeProcessor()

async def post_handler(request):
    try:
        data = await request.json()
        audio_url = data['audio_url']
        image_url = data['image_url']

        print(audio_url, image_url)

        convert_and_process_audio(audio_url)

        return web.Response(text=str("received"), status=200)
    except ValueError as e:
        print(e)
        return web.Response(text=str(e), status=500)
    
def convert_and_process_audio(data_url, output_filename="audio.wav"):
    try:
        # Strip the metadata prefix
        if data_url.startswith("data:audio/webm;base64,"):
            base64_audio = data_url.split(",")[1]
        else:
            raise ValueError("Invalid data URL format.")

        # Decode the base64 audio data
        audio_data = base64.b64decode(base64_audio)

        # Temporary .webm file
        temp_webm_file = "temp_audio.webm"
        with open(temp_webm_file, "wb") as webm_file:
            webm_file.write(audio_data)

        # Convert .webm to .wav using pydub
        audio = AudioSegment.from_file(temp_webm_file, format="webm")
        audio.export(output_filename, format="wav")
        print(f"Audio saved as {output_filename}.")

        # Process the audio file (Replace this with your logic)
        # process_audio(output_filename)

    finally:
        print("done")
        # Clean up temporary files
        # if os.path.exists(temp_webm_file):
        #     os.remove(temp_webm_file)
        # if os.path.exists(output_filename):
        #     os.remove(output_filename)
        # print("Temporary files deleted.")

app = web.Application()
app.router.add_post("/api/detect", post_handler)
app.add_routes([web.static('/', "./build", show_index=True)])

if __name__ == '__main__':
    web.run_app(app)