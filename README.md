Clipit 2

About: A clipping tool designed to make finding and clipping content easier than ever.
Gemini — For searching between product information, show transcripts, and chat logs.
OpenAI – For image recognition and chatbot

Make sure to setup a new venv and install the requirements before launching and editing.
Standalone package is available!



# Instructions for Doppler Key Setup

doppler login
doppler setup --project clipit --config dev
pip3 install -r requirements.txt
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt



# Run with
doppler run -- ./venv/bin/python main.py
