# YouTube Ad Analyzer – Marketing Insights Pro

This Streamlit application analyzes YouTube video data to extract marketing insights such as engagement rates, audience sentiment, and channel performance. 
The app utilizes the YouTube Data API and Natural Language Processing for sentiment analysis of video comments.

## Features
- Connects to YouTube Data API to retrieve video data such as views, likes, and comments.
- Sentiment analysis on comments using VADER Sentiment Analysis.
- Interactive visualizations like word clouds, sentiment trends, and engagement rate charts.
- Automatically saves results to Google Sheets for long-term tracking and analysis.
- Advanced thumbnail analysis using Tesseract OCR for text recognition and OpenCV for face detection.

##  Requirements
All dependencies are listed in the `requirements.txt` file.

To install the necessary packages, run:
pip install -r requirements.txt

## Google Sheets and YouTube API Setup

A) Google Sheets (Service Account)
  1. Create a Google Cloud Project:
      - Go to Google Cloud Console, create a new project, or select an existing one.
  2. Enable the APIs:
      - Enable Google Sheets API and Google Drive API in your Google Cloud Console project.
  3. Create a Service Account:
      - Go to IAM & Admin → Service Accounts and create a new service account.
      - Download the JSON credentials file for the service account.
  4.Grant access to your Google Sheet:
      - Open the Google Sheets document and share it with the service account's email (you'll find this email in the JSON file you downloaded).
  5. Store the credentials securely:
      - For Hugging Face users: Go to Settings → Secrets and add a new secret.
      - Name it GOOGLE_CREDENTIALS_JSON and paste the JSON content.
        - For local setup:
          - In Windows (PowerShell), run:
            powershell
            setx GOOGLE_CREDENTIALS_JSON "<PASTE_YOUR_JSON_HERE>"
          - In Mac/Linux (Terminal), run:
            export GOOGLE_CREDENTIALS_JSON="<PASTE_YOUR_JSON_HERE>"

B) YouTube Data API Key
  1. Create the API Key:
      - Go to Google Cloud Console, select your project, and enable the YouTube Data API v3.
      - Create an API Key under APIs & Services → Credentials.
  3. Store the API Key securely:
      - For Hugging Face users: Go to Settings → Secrets and add a new secret.
     - Name it YT_API_KEY and paste your API key.
        - For local setup:
          - In Windows (PowerShell), run:
            powershell
            setx YT_API_KEY "<YOUR_YOUTUBE_API_KEY>"
          - In Mac/Linux (Terminal), run:
            export YT_API_KEY="<YOUR_YOUTUBE_API_KEY>"

## Running Locally
To run the application locally, use the following command:
streamlit run streamlit_app.py

## Docker Setup
You can also run this application using Docker. This is a great way to set up the app in an isolated environment.
1. Build the Docker image:
   -  In the root directory of the project (where your Dockerfile is located), build the Docker image by running the following command:
       docker build -t youtube-ad-analyzer .

2. Run the Docker container:
   -  After building the image, run the container with the following command:
       docker run -p 8501:8501 youtube-ad-analyzer

This will start the application, and you can access it in your web browser by navigating to http://localhost:8501.

## Project Structure
  - streamlit_app.py -> Main Streamlit app
  - requirements.txt -> Python dependencies
  - Dockerfile -> Docker configuration file
  - .gitignore       -> Git ignore file (to avoid uploading sensitive data)
  - .env.example     -> Example environment variable configuration

Use environment variables (or Secrets in Hugging Face) to keep your credentials secure.

## License
This project is licensed under the MIT License.
