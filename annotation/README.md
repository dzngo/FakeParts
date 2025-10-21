# FakePart Human Evaluation

This repo contains a Streamlit app used to evaluate AI-generated videos. The app
loads a catalogue of Drive videos (fake + real), removes their audio on the fly,
serves them for evalutation, and stores the answers in Google Sheets.



## 1. Build the Video Catalogue

1. Create a JSON mapping of Drive folders (recursively scanned) to categories:
   ```json
   {
     "Fake": {
       "Outpainting-AKiRa": ["<drive-folder-id>", "<another-folder-id>"],
       "Inpainting VOS2019-Propainter": ["<drive-folder-id>"]
     },
     "Real": {
       "Real": ["<drive-folder-id>"]
     }
   }
   ```
2. Generate the video catalogue and write it to a Google Sheet:
   ```bash
   python generate_video_list.py --config mapping.json \
       --sheet-url https://docs.google.com/spreadsheets/d/<sheet-id>/edit \
       --worksheet VideoCatalog \
       --output video_catalog.csv  # optional local copy
   ```
   The sheet must already exist and be shared with the service-account email
   (Editor access). If `--worksheet` is omitted the first worksheet is used.
   `--output` is optional and only produces a local CSV for inspection.

## 2. Configure Secrets / Environment

The Streamlit app looks for these values (either in environment variables or in
`.streamlit/secrets.toml`):

```toml
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/<sheet-id>"
VIDEO_CATALOG_SHEET_URL = "https://docs.google.com/spreadsheets/d/<catalog-sheet-id>"
VIDEO_CATALOG_WORKSHEET = "VideoCatalog"  # optional
DEFAULT_SAMPLE_SIZE = 10  # optional override
FAKE_METHODS_BALANCED = false  # optional override (set to true to balance fake methods)
GCP_CREDENTIALS = '''{ ... service account JSON ... }'''  # or set GOOGLE_APPLICATION_CREDENTIALS
```

Alternatively set `GOOGLE_APPLICATION_CREDENTIALS` to the path of the JSON key
file before running scripts or the app.

When deploying to Streamlit Cloud, add a `packages.txt` containing `ffmpeg` so
it is installed automatically.

## 3. Run the App Locally

```bash
pip install -r requirements.txt
streamlit run annotation/app.py
```

The app will download the video catalogue, sample equal numbers of fake and
real videos, remove their audio, and log each evaluation to Google Sheets. Set
`FAKE_METHODS_BALANCED=true` if you prefer to balance the sampled fake videos
across methods.
