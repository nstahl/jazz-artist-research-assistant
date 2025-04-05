# jazz-artist-research-assistant
Research assistant to look up jazz artists and collect relevant info about them



## Getting Started

> Activate your python `venv` if you're using one.

First, install dependencies from requirements.txt.

```bash
pip install -r requirements.txt
```

Then start Jupyter Notebook and open the python notebook.


```bash
jupyter notebook
```

## Running the Main Script

The main script (`artist-name-generator.py`) does the following:

1. Reads artist names from `jazz_concerts_data.json`
2. For each artist, it:
   - Generates a biography using AI
   - Finds their official website
   - Finds their Instagram profile
   - Finds YouTube performance videos
3. Processes multiple artists concurrently (3 at a time)
4. Saves all results to `all_artist_states.json`

To run the script:

```bash
python artist-name-generator.py
```
