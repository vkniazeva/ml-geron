import os
import pandas as pd

def fetch_file_data(filename, chapter):
    """
    Load housing data from the assets folder.

    Parameters:
    - filename: str, name of the CSV file
    - chapter: str, chapter folder inside assets

    Returns:
    - pandas.DataFrame with the housing data
    """

    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    csv_path = os.path.join(project_root, "assets", chapter, filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        data = pd.read_csv(csv_path)
        return data
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {csv_path} is empty.")
    except pd.errors.ParserError:
        raise ValueError(f"The file {csv_path} could not be parsed.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading {csv_path}: {e}")