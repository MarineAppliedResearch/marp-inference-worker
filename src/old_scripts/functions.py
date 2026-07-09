import cv2
import yaml
from database_video_annotations import DatabaseVideoAnnotationsRangeFinder, AnnotationRectangle
import requests
import urllib.parse
import datetime

api_base_url = 'http://192.168.1.203:3081/api'  # API endpoint base URL
data_yaml_path = 'data.yaml'               # Path to the data.yaml file containing class names

# Initialize the range finder
range_finder = DatabaseVideoAnnotationsRangeFinder()


import random
import datetime

# ==========================================================
# API: createDataset (FAKE ENDPOINT)
# ==========================================================
def createDataset(name, location, description="", numSamples=0, numClasses=0, source="model_training_live.py", notes=""):
    """
    POST /api/datasets
    ----------------------------------------------------------
    Simulates the creation of a new dataset record in the MARP
    database. This mock version behaves identically to the real
    API call but generates and returns fake data locally.

    Parameters:
        name (str): Name of the dataset.
        location (str): Filesystem or URI path where dataset resides.
        description (str): Short description of dataset purpose.
        numSamples (int): Number of total samples in dataset.
        numClasses (int): Number of unique species/classes.
        source (str): How the dataset was created ("manual", "auto", etc).
        notes (str): Optional notes or comments.

    Returns:
        dict: A simulated dataset record in the same structure
              the real API would return, e.g.:
              {
                  "id": 4821,
                  "name": "FishDataset2025",
                  "location": "datasets/FishDataset2025",
                  "description": "Dataset built from 22 species",
                  "num_samples": 1000,
                  "num_classes": 22,
                  "source": "manual",
                  "notes": "",
                  "created_at": "2025-10-08T14:35:12.123Z"
              }
    """
    print("\n[FAKE API] POST /api/datasets — Creating dataset record...")
    datasetId = random.randint(1000, 9999)
    datasetRecord = {
        "id": datasetId,
        "name": name,
        "location": location,
        "description": description,
        "num_samples": numSamples,
        "num_classes": numClasses,
        "source": source,
        "notes": notes,
        "created_at": datetime.datetime.now().isoformat()
    }

    print(f"[FAKE API] Dataset created → ID {datasetId} ({name}) at {location}")
    return datasetRecord

# ==========================================================
# API: addDatasetObservations (FAKE ENDPOINT)
# ==========================================================
def addDatasetObservations(datasetId, observations):
    """
    POST /api/dataset_observations/bulk
    ----------------------------------------------------------
    Simulates adding many observation entries to a dataset in
    the MARP database. The mock version accepts a list of
    observation mappings and prints the first few for debugging.

    Parameters:
        datasetId (int): ID of the dataset to associate.
        observations (list[dict]): Each element should contain:
            {
                "observation_id": int,
                "inclusion_type": str,    # e.g. "train", "val", "test"
                "num_keyframes": int,
                "selected_by": str        # e.g. "manual", "auto"
            }

    Returns:
        dict: Simulated response containing the dataset ID and
              number of inserted records, e.g.:
              {
                  "inserted": 215,
                  "dataset_id": 4821
              }
    """
    count = len(observations)
    print(f"\n[FAKE API] POST /api/dataset_observations/bulk — Adding {count} records to dataset {datasetId}...")
    if count > 0:
        print(f"[FAKE API] Preview of first 3 entries: {observations[:3]}")
    else:
        print("[FAKE API] No observations provided.")
    result = {"inserted": count, "dataset_id": datasetId}
    print(f"[FAKE API] Successfully added {count} dataset_observations.")
    return result


"""
    Creates a new dataset_observation record in the MARP database via the API.

    Parameters:
        dataset_observation_data (dict): Example:
            {
                    "dataset_id": 1,
                    "observation_id": 12345,
                    "inclusion_type": "train",
                    "selection_method": "manual",
                    "weight": 9,
                    "notes" : "notes here",
                    "added_at": "2025-10-08T00:00:00Z"
            }

    Returns:
        dict: The created dataset_observation record, or None on failure.
"""
def createDatabaseDatasetObservation(dataset_observation_data):
    
    try:
        url = api_base_url + "/dataset_observation"
        payload = { "dataset_observation": dataset_observation_data }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        print(f"[INFO] Dataset observation added: observation_id={dataset_observation_data['observation_id']}")
        return response.json()

    except Exception as e:
        print(f"[ERROR] createDatabaseDatasetObservation(): {e}")
        return None


"""
    Performs a bulk insert of dataset_observations via the MARP API.

    Parameters:
        dataset_observations (list of dict): Example:
            [
                {
                    "dataset_id": 1,
                    "observation_id": 12345,
                    "inclusion_type": "train",
                    "selection_method": "manual",
                    "weight": 9,
                    "notes" : "notes here",
                    "added_at": "2025-10-08T00:00:00Z"
                },
                ...
            ]

    Returns:
        dict: {"inserted": <count>} on success, or None on failure.

        THIS SOMETIMES GIVES THE ERROR 413 CLIENT erROR, PAYLOAD TO LARGE FOR URL
"""
def createDatabaseDatasetObservationsBulk(dataset_observations):
    
    try:
        url = api_base_url + "/dataset_observations/bulk"
        payload = {"dataset_observations": dataset_observations}

        response = requests.post(url, json=payload)
        response.raise_for_status()

        print(f"[INFO] Bulk-inserted {response.json().get('inserted', 0)} dataset observations.")
        return response.json()
    except Exception as e:
        print(f"[ERROR] createDatabaseDatasetObservationsBulk(): {e}")
        return None


def getDatabaseDatasets():
    """
    Fetches the list of datasets from the MARP API.
    Returns a list of datasets dictionaries.
    Example:
        [
            {
                "id": 1,
                "name": "Lingcod",
                "description": "Ophiodon elongatus",
                "location": "LING",
                "num_samples": "—",
                "num_classes": 2,
                "source": "test",
                "notes": "test"
            },
            ...
        ]
    """
    try:
        url = api_base_url+"/dataset"  # adjust if your API runs elsewhere
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] get_species(): {e}")
        return []
    

# ------------------------------------------------------------
# getDatabaseModels
# ------------------------------------------------------------
# Fetches all model records from the MARP API.
# Returns:
#   list of dicts, each representing a model entry.
# Example response:
#   [
#       {
#           "id": 1,
#           "name": "YOLOv8-Fish-2025",
#           "model_type": "YOLOv8",
#           "architecture_version": "v8n",
#           "created_at": "2025-10-08T00:00:00.000Z",
#           "status": "trained",
#           "notes": "Base model for fish detection"
#       },
#       ...
#   ]
# ------------------------------------------------------------
def getDatabaseModels():
    try:
        url = api_base_url + "/ml_models"
        response = requests.get(url)
        response.raise_for_status()
        models = response.json()
        print(f"[INFO] Retrieved {len(models)} models from database.")
        return models
    except Exception as e:
        print(f"[ERROR] getDatabaseModels(): {e}")
        return []
    

# ==========================================================
# CREATE DATABASE MODEL
# ----------------------------------------------------------
# Endpoint: POST /api/model
# Creates a new ML model record in the database.
#
# Args:
#   model_data (dict): Model fields matching the ml_models table.
#       Example:
#       {
#           "name": "yolov8_fish_2025",
#           "parent_model_id": 1,
#           "model_type": "YOLOv8",
#           "architecture_version": "custom-2025a",
#           "storage_path": "models/yolov8_fish_2025/weights",
#           "status": "training",
#           "notes": "Fine-tuned from yolov8_base on Fish2025 dataset"
#       }
#
# Returns:
#   dict: JSON response from the API with created model data.
# ==========================================================
def createDatabaseModel(model_data):
    try:
        url = f"{api_base_url}/model"
        response = requests.post(url, json={"model": model_data})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] createDatabaseModel(): {e}")
        return None


# ==========================================================
# UPDATE DATABASE MODEL
# ----------------------------------------------------------
# Endpoint: PUT /api/model/:id
# Updates an existing ML model record with new data.
#
# Args:
#   model_id (int): The model's unique ID.
#   update_data (dict): Fields to update.
#       Example:
#       {
#           "storage_path": "models/yolov8_fish_2025/weights",
#           "status": "trained",
#           "updated_at": "2025-10-08T14:30:00Z"
#       }
#
# Returns:
#   dict: JSON response from the API with updated model data.
# ==========================================================
def updateDatabaseModel(model_id, update_data):
    try:
        url = f"{api_base_url}/model/{model_id}"
        response = requests.put(url, json={"model": update_data})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] updateDatabaseModel(): {e}")
        return None


# ------------------------------------------------------------
# getDatabaseDatasetById
# ------------------------------------------------------------
# Fetches full dataset details from the MARP API using its ID.
# Parameters:
#   dataset_id (int) – ID of the dataset to fetch
# Returns:
#   dict: dataset info, or None on failure.
# ------------------------------------------------------------
def getDatabaseDatasetById(dataset_id):
    try:
        url = f"{api_base_url}/dataset/{dataset_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] getDatabaseDatasetById({dataset_id}): {e}")
        return None
    

def createDatabaseDataset(dataset_data):
    """
    Creates a new dataset record in the MARP database via the API.

    Parameters:
        dataset_data (dict): Example:
            {
                "name": "yolo_dataset_campa2025_test6",
                "description": "The 6th Campa 2025 test dataset",
                "location": "yolo_dataset_campa2025_test6/",
                "num_samples": 31970,
                "num_classes": 4,
                "source": "manual",
                "notes": "Manually created"
            }

    Returns:
        dict: The created dataset record returned by the API, or None on failure.
    """
    try:
        url = api_base_url + "/dataset"
        payload = { "dataset": dataset_data }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        print(f"[INFO] Dataset created successfully: {dataset_data['name']}")
        return response.json()

    except Exception as e:
        print(f"[ERROR] createDatabaseDataset(): {e}")
        return None


"""
    Fetches the list of species from the MARP API.
    Returns a list of species dictionaries.
"""
def getSpecies():
    
    try:
        url = api_base_url+"/species"  # adjust if your API runs elsewhere
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] get_species(): {e}")
        return []
    
"""
    Fetches a single species record by its common name (case-insensitive).
    Returns a species dictionary or None if not found.
"""
def getSpeciesByComname(comname):
    try:
        url = f"{api_base_url}/species/by-comname/{comname}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] getSpeciesByComname(): {e}")
        return None
    

"""
    Inserts a record into the model_species table via the MARP API.
    Returns the created record as a dictionary.
"""
def createDatabaseModelSpecies(record):
    try:
        url = f"{api_base_url}/model_species"
        response = requests.post(url, json=record)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] createDatabaseModelSpecies(): {e}")
        return {}



def createDatabaseMetricsCurve(data):
    try:
        url = api_base_url + "/metrics_curve"
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] createDatabaseMetricsCurve(): {e}")
        return None
    
def createDatabaseMetricsCurvesBulk(records, batch_size=200):
    """POST metrics_curves records in batches."""
    try:
        url = api_base_url + "/metrics_curves/bulk"
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            response = requests.post(url, json=chunk)
            response.raise_for_status()
        print(f"[DB] Successfully uploaded {len(records)} curve points in batches of {batch_size}")
    except Exception as e:
        print(f"[ERROR] createDatabaseMetricsCurvesBulk(): {e}")

def getObservationsWithKeyframesByComnames(comname_list):
    """
    Fetches observations that have associated keyframes and match the provided comname list.

    :param comname_list: A list of comnames (strings) to filter observations.
    :type comname_list: list
    :return: A list of observations if the request is successful, otherwise an empty list.
    :rtype: list
    """
    url = api_base_url+"/getObservationsWithKeyframesByComnames"
    try:
        if not comname_list or not isinstance(comname_list, list):
            print("Invalid comname_list provided. Must be a non-empty list of strings.")
            return []

        # Join the list of comnames without additional encoding
        query_string = {"comnameList": ",".join(comname_list)}
        
        # Send the GET request to the API with the query parameter
        response = requests.get(url, params=query_string)
        response.raise_for_status()
        
        # Parse the JSON response
        observations = response.json()
        if not isinstance(observations, list):
            print("Unexpected API response format. Expected a list of observations.")
            return []
        return observations
    except requests.exceptions.RequestException as e:
        print(f"Error fetching observations with keyframes by comnames: {e}")
        return []


def getDistinctComnamesWithKeyframes():
    """
    Fetches a list of distinct comnames (common names) that have associated keyframes.

    This function sends an HTTP GET request to the API endpoint
    `http://localhost:3081/api/getDistinctComnamesWithKeyframes` and retrieves a list of
    comnames with associated keyframes. If the API request fails or the response format
    is invalid, it handles the error gracefully and returns an empty list.

    :return: A list of distinct comnames (strings) if the request is successful, otherwise an empty list.
    :rtype: list
    """
    # Define the API endpoint URL
    url = api_base_url+"/getDistinctComnamesWithKeyframes"
    
    try:
        # Send a GET request to the API
        response = requests.get(url)
        
        # Raise an HTTPError if the status code indicates a failure (e.g., 4xx or 5xx)
        response.raise_for_status()
        
        # Parse the JSON response from the API
        comnames = response.json()
        
        # Check if the API response is in the expected format (a list of strings)
        if not isinstance(comnames, list):
            print("Unexpected API response format. Expected a list of comnames.")
            return []  # Return an empty list if the response format is invalid
        
        # Return the list of comnames if everything is successful
        return comnames
    
    except requests.exceptions.RequestException as e:
        # Catch any exceptions related to the HTTP request (e.g., connection errors, timeouts)
        print(f"Error fetching distinct comnames: {e}")
        return []  # Return an empty list in case of an error

def getObservationsByVideo(video_name):
        
    # Set up the parameters for the GET request
    params = {'videoName': video_name}
    
    # Make the GET request to the API endpoint
    response = requests.get(api_base_url+"/getObservationsByVideo", params=params)
    response.raise_for_status()  # Raise an exception if the request was unsuccessful
    observations = response.json()

    return observations

# Function to convert mediaPosition (in format HH:MM:SS.SSS) to seconds
def media_position_to_seconds(media_position):
    try:
        time_parts = media_position.split(':')
        if len(time_parts) != 3:
            raise ValueError(f"Invalid mediaPosition format: {media_position}")
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = float(time_parts[2])
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except ValueError as e:
        print(f"Error parsing mediaPosition '{media_position}': {e}")
        return None
    
# Load classnames from data_yaml_path
def loadClassNames():

    returnVal = None

    # Load class names from data.yaml
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            class_names = data['names']
            print(f"Loaded {len(class_names)} class names from '{data_yaml_path}'.")
            returnVal = class_names
    except FileNotFoundError:
        print(f"Error: The file '{data_yaml_path}' was not found.")
    except Exception as e:
        print(f"Error loading data.yaml: {e}")

    return returnVal
    

# Function to extract a frame from a video at a specific time
def extract_frame(video_path, time_in_seconds, output_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Saved frame at {time_in_seconds}s to {output_path}")
    else:
        print(f"Failed to extract frame at {time_in_seconds}s from {video_path}")
    cap.release()


# Function to write the YOLO annotation file
def write_yolo_annotation(output_path, annotations, class_id):
    with open(output_path, 'w') as f:
        for ann in annotations:
            x_center, y_center, width, height = ann
            # Ensure the values are between 0 and 1
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            width = min(max(width, 0), 1)
            height = min(max(height, 0), 1)
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def printSymbolBasedOnProgress(symbolToPrint, progress, totalPossibleProgress):
    # Progressively change the color of the printed period as frame_index approaches 200
    progress = (progress % totalPossibleProgress) / totalPossibleProgress  # Normalize progress between 0 and 1
    red = int(255 * progress)  # Red increases with progress
    green = int(255 * (1 - progress))  # Green decreases with progress

    # Generate ANSI escape code for the color
    color_code = f"\033[38;2;{red};{green};0m"

    # Print the colored period
    print(f'{color_code}.', end='', flush=True)

    # Reset the color back to default after printing
    print("\033[0m", end='', flush=True)


# ------------------------------------------------------------
# createDatabaseTrainingRun
# ------------------------------------------------------------
# Sends a POST request to the backend API to create a new
# training run record in the database.
#
# Parameters:
#   training_run_record (dict): Dictionary containing all fields
#       required by the 'training_runs' table schema.
#
# Returns:
#   dict: The JSON response containing the created training run.
# ------------------------------------------------------------
def createDatabaseTrainingRun(training_run_record):
    try:
        url = f"{api_base_url}/training_run"
        response = requests.post(url, json={"training_run": training_run_record})
        response.raise_for_status()
        print("[DB] Created new training_run entry")
        return response.json()
    except Exception as e:
        print(f"[ERROR] createDatabaseTrainingRun(): {e}")
        return {}
    

# ------------------------------------------------------------
# updateDatabaseTrainingRun
# ------------------------------------------------------------
# Sends a PUT request to the backend API to update an existing
# training run record in the database.
#
# Parameters:
#   training_run_id (int): ID of the training run to update.
#   update_training_run_data (dict): Key-value pairs of fields
#       to be updated.
#
# Returns:
#   dict: The JSON response with the updated record (if successful).
# ------------------------------------------------------------
def updateDatabaseTrainingRun(training_run_id, update_training_run_data):
    try:
        url = f"{api_base_url}/training_run/{training_run_id}"
        response = requests.put(url, json={"training_run": update_training_run_data})
        response.raise_for_status()
        print(f"[DB] Updated training_run id={training_run_id}")
        return response.json()
    except Exception as e:
        print(f"[ERROR] updateDatabaseTrainingRun(): {e}")
        return {}

# ------------------------------------------------------------
# createDatabaseEpoch
# ------------------------------------------------------------
# Sends a POST request to the backend API to create a new epoch
# record for a specific training run.
#
# Parameters:
#   epoch_record (dict): Contains all fields from the epochs table.
#
# Returns:
#   dict: JSON response from the API.
# ------------------------------------------------------------
def createDatabaseEpoch(epoch_record):
    try:
        url = f"{api_base_url}/epoch"
        response = requests.post(url, json={"epoch": epoch_record})
        response.raise_for_status()
        print(f"[DB] Created new epoch entry for run_id={epoch_record.get('training_run_id')}")
        return response.json()
    except Exception as e:
        print(f"[ERROR] createDatabaseEpoch(): {e}")
        return {}
    
# ------------------------------------------------------------
# updateDatabaseEpoch
# ------------------------------------------------------------
# Sends a PUT request to the backend API to update an existing
# epoch record in the database.
#
# Parameters:
#   epoch_id (int): ID of the epoch to update.
#   update_epoch_data (dict): Fields to update.
#
# Returns:
#   dict: JSON response from the API.
# ------------------------------------------------------------
def updateDatabaseEpoch(epoch_id, update_epoch_data):
    try:
        url = f"{api_base_url}/epoch/{epoch_id}"
        response = requests.put(url, json={"epoch": update_epoch_data})
        response.raise_for_status()
        print(f"[DB] Updated epoch id={epoch_id}")
        return response.json()
    except Exception as e:
        print(f"[ERROR] updateDatabaseEpoch(): {e}")
        return {}
    

# ------------------------------------------------------------
# createDatabaseMetricsSummary
#   summary_record = {
#     "training_run_id": training_run_id,
#     "dataset_split": "val",
#     "precision": metrics.box.map(0),  # or metrics.results_dict["metrics/precision(B)"]
#     "recall": metrics.box.map(1),
#     "map50": metrics.box.map(2),
#     "map5095": metrics.box.map(3),
#     "f1_score": None,
#     "confusion_matrix_path": os.path.join(output_folder, new_model_name, "confusion_matrix.png"),
#     "result_plot_path": os.path.join(output_folder, new_model_name, "results.png"),
#     "timestamp": datetime.now().isoformat(),
# }
# ------------------------------------------------------------
def createDatabaseMetricsSummary(summary_record):
    try:
        url = f"{api_base_url}/metrics_summary"
        response = requests.post(url, json={"metrics_summary": summary_record})
        response.raise_for_status()
        print(f"[DB] Created metrics_summary split={summary_record.get('dataset_split')}")
        return response.json()
    except Exception as e:
        print(f"[ERROR] createDatabaseMetricsSummary(): {e}")
        return {}
    

# ------------------------------------------------------------
# createDatabaseMetricsCurve
# curve_record = {
#         "metrics_summary_id": summary_id,
#         "confidence_threshold": p["confidence"],
#         "precision": p["precision"],
#         "recall": p["recall"],
#         "f1_score": p["f1"],
#         "support": p.get("support", None),
#     }
# ------------------------------------------------------------
def createDatabaseMetricsCurve(curve_record):
    try:
        url = f"{api_base_url}/metrics_curve"
        response = requests.post(url, json={"metrics_curve": curve_record})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] createDatabaseMetricsCurve(): {e}")
        return {}