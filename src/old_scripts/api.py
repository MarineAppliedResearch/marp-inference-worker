"""
api.py
---------
Provides HTTP API wrappers for interacting with the backend server.

Implements:
- GET /api/user/:name
- (later) POST and PUT helpers
"""

from urllib.parse import quote
import requests


# Base URL of your server (adjust as needed)
BASE_URL = "http://192.168.1.203:3081"


def get_user(name: str) -> dict:
    """
    Perform a GET request to /api/user/:name

    Expected server response:
        [
            {
                "user_id": int,
                "name": str,
                "createdAt": str (ISO timestamp),
                "updatedAt": str (ISO timestamp)
            }
        ]

    - Automatically URL-encodes the name (so spaces work).
    - Ensures the response is a list with exactly one user.
    - Returns the user object dict, or {} on error.

    :param name: User's name string
    :return: dict with user fields, or {} if not found/error
    """
    safe_name = quote(name)  # Encode spaces/special characters safely
    url = f"{BASE_URL}/api/user/{safe_name}"

    try:
        print(f"[API] GET {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        if isinstance(data, list) and len(data) == 1:
            return data[0]
        else:
            print(f"[API WARN] Unexpected response format: {data}")
            return {}

    except requests.exceptions.RequestException as e:
        print(f"[API ERROR] Failed to GET user '{name}': {e}")
        return {}
    

def create_user_by_name(name: str) -> dict:
    """
    Perform a POST request to /api/user/createUserByName/:userName

    Expected server behavior:
      - Creates a new user with the given name
      - Returns a JSON object of the created user

    Example response:
        {
            "user_id": int,
            "name": str,
            "createdAt": str (ISO timestamp),
            "updatedAt": str (ISO timestamp)
        }

    :param name: User name string (will be URL-encoded)
    :return: dict with user fields, or {} on error
    """
    from urllib.parse import quote
    import requests

    safe_name = quote(name)  # Encode spaces/special characters
    url = f"{BASE_URL}/api/user/createUserByName/{safe_name}"

    try:
        print(f"[API] POST {url}")
        response = requests.post(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[API ERROR] Failed to create user '{name}': {e}")
        return {}
    

def create_observation(observation):
    """
    Send an observation (with optional keyframes) to the backend.
    
    :param observation: dict representing the observation, may include "keyframes"
    :return: the created observation (JSON) or None
    """
    url = f"{BASE_URL}/api/observation"
    payload = {"observation": observation}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"[API] Error creating observation: {e}")
        return None
    

def get_projects() -> list:
    """
    Perform a GET request to /api/projects

    Expected server behavior:
      - Returns a JSON array of project objects

    Example response:
        [
            {
                "project_id": 1,
                "name": "Project A",
                "createdAt": "2023-01-11T17:54:47.334Z",
                "updatedAt": "2023-01-11T17:54:47.334Z"
            },
            {
                "project_id": 2,
                "name": "Project B",
                "createdAt": "2023-01-12T10:22:11.123Z",
                "updatedAt": "2023-01-12T10:22:11.123Z"
            }
        ]

    :return: list of project dicts, or [] on error
    """
    import requests

    url = f"{BASE_URL}/api/projects"

    try:
        print(f"[API] GET {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        else:
            print(f"[API WARN] Unexpected response format: {data}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"[API ERROR] Failed to GET projects: {e}")
        return []
    

def get_sessions_by_user_and_project(user_id: int, project_id: int) -> list:
    """
    Perform a GET request to /api/sessions/user/:userID/project/:projectID

    Expected server behavior:
      - Returns a JSON array of session objects for the given user + project

    Example response:
        [
            {
                "session_id": 101,
                "user_id": 19,
                "project_id": 1,
                "dive": "D01",
                "line": "L01",
                "type": "fish",
                "createdAt": "2023-02-01T12:34:56.789Z",
                "updatedAt": "2023-02-01T12:34:56.789Z"
            },
            {
                "session_id": 102,
                "user_id": 19,
                "project_id": 1,
                "dive": "D02",
                "line": "L03",
                "type": "inverts",
                "createdAt": "2023-02-02T09:12:34.567Z",
                "updatedAt": "2023-02-02T09:12:34.567Z"
            }
        ]

    :param user_id: integer ID of the user
    :param project_id: integer ID of the project
    :return: list of session dicts, or [] on error
    """
    import requests

    url = f"{BASE_URL}/api/sessions/user/{user_id}/project/{project_id}"

    try:
        print(f"[API] GET {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            return data
        else:
            print(f"[API WARN] Unexpected response format: {data}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"[API ERROR] Failed to GET sessions for user={user_id}, project={project_id}: {e}")
        return []
    

def create_session(project_id: int, user_id: int, dive: str, line: str, type_: str) -> dict:
    """
    Perform a POST request to /api/session to create a new session.

    Required fields:
      - project_id: int
      - user_id: int
      - dive: str
      - line: str
      - lineId: str (auto-generated as f"{dive}_{line}")
      - type: str

    Example response (server-generated fields may vary):
        {
            "session_id": 123,
            "project_id": 58,
            "user_id": 19,
            "dive": "8",
            "line": "1000",
            "lineId": "8_1000",
            "type": "Invert",
            "createdAt": "2023-03-01T12:00:00.000Z",
            "updatedAt": "2023-03-01T12:00:00.000Z"
        }

    :return: dict with created session data, or {} on error
    """
    import requests

    url = f"{BASE_URL}/api/session"
    payload = {
        "session": {
            "project_id": project_id,
            "user_id": user_id,
            "dive": dive,
            "line": line,
            "lineId": f"{dive}_{line}",
            "type": type_
        }
    }

    try:
        print(f"[API] POST {url} payload={payload}")
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[API ERROR] Failed to create session: {e}")
        return {}
    

def get_last_video_info(session_id: int) -> dict:
    """
    Call GET /api/observation/getLastVideoInfo/:session_id

    Retrieves the record with the max observation_id for the given session.
    Response is expected to include:
        - videoLocation
        - mediaPosition
        - actualPosition

    :param session_id: The session ID to look up
    :return: dict containing last observation info, or {} if none/error
    """
    import requests

    url = f"{BASE_URL}/api/observation/getLastVideoInfo/{session_id}"
    try:
        print(f"[API] GET {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[API ERROR] Failed to get last video info for session {session_id}: {e}")
        return {}


if __name__ == "__main__":
    user = get_user("Isaac Travers")
    print("User response:", user)