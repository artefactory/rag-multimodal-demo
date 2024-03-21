import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/")
ASSETS_PATH = Path(__file__).parent / "assets"

# If the API runs in admin mode, it will allow the creation of new users
ADMIN_MODE = bool(int(os.getenv("ADMIN_MODE", False)))
