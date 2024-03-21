"""Set up a FastAPI application with routes for different RAG models."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from hydra import compose, initialize
from langserve import add_routes

from backend import ENABLE_AUTHENTICATION
from backend.api_plugins import authentication_routes, session_routes
from backend.rag_1.chain import get_chain as get_chain_rag_1
from backend.rag_1.config import validate_config as validate_config_1
from backend.rag_2.chain import get_chain as get_chain_rag_2
from backend.rag_2.config import validate_config as validate_config_2
from backend.rag_3.chain import get_chain as get_chain_rag_3
from backend.rag_3.config import validate_config as validate_config_3

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs() -> RedirectResponse:
    """Redirect the root URL to the /docs endpoint."""
    return RedirectResponse("/docs")


with initialize(config_path="../backend/rag_1", version_base=None):
    config_1 = compose(config_name="config")
    print(config_1)

    # validate config
    _ = validate_config_1(config_1)

with initialize(config_path="../backend/rag_2", version_base=None):
    config_2 = compose(config_name="config")
    print(config_2)

    # validate config
    _ = validate_config_2(config_2)

with initialize(config_path="../backend/rag_3", version_base=None):
    config_3 = compose(config_name="config")
    print(config_3)

    # validate config
    _ = validate_config_3(config_3)

if ENABLE_AUTHENTICATION:
    auth = authentication_routes(app)
    session_routes(app, authentication=auth)
    dependencies = [auth]
else:
    dependencies = None

chain_rag_1 = get_chain_rag_1(config_1)
add_routes(app, chain_rag_1, path="/rag-1", dependencies=dependencies)

chain_rag_2 = get_chain_rag_2(config_2)
add_routes(app, chain_rag_2, path="/rag-2", dependencies=dependencies)

chain_rag_3 = get_chain_rag_3(config_3)
add_routes(app, chain_rag_3, path="/rag-3", dependencies=dependencies)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
