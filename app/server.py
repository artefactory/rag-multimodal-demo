from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from hydra import compose, initialize
from langserve import add_routes
from omegaconf import OmegaConf

from backend.rag_3.chain import get_chain as get_chain_rag_3
from backend.rag_3.config import Config as Config_3

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


with initialize(config_path="../backend/rag_3", version_base=None):
    config_3 = compose(config_name="config")
    print(config_3)

    # validate config
    cfg_obj = OmegaConf.to_object(config_3)
    _ = Config_3(**cfg_obj)

chain_rag_3 = get_chain_rag_3(config_3)
add_routes(app, chain_rag_3, path="/rag-3")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
