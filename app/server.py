from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from backend.rag_3.chain import get_chain as get_chain_rag_3
from hydra import initialize, compose

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


with initialize(config_path="../backend/rag_3", version_base=None):
    config_3 = compose(config_name="config")
    print(config_3)

chain_rag_3 = get_chain_rag_3(config_3)
add_routes(app, chain_rag_3, path="/rag-3")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
