import uvicorn

from src.qca_replication.api import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
