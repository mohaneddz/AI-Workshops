from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION, SERVER_HOST, SERVER_PORT, STATIC_DIR
from app.routes.web import router


def create_app() -> FastAPI:
    app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
