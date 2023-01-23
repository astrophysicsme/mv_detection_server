from fastapi import FastAPI
import uvicorn

from configurations import cfg

from routers import pages


app = FastAPI()

app.version = cfg.SERVER.VERSION
app.title = cfg.SERVER.TITLE
app.debug = cfg.SERVER.DEBUG

app.include_router(pages.router)

# pyinstaller --noconfirm main.spec
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=cfg.SERVER.PORT)
