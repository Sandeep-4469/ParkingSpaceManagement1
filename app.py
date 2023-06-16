import uvicorn
from fastapi import FastAPI
from fastapi_sqlalchemy import DBSessionMiddleware, db
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import os
from dotenv import load_dotenv

load_dotenv('.env')

app = FastAPI()
app.mount("/static", StaticFiles(directory = "static"), name = "static")


# @app.get("/")
# async def root():
#    return {"message": "hello world"}

@app.get("/")
def read_root():
    with open("templates/base.html", 'r') as file:
        content = file.read()
    return HTMLResponse(content=content)

# To run locally
if __name__ == '__main__':
   uvicorn.run(app, host='0.0.0.0', port=8000)