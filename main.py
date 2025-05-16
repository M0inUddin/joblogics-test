from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse

from utils.logger import logger

from service.agent import call_graphql_api

app = FastAPI()
origins = [
    "http://localhost:3000",
    "https://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>FastAPI CORS Example</title>
        </head>
        <body>
            <h1>FastAPI CORS Example</h1>
            <p>This is a simple FastAPI application with CORS enabled.</p>
        </body>
    </html>
    """


@app.post("/query")
async def query(query: str):
    try:
        logger.info(f"Received query")
        logger.info(f"Query: {query}")
        # Process the query and return a response
        response = call_graphql_api(query)
        logger.info(f"Response: {response}")
        return JSONResponse(
            content=jsonable_encoder(
                {"message": "Query received", "query": query, "response": response}
            ),
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return JSONResponse(
            content=jsonable_encoder({"error": "Failed to process query"}),
            status_code=500,
        )


import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
