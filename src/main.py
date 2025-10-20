import datetime
from fastapi import FastAPI

app = FastAPI(
    title = "Banking System AI Microservice",
    version = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allowed_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

@app.except_handler(Exception)
async def global_exception_handler(request : req, exc):
    return JSONResponse(
        status_code = 5000,
        content = {
            "error" : exc,
            "status_code" : "5000",
            "timestamp" : datetime.now().isoformat(),
            "message" : "Internal server error"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host = "0.0.0.0",
        port = 8000,
        reload = True                                              
    )
