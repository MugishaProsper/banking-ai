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