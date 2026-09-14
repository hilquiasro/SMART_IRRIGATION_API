from fastapi import FastAPI

app = FastAPI(
    title="Smart Irrigation API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "Smart Irrigation API funcionando!"
    }

@app.get("/ping")
def ping():
    return {
        "status": "ok"
    }