from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def index():
    return {"message": "Hello World"}
    
@app.get("/cool")
def cool_stuff():
    return {"message": "This is neat"}

@app.post("/echo")
def echo(message: str):
    return {"echo": message}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)