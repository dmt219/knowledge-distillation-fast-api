from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from faker import Faker

app = FastAPI()
fake = Faker()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to your todo list."}

@app.post('/generate')
def generate():
    return { 'message': fake.paragraphs(nb=1) }