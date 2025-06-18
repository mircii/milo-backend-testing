from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ranked_responses import predict_intent_and_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

@app.post("/predict")
def predict(msg: Message):
    user_input = msg.message

    intent, response, confidence = predict_intent_and_response(user_input)

    print(f"\n\nUser Input: {user_input}")
    print(f"\nMilo confidence: {confidence}")
    print(f"\nPredicted tag: {intent}")
    print(f"\nMILO03: {response}\n\n")

    if isinstance(response, dict) and response.get("type") == "fallback_options":
        return {
            "type": "fallback",
            "tag": response["tag"],
            "message": response["message"],
            "options": response["options"]
        }

    return {"type": "text", "message": response}