import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

class ApplicantInput(BaseModel):
    experienceYears: float = Field(..., ge=0, le=10, description="Applicant experience years (0-10)")
    technicalScore: float = Field(..., ge=0, le=100, description="Applicant technical score (0-100)")

app = FastAPI(
    title="Hiring Prediction API",
    description="Predicts whether an applicant will be hired using an SVM model.",
    version="1.0.0"
)

# --- API Endpoints ---
@app.get("/", tags=["Welcome"])
def readRoot():
    return {"message": "Welcome to the Hiring Prediction API! POST to /predict for predictions."}

@app.post("/predict", tags=["Prediction"])
def predictApplicant(applicant):
    model = joblib.load("bestModel.pkl")
    scaler = joblib.load("scaler.pkl")
    try:
        inputData = pd.DataFrame([[applicant.experienceYears, applicant.technicalScore]],
                                 columns=['experienceYears', 'technicalScore'])
        inputScaled = scaler.transform(inputData)
        prediction = model.predict(inputScaled)
        probabilities = model.predict_proba(inputScaled)
        result = "Red" if int(prediction[0]) == 1 else "Kabul"
        probabilityNotHired = float(probabilities[0][1])
        probabilityHired = float(probabilities[0][0])
        return {
            "experienceYears": applicant.experienceYears,
            "technicalScore": applicant.technicalScore,
            "prediction": int(prediction[0]),
            "readableResult": result,
            "probabilityNotHired": round(probabilityNotHired, 2),
            "probabilityHired": round(probabilityHired, 2)
        }

    except Exception as predictionError:
        print(f"Error during prediction: {predictionError}")

if __name__ == "__main__":
    print("Access the API documentation at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000) 