import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dense_neural_class import Dense_Neural_Diy  

app = FastAPI()

def load_model(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, filename + '.pkl')
    
    with open(filepath, 'rb') as file:
        model_loaded = pickle.load(file)
    
    return model_loaded

model = load_model('model')

# Defining request model
class ImageVector(BaseModel):
    image_vector: list

# Defining post request
@app.post("/predict")
def predict(data: ImageVector):
    try:
        image_vector = np.array(data.image_vector).reshape(1, -1) # Convert list to NumPy array => flatten image
        result = model.predict(image_vector)[0] # Make prediction
        return {"prediction": int(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

