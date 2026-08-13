from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd

from schemas import RiskPredictionInput, RiskPredictionOutput, ChatInput, ChatOutput, ImagePredictionOutput

app = FastAPI(title="Coral Reef Health API")

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load ML model (Random Forest) ----------
ml_model = joblib.load('../models/bleaching_risk_model.pkl')

# ---------- Load DL model (CNN) ----------
dl_model = models.mobilenet_v2(weights=None)
num_features = dl_model.classifier[1].in_features
dl_model.classifier[1] = nn.Linear(num_features, 2)
dl_model.load_state_dict(torch.load('../models/best_coral_model.pth', map_location='cpu'))
dl_model.eval()

dl_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class_names = ['bleached_corals', 'healthy_corals']

# ---------- Root and health check ----------
@app.get("/")
def root():
    return {"message": "Coral Reef Health API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Image prediction endpoint ----------
@app.post("/predict/image", response_model=ImagePredictionOutput)
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        tensor = dl_transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = dl_model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        return ImagePredictionOutput(
            prediction=class_names[predicted.item()],
            confidence=round(confidence.item() * 100, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # ---------- Load RAG components ----------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = Chroma(persist_directory='../data/chroma_db', embedding_function=embeddings)
llm = OllamaLLM(model="llama3.2")

prompt_template = """Use the following context about coral reefs to answer the question. If the answer isn't in the context, say you don't have enough information rather than guessing.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# ---------- Risk prediction endpoint ----------
FEATURE_COLUMNS = ml_model.feature_names_in_.tolist()

@app.post("/predict/risk", response_model=RiskPredictionOutput)
def predict_risk(data: RiskPredictionInput):
    try:
        input_dict = data.dict()
        row = pd.DataFrame([input_dict])

        # One-hot encode categorical fields to match training format
        row_encoded = pd.get_dummies(row, columns=['Ocean_Name', 'Exposure', 'Realm_Name', 'Country_Name'])

        # Align columns with what the model expects (fill missing with 0)
        for col in FEATURE_COLUMNS:
            if col not in row_encoded.columns:
                row_encoded[col] = 0
        row_encoded = row_encoded[FEATURE_COLUMNS]

        prediction = ml_model.predict(row_encoded)[0]
        probabilities = ml_model.predict_proba(row_encoded)[0]
        confidence = max(probabilities) * 100

        return RiskPredictionOutput(
            predicted_category=prediction,
            confidence=round(confidence, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------- Chat endpoint ----------
@app.post("/chat", response_model=ChatOutput)
def chat(data: ChatInput):
    try:
        result = qa_chain.invoke({"query": data.question})
        sources = list(set([doc.metadata['source'] for doc in result['source_documents']]))
        return ChatOutput(answer=result['result'], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))