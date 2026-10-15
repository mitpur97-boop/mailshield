# 🛡️ MailShield — AI-Powered Spam Message Classifier

MailShield is a full-stack machine learning web application that classifies email and text messages as **SPAM** or **NOT SPAM** using a genuinely trained **Multinomial Naive Bayes** model with **TF-IDF** feature extraction.

> Built as a Machine Learning college project. No external AI APIs, no hardcoded rules — just real ML.

---

## ✨ Features

- **Real ML Prediction** — Messages are classified by a trained scikit-learn pipeline, not keyword rules
- **Confidence Scores** — Displays spam/ham probabilities from the model
- **Contributing Words** — Shows which words influenced the prediction using TF-IDF weights
- **Model Insights Dashboard** — Live accuracy, precision, recall, F1, and confusion matrix from the actual model
- **Educational Content** — Explains how TF-IDF and Naive Bayes work
- **Example Messages** — Click-to-try examples (spam, legitimate, promotional, ambiguous)
- **Responsive Design** — Works on desktop, tablet, and mobile
- **Privacy-First** — No messages are stored

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React, Vite, JavaScript, CSS, Lucide React |
| **Backend** | Python, FastAPI, Uvicorn |
| **ML** | scikit-learn, pandas, numpy, joblib |
| **Algorithm** | Multinomial Naive Bayes |
| **Features** | TF-IDF (Term Frequency–Inverse Document Frequency) |
| **Dataset** | UCI SMS Spam Collection |

---

## 🧠 ML Approach

### Pipeline

```
Raw Message → Text Preprocessing → TF-IDF Vectorization → Multinomial Naive Bayes → Prediction + Probability
```

### Text Preprocessing
- Convert to lowercase
- Normalize whitespace
- Preserve URLs, numbers, and special characters (useful spam indicators)
- Light-touch approach — let TF-IDF handle feature selection

### TF-IDF Configuration
- Max 5,000 features
- Unigrams and bigrams (1,2)
- English stop words removed
- Sublinear TF scaling
- Minimum document frequency of 2

### Classifier
- Multinomial Naive Bayes with alpha=0.1 (Laplace smoothing)
- Trained on 80% of the dataset, tested on 20%
- Stratified train/test split with random_state=42

---

## 📊 Dataset

**UCI SMS Spam Collection**
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- ~5,574 SMS messages
- Two classes: `ham` (legitimate) and `spam`
- Tab-separated format

The training script automatically downloads the dataset on first run.

---

## 📁 Project Structure

```
mailshield/
├── frontend/                  # React + Vite frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx     # Navigation header
│   │   │   ├── Hero.jsx       # Hero section
│   │   │   ├── Classifier.jsx # Main classifier UI
│   │   │   ├── HowItWorks.jsx # ML pipeline explanation
│   │   │   ├── ModelInsights.jsx # Metrics dashboard
│   │   │   └── Footer.jsx     # Footer
│   │   ├── services/
│   │   │   └── api.js         # Backend API client
│   │   ├── App.jsx            # Root component
│   │   ├── main.jsx           # Entry point
│   │   └── *.css              # Component styles
│   ├── .env.example           # Environment template
│   ├── package.json
│   └── index.html
│
├── backend/                   # FastAPI backend
│   ├── main.py                # API server (FastAPI)
│   ├── ml/
│   │   ├── train.py           # Model training script
│   │   ├── evaluate.py        # Model evaluation script
│   │   └── preprocess.py      # Text preprocessing
│   ├── model/
│   │   ├── spam_classifier.pkl    # Trained model (generated)
│   │   └── model_metadata.json   # Model metrics (generated)
│   ├── data/
│   │   └── SMSSpamCollection     # Dataset (auto-downloaded)
│   └── requirements.txt
│
├── README.md
└── .gitignore
```

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.9+**
- **Node.js 18+** and npm
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/mailshield.git
cd mailshield
```

### 2. Backend Setup

```bash
cd backend

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train the Model

```bash
# From the backend directory, with venv activated:
python ml/train.py
```

This will:
- Download the UCI SMS Spam Collection dataset (first time only)
- Preprocess the text
- Train the TF-IDF + Multinomial NB pipeline
- Evaluate on the test set and print metrics
- Save the model to `model/spam_classifier.pkl`
- Save metadata to `model/model_metadata.json`

### 4. Start the Backend

```bash
# From the backend directory:
python main.py
```

The API will be available at: **http://localhost:8000**

Swagger docs: **http://localhost:8000/docs**

### 5. Frontend Setup

Open a **new terminal**:

```bash
cd frontend

# Install dependencies
npm install

# Create environment file (or copy .env.example)
# .env should contain:
# VITE_API_URL=http://localhost:8000

# Start development server
npm run dev
```

The frontend will be available at: **http://localhost:5173**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message and endpoint list |
| `GET` | `/health` | Health check (model loaded status) |
| `GET` | `/model-info` | Model metadata and performance metrics |
| `POST` | `/predict` | Classify a message |

### POST /predict

**Request:**
```json
{
  "message": "Congratulations! You have won a prize..."
}
```

**Response:**
```json
{
  "prediction": "SPAM",
  "confidence": 0.9640,
  "spam_probability": 0.9640,
  "ham_probability": 0.0360,
  "message_length": 45,
  "top_contributing_words": [
    { "word": "congratulations", "contribution": 0.8234 },
    { "word": "won", "contribution": 0.6521 }
  ]
}
```

---

## 📈 Model Evaluation

Metrics are calculated from the actual trained model on the test set. Example results:

| Metric | Value |
|--------|-------|
| Accuracy | ~97–98% |
| Precision | ~99%+ |
| Recall | ~90–94% |
| F1 Score | ~95%+ |

Exact values depend on the train/test split and are displayed on the website from `/model-info`.

---

## 🌐 Deployment

### Vercel (Frontend)

1. Push the project to GitHub
2. Import the repository on [Vercel](https://vercel.com)
3. Set the **Root Directory** to `frontend`
4. Set the build command to `npm run build` and output to `dist`
5. Add environment variable: `VITE_API_URL=https://your-backend-url.com`

### Backend Deployment

The FastAPI backend can be deployed to:
- **Render** — Add a `render.yaml` or use their Python environment
- **Railway** — Connect the repo and set the start command
- **Any VPS** — Run with `uvicorn main:app --host 0.0.0.0 --port 8000`

Set the `CORS_ORIGINS` environment variable to your frontend's deployed URL.

---

## 🔒 Security & Privacy

- Input validated with Pydantic (max 10,000 characters)
- No user messages are stored or logged
- No external API calls for classification
- CORS restricted to configured origins
- No stack traces exposed to users

---

## 🗺️ Future Improvements

- [ ] Support multiple ML algorithms (SVM, Logistic Regression) for comparison
- [ ] Add message history (opt-in, local storage only)
- [ ] Batch classification (upload a CSV)
- [ ] More detailed feature importance visualization
- [ ] Dark mode
- [ ] Multilingual spam detection
- [ ] Docker containerization
- [ ] CI/CD pipeline

---

## 📄 License

This project is for educational purposes. The UCI SMS Spam Collection dataset is provided by the UCI Machine Learning Repository.

---

**Built with ❤️ using React, FastAPI, and scikit-learn**
