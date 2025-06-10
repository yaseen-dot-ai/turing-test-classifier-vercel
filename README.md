# Turing Test Classifier

AI vs Human text classification with explainable results using GPT-4.1, Claude 3.7 Sonnet, and RoBERTa.

## Setup

### 1. Install Dependencies

```bash
# Frontend
npm install

# Backend (in virtual environment)
pip install -r requirements.txt
```

### 2. Environment Variables

Create `.env` file:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 3. Development

```bash
# Start backend (terminal 1)
cd api
uvicorn index:app --port 8000 --reload

# Start frontend (terminal 2) 
npm run dev
```

Visit `http://localhost:3000`

## Deployment to Vercel

```bash
# Build and deploy
npm run build
vercel --prod
```

## API Endpoints

- `POST /api/predict` - Get predictions for text samples
- `POST /api/run` - Run full experiment with metrics

## Features

- **Predict Mode**: Quick text classification
- **Run Experiment**: Full evaluation with precision/recall metrics
- **Luxurious UI**: Dark theme with glassmorphism design
- **JSON Input**: Schema-validated input areas
- **Responsive Tables**: Beautiful result display 
