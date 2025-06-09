# Turing Test Classifier Service

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory with the following (fill in your keys):
   ```
   OPENAI_API_KEY=
   ANTHROPIC_API_KEY=
   S3_BUCKET=
   ```

3. Run the service:
   ```bash
   uvicorn main:app --reload
   ```

## Usage

POST to `/run` with JSON body:

```
{
  "samples": [
    {"text": "...", "label": "HUMAN"},
    ...
  ],
  "override_models": ["roberta"],
  "return_preds": true
}
``` 