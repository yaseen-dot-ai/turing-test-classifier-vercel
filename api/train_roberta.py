import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class QuickDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        # HUMAN=0.0, AI=1.0, AMBIGUOUS=0.5
        self.targets = [0.0 if l=="HUMAN" else 1.0 if l=="AI" else 0.5 for l in labels]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten(),
            'labels': torch.tensor(self.targets[idx], dtype=torch.float32)
        }

def quick_train():
    # Load data
    with open('../experiments/krishna.json', 'r') as f:
        data = json.load(f)
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    
    # Load model
    from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
    from transformers import AutoTokenizer
    
    try:
        print("Loading model from ./quick-tuned-roberta..")
        model = RobertaClassifier.from_pretrained(
            "./quick-tuned-roberta"
        )
    except:
        print("Loading model from SuperAnnotate/roberta-large-llm-content-detector..")
        model = RobertaClassifier.from_pretrained(
            "SuperAnnotate/roberta-large-llm-content-detector"
        )
    tokenizer = AutoTokenizer.from_pretrained("SuperAnnotate/roberta-large-llm-content-detector")
    
    # Quick setup
    dataset = QuickDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    
    # Train for 2 epochs (should be enough for 333 samples)
    for epoch in range(2):
        for batch in tqdm(dataloader, desc="Training epoch %d" % epoch):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            targets = batch['labels'].to(device)
            
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            probs = torch.sigmoid(logits).squeeze(1)
            
            loss = criterion(probs, targets)
            loss.backward()
            optimizer.step()
    
    # Save
    model.save_pretrained("./quick-tuned-roberta")
    tokenizer.save_pretrained("./quick-tuned-roberta")
    print("Done! Model saved.")

if __name__ == "__main__":
    quick_train()