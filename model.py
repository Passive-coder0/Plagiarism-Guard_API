from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data once, outside the class
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class AIDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        # Load the fine-tuned RoBERTa model (updated path)
        self.model = RobertaForSequenceClassification.from_pretrained(
            "./fine_tuned_model",  # Ensure the fine-tuned model is loaded
            num_labels=2  # Ensure the model is trained for classification
        ).to(self.device)

        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.max_length = 512
        self.optimal_threshold = 0.5

    def detect(self, text):
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                ai_prob = probs[0][1].item()

            is_ai_generated = ai_prob >= self.optimal_threshold

            return {
                'ai_generated': is_ai_generated,
                'confidence': round(ai_prob * 100, 1),
                'threshold': self.optimal_threshold
            }

        except Exception as e:
            return {'error': str(e)}
