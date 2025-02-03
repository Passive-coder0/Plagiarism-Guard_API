from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class AIDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=2,
            hidden_dropout_prob=0.7,
            attention_probs_dropout_prob=0.5
        ).to(self.device)
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.max_length = 512
        self.optimal_threshold = 0.5

    def calculate_metrics(self, text):
        # Calculate similar words score (example implementation)
        words = word_tokenize(text.lower())
        unique_words = len(set(words))
        total_words = len(words)
        similar_words_score = (unique_words / total_words) * 100 if total_words > 0 else 0

        # Calculate sources attribution score (example implementation)
        # Here you might want to implement actual source checking logic
        sources_score = min(len([w for w in words if w in ['according', 'stated', 'cited']]) * 20, 100)

        # Calculate citations score (example implementation)
        citations_score = min(text.count('(') * 20, 100)  # Simple citation count

        # Calculate plagiarism score
        plagiarism_score = 100 - ((similar_words_score + sources_score + citations_score) / 3)

        return {
            'similar_words_score': round(similar_words_score, 1),
            'sources_score': round(sources_score, 1),
            'citations_score': round(citations_score, 1),
            'plagiarism_score': round(plagiarism_score, 1)
        }

    def generate_feedback(self, metrics, is_ai_generated):
        feedback = []
        
        if metrics['plagiarism_score'] > 30:
            feedback.append("Your essay shows significant similarity to existing content. Try to express ideas more originally.")
        
        if metrics['similar_words_score'] < 70:
            feedback.append("Consider using more diverse vocabulary to enhance your writing.")
        
        if metrics['sources_score'] < 50:
            feedback.append("Include more source attributions to strengthen your arguments.")
        
        if metrics['citations_score'] < 50:
            feedback.append("Add more citations to support your claims and improve credibility.")
        
        if is_ai_generated:
            feedback.append("The text shows patterns typical of AI-generated content. Try to make your writing more personal and authentic.")

        if not feedback:
            feedback.append("Good work! Your essay shows originality and proper academic writing practices.")

        return " ".join(feedback)

    def detect(self, text):
        try:
            # AI detection
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

            # Calculate additional metrics
            metrics = self.calculate_metrics(text)
            
            # Generate feedback
            feedback = self.generate_feedback(metrics, is_ai_generated)

            return {
                'plagiarism_percentage': f"{metrics['plagiarism_score']}%",
                'ai_generated': is_ai_generated,
                'confidence': round(ai_prob * 100, 1),
                'sources_attribution': metrics['sources_score'],
                'similar_words': metrics['similar_words_score'],
                'citations': metrics['citations_score'],
                'feedback': feedback,
                'threshold': self.optimal_threshold
            }

        except Exception as e:
            return {
                'error': str(e),
                'plagiarism_percentage': "0%",
                'ai_generated': None,
                'confidence': None,
                'sources_attribution': 0,
                'similar_words': 0,
                'citations': 0,
                'feedback': "Error analyzing text"
            }