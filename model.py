import torch
import re
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

class AIDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=2
        ).to(self.device)
        self.threshold = 0.8

    def calculate_readability_score(self, text):
        words = text.split()
        sentences = text.split('.')
        if len(sentences) == 0:
            return 0, ""
        
        avg_words_per_sentence = len(words) / len(sentences)
        readability_score = min(100, max(0, 100 - (abs(avg_words_per_sentence - 15) * 3)))
        
        feedback = ""
        if avg_words_per_sentence > 25:
            feedback = "Your sentences are quite long. Consider breaking them down for better readability. "
        elif avg_words_per_sentence < 10:
            feedback = "Your sentences are very short. Try combining some for better flow. "
            
        return round(readability_score), feedback

    def calculate_ai_generated_score(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            ai_score = probs[0][1].item() * 100
        
        feedback = ""
        if ai_score > 70:
            feedback = "Your text shows strong indicators of AI generation. Try adding more personal experiences and unique perspectives. "
        elif ai_score > 40:
            feedback = "Consider adding more human elements to your writing by including personal anecdotes or unique viewpoints. "
            
        return round(ai_score), feedback

    def check_sources_attribution(self, text):
        source_patterns = [r'according to', r'cited by', r'referenced in', r'source:', r'as stated in']
        score = sum(20 for pattern in source_patterns if re.search(pattern, text.lower()))
        
        feedback = ""
        if score < 40:
            feedback = "Your text lacks proper source attribution. Try incorporating more references to support your arguments. "
            
        return min(100, score), feedback

    def check_citations(self, text):
        citation_patterns = [r'$$\w+,?\s*\d{4}$$', r'$$\d+$$', r'\d+\.\s*\w+', r'et al\.', r'$$\d{4}$$']
        score = sum(20 for pattern in citation_patterns if re.search(pattern, text))
        
        feedback = ""
        if score < 40:
            feedback = "Your text needs more citations. Remember to cite sources when presenting facts or statistics. "
            
        return min(100, score), feedback

    def detect(self, text):
        try:
            readability_score, readability_feedback = self.calculate_readability_score(text)
            ai_score, ai_feedback = self.calculate_ai_generated_score(text)
            sources_score, sources_feedback = self.check_sources_attribution(text)
            citations_score, citations_feedback = self.check_citations(text)

            overall_feedback = (
                f"Analysis Results:\n\n"
                f"Your text received the following scores:\n"
                f"- Readability: {readability_score}/100\n"
                f"- AI Generated: {ai_score}/100\n"
                f"- Sources Attribution: {sources_score}/100\n"
                f"- Citations: {citations_score}/100\n\n"
                f"Recommendations:\n"
                f"{readability_feedback}"
                f"{ai_feedback}"
                f"{sources_feedback}"
                f"{citations_feedback}\n"
                f"Overall: {self.generate_overall_recommendation(readability_score, ai_score, sources_score, citations_score)}"
            )

            return {
                'plagiarism_percentage': f"{readability_score}%",
                'ai_generated': ai_score > 50,
                'sources_attribution': sources_score,
                'similar_words': readability_score,
                'citations': citations_score,
                'feedback': overall_feedback
            }

        except Exception as e:
            return {'error': str(e)}

    def generate_overall_recommendation(self, readability, ai_score, sources, citations):
        overall_score = (readability + (100 - ai_score) + sources + citations) / 4
        
        if overall_score < 50:
            return "Your content needs significant improvement. Focus on making your writing more original, well-supported, and easier to read."
        elif overall_score < 70:
            return "Your content shows promise but needs refinement. Consider implementing the suggestions above to strengthen your writing."
        else:
            return "Your content is generally strong, though there's still room for improvement in specific areas mentioned above."
        
        
