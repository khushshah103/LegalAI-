from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

class NERProcessor:
    def __init__(self, model_name="dslim/bert-base-NER"):
        print(f"Loading NER model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline(
            "ner", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            aggregation_strategy="simple"
        )

    def extract_entities(self, text):
        """
        Extracts entities from the given text.
        """
        if not text:
            return []
        
        # BERT has a 512 token limit. Simple truncation for now.
        # In Phase 3, we will handle longer context with chunking.
        results = self.ner_pipeline(text[:2000]) # Process a reasonable chunk for testing
        return results

    def format_entities(self, results):
        """
        Formats the NER results into a readable dictionary.
        """
        formatted = {}
        for entity in results:
            label = entity['entity_group']
            word = entity['word']
            if label not in formatted:
                formatted[label] = []
            if word not in formatted[label]:
                formatted[label].append(word)
        return formatted

if __name__ == "__main__":
    # Test text
    test_text = """
    This Agreement is made on this 25th day of October, 2023, by and between 
    Reliance Industries Limited, a company incorporated under the laws of India, 
    and Mr. John Doe, an individual residing in New York.
    The total consideration for the services shall be $5,000 payable within 30 days.
    """
    
    processor = NERProcessor()
    entities = processor.extract_entities(test_text)
    formatted = processor.format_entities(entities)
    
    print("\nExtracted Entities:")
    for label, words in formatted.items():
        print(f"{label}: {', '.join(words)}")
