from hazm import *
import torch
from transformers import AutoTokenizer, AutoModel


class PersianNLPProcessor:
    def __init__(self, use_transformer=False):
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer()
        self.tagger = POSTagger(model='resources/pos_tagger.model')
        self.chunker = Chunker(model='resources/chunker.model')
        
        if use_transformer:
            self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
            self.model = AutoModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    
    def preprocess_text(self, text):
        text = self.normalizer.normalize(text)
        words = word_tokenize(text)
        pos_tags = self.tagger.tag(words)
        chunks = self.chunker.parse(pos_tags)
        
        return {
            'normalized_text': text,
            'tokens': words,
            'pos_tags': pos_tags,
            'chunks': chunks
        }
    
    def extract_rule_components(self, text):
        processed = self.preprocess_text(text)

        patterns = {
            'obligation': ['باید', 'ضروری', 'الزامی'],
            'prohibition': ['نباید', 'ممنوع', 'مجاز نیست'],
            'condition': ['اگر', 'چنانچه', 'در صورتی که'],
            'deadline': ['تا تاریخ', 'حداکثر', 'مهلت']
        }
        
        components = {
            'type': None,
            'subject': None,
            'action': None,
            'condition': None,
            'deadline': None
        }

        for token, tag in processed['pos_tags']:
            if tag.startswith('N') and not components['subject']:
                components['subject'] = token
            elif tag.startswith('V'):
                components['action'] = token

        for token in processed['tokens']:
            for rule_type, keywords in patterns.items():
                if any(keyword in token for keyword in keywords):
                    components['type'] = rule_type
                    break
        
        return components


class ParsBERTProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
        self.model = AutoModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")
        
    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
    
    def classify_rule_type(self, text):
        embeddings = self.get_embeddings(text)
        return embeddings


class HybridPersianProcessor:
    def __init__(self):
        self.hazm_processor = PersianNLPProcessor()
        self.bert_processor = ParsBERTProcessor()
        
        import stanza
        stanza.download('fa')
        self.nlp = stanza.Pipeline('fa')
    
    def process_rule(self, text):
        hazm_results = self.hazm_processor.preprocess_text(text)
        bert_embeddings = self.bert_processor.get_embeddings(text)
        doc = self.nlp(text)
        
        components = {
            'basic_analysis': hazm_results,
            'embeddings': bert_embeddings,
            'dependencies': [(word.text, word.head) for sent in doc.sentences for word in sent.words]
        }
        
        return components

def setup_nlp_processor():
    return HybridPersianProcessor()
