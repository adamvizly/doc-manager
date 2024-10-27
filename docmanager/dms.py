from fastapi import FastAPI, HTTPException, FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Union
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from databases import Database
import contextlib
from fastapi.middleware.cors import CORSMiddleware
import re
import json
from .persian import setup_nlp_processor


app = FastAPI(
    title="Advanced Document Management System",
    description="System for managing and validating documents against rules"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = "sqlite:///./rules.db"
database = Database(DATABASE_URL)

nlp = setup_nlp_processor()

class Rule(BaseModel):
    id: Optional[int] = None
    content: str
    category: str
    components: Dict
    rule_type: str
    priority: int
    confidence: float
    created_at: datetime

class Document(BaseModel):
    content: str
    title: str
    type: str

class ValidationResult(BaseModel):
    is_valid: bool
    discrepancies: List[Dict]
    recommendations: List[str]

class TrainingData(BaseModel):
    examples: List[Dict[str, str]]

class AdvancedRuleExtractor:
    def __init__(self):
        self.nlp = nlp
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
        )
        self.classifier = RandomForestClassifier()
        self.is_trained = False
        
        self.rule_patterns = [
            r'باید\s+\w+',
            r'ضروری است\s+که',
            r'الزامی است\s+که',
            r'موظف است\s+که',
            r'نباید\s+\w+',
            r'ممنوع است',
            r'مجاز نیست',
            r'در صورتی که',
            r'چنانچه',
            r'اگر\s+\w+',
            r'تا تاریخ',
            r'حداکثر ظرف مدت',
            r'در مهلت'
        ]

    def train_with_examples(self, training_data: List[Dict[str, str]]):
        texts = [item['text'] for item in training_data]
        labels = [item['is_rule'] for item in training_data]
        
        X = self._create_features(texts)
        y = np.array(labels)
        
        self.classifier.fit(X, y)
        self.is_trained = True

    def _create_features(self, texts: List[str]) -> np.ndarray:
        tfidf_features = self.vectorizer.fit_transform(texts)
        pattern_features = self._extract_pattern_features(texts)
        return np.hstack([tfidf_features.toarray(), pattern_features])

    def _extract_pattern_features(self, texts: List[str]) -> np.ndarray:
        features = np.zeros((len(texts), len(self.rule_patterns) + 3))
        
        for i, text in enumerate(texts):
            for j, pattern in enumerate(self.rule_patterns):
                features[i, j] = len(re.findall(pattern, text))
            
            doc = self.nlp(text)
            features[i, -3] = len([t for t in doc if t.is_digit])
            features[i, -2] = len([t for t in doc if t.like_email])
            features[i, -1] = len([t for t in doc if t.like_url])
            
        return features

    def extract_rules_from_email(self, email_content: str) -> List[Dict]:
        if not self.is_trained:
            raise ValueError("Model must be trained before extraction")
        
        doc = self.nlp(email_content)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        X = self._create_features(sentences)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        rules = []
        for i, (sentence, is_rule, prob) in enumerate(zip(sentences, predictions, probabilities)):
            if is_rule:
                rule = self._process_rule(sentence, prob[1])
                if rule:
                    rules.append(rule)
        
        return rules

    def _process_rule(self, text: str, confidence: float) -> Dict:
        doc = self.nlp(text)
        
        components = {
            'subject': None,
            'action': None,
            'condition': None,
            'deadline': None
        }
        
        for token in doc:
            if token.dep_ == 'nsubj':
                components['subject'] = token.text
            elif token.dep_ == 'ROOT':
                components['action'] = token.text
        
        condition_patterns = [r'اگر\s*(.*?)\s*[،\.]', r'در صورتی که\s*(.*?)\s*[،\.]']
        for pattern in condition_patterns:
            matches = re.findall(pattern, text)
            if matches:
                components['condition'] = matches[0]
                break
        
        deadline_patterns = [r'تا تاریخ\s*(.*?)\s*[،\.]', r'حداکثر ظرف\s*(.*?)\s*[،\.]']
        for pattern in deadline_patterns:
            matches = re.findall(pattern, text)
            if matches:
                components['deadline'] = matches[0]
                break

        return {
            'content': text,
            'confidence': confidence,
            'components': components,
            'rule_type': self._determine_rule_type(text),
            'priority': self._determine_priority(text),
            'category': self._determine_category(text),
            'created_at': datetime.now()
        }

    def _determine_rule_type(self, text: str) -> str:
        if any(word in text.lower() for word in ['باید', 'ضروری', 'الزامی']):
            return 'obligation'
        elif any(word in text.lower() for word in ['نباید', 'ممنوع', 'مجاز نیست']):
            return 'prohibition'
        elif any(word in text.lower() for word in ['می‌تواند', 'مجاز است']):
            return 'permission'
        return 'information'

    def _determine_priority(self, text: str) -> int:
        high_priority = ['فوری', 'ضروری', 'مهم', 'حیاتی']
        medium_priority = ['توصیه', 'پیشنهاد', 'بهتر است']
        
        if any(word in text.lower() for word in high_priority):
            return 1
        elif any(word in text.lower() for word in medium_priority):
            return 2
        return 3

    def _determine_category(self, text: str) -> str:
        categories = {
            'financial': ['پرداخت', 'هزینه', 'مبلغ', 'ریال', 'تومان'],
            'deadline': ['مهلت', 'تاریخ', 'زمان', 'موعد'],
            'procedure': ['روش', 'فرآیند', 'مراحل', 'اقدام'],
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        return 'general'


class ComparisonEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000
        )

    def compare_document_with_rules(self, document: Document, rules: List[Rule]) -> ValidationResult:
        discrepancies = []
        recommendations = []

        doc_text = [document.content]
        rule_texts = [rule.content for rule in rules]
        all_texts = doc_text + rule_texts
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            doc_vector = tfidf_matrix[0:1]
            rule_vectors = tfidf_matrix[1:]

            similarities = cosine_similarity(doc_vector, rule_vectors)[0]

            for i, (rule, similarity) in enumerate(zip(rules, similarities)):
                if similarity < 0.3:
                    discrepancy = self._analyze_discrepancy(document, rule, similarity)
                    if discrepancy:
                        discrepancies.append(discrepancy)
                        recommendations.append(self._generate_recommendation(discrepancy))

        except Exception as e:
            print(f"Error in comparison: {str(e)}")
            discrepancies, recommendations = self._fallback_comparison(document, rules)

        return ValidationResult(
            is_valid=len(discrepancies) == 0,
            discrepancies=discrepancies,
            recommendations=recommendations
        )

    def _analyze_discrepancy(self, document: Document, rule: Rule, similarity: float) -> Dict:
        severity = "high" if similarity < 0.1 else "medium" if similarity < 0.2 else "low"
        
        return {
            "rule_id": rule.id,
            "rule_content": rule.content,
            "rule_type": rule.rule_type,
            "components": rule.components,
            "violation_type": "content_mismatch",
            "severity": severity,
            "similarity_score": float(similarity)
        }

    def _generate_recommendation(self, discrepancy: Dict) -> str:
        severity_messages = {
            "high": "نیاز به بازنگری کامل دارد",
            "medium": "نیاز به اصلاحات جزئی دارد",
            "low": "پیشنهاد می‌شود مجدداً بررسی شود"
        }
        
        component_messages = []
        for key, value in discrepancy['components'].items():
            if value:
                component_messages.append(f"{key}: {value}")

        return f"{severity_messages[discrepancy['severity']]} - {' - '.join(component_messages)}"

    def _fallback_comparison(self, document: Document, rules: List[Rule]) -> Tuple[List, List]:
        discrepancies = []
        recommendations = []
        
        for rule in rules:
            if rule.content.lower() not in document.content.lower():
                discrepancy = {
                    "rule_id": rule.id,
                    "rule_content": rule.content,
                    "rule_type": rule.rule_type,
                    "components": rule.components,
                    "violation_type": "missing_content",
                    "severity": "medium"
                }
                discrepancies.append(discrepancy)
                recommendations.append(f"محتوای سند با قانون {rule.id} مطابقت ندارد")
        
        return discrepancies, recommendations

rule_extractor = AdvancedRuleExtractor()
comparison_engine = ComparisonEngine()

async def init_db():
    await database.execute("""
        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            category TEXT,
            components TEXT,
            rule_type TEXT,
            priority INTEGER,
            confidence FLOAT,
            created_at TIMESTAMP
        )
    """)

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    await init_db()
    
    yield 
    
    await database.disconnect()


@app.post("/api/train", response_model=Dict[str, str])
async def train_system(training_data: TrainingData):
    """Train the rule extraction system"""
    try:
        rule_extractor.train_with_examples(training_data.examples)
        return {"status": "success", "message": "System trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rules/extract", response_model=Dict[str, Union[str, List[Dict]]])
async def extract_rules(email_content: str):
    """Extract rules from email content"""
    try:
        rules = rule_extractor.extract_rules_from_email(email_content)
        
        async with database.transaction():
            for rule in rules:
                query = """
                    INSERT INTO rules (content, category, components, rule_type, priority, confidence, created_at)
                    VALUES (:content, :category, :components, :rule_type, :priority, :confidence, :created_at)
                """
                values = {
                    **rule,
                    'components': json.dumps(rule['components'])
                }
                await database.execute(query=query, values=values)
            
        return {"status": "success", "rules": rules}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/validate", response_model=ValidationResult)
async def validate_document(document: Document):
    """Validate document against stored rules"""
    try:
        query = "SELECT * FROM rules"
        db_rules = await database.fetch_all(query=query)
        
        rules = [
            Rule(
                id=rule['id'],
                content=rule['content'],
                category=rule['category'],
                components=json.loads(rule['components']),
                rule_type=rule['rule_type'],
                priority=rule['priority'],
                confidence=rule['confidence'],
                created_at=rule['created_at']
            )
            for rule in db_rules
        ]
        
        validation_result = comparison_engine.compare_document_with_rules(document, rules)
        return validation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
