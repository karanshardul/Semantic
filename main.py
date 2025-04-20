from fastapi import FastAPI, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import emoji
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from gensim.models import KeyedVectors
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
import textstat
from gensim.scripts.glove2word2vec import glove2word2vec
import os
import nltk
import logging
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Semantic Evaluation API",
    description="API for evaluating caption semantic similarity across different platforms",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- NLTK Resources Management ----------
def download_nltk_resources():
    """Download NLTK resources only if they don't exist already"""
    nltk_data_path = nltk.data.path[0]  # Default data path
    
    resources = {
        'stopwords': os.path.join(nltk_data_path, 'corpora', 'stopwords'),
        'wordnet': os.path.join(nltk_data_path, 'corpora', 'wordnet'),
        'punkt': os.path.join(nltk_data_path, 'tokenizers', 'punkt')
    }
    
    for resource, path in resources.items():
        if not os.path.exists(path):
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)
        else:
            logger.info(f"NLTK resource already exists: {resource}")

# Global variable to track if model is loaded
glove_model = None

def load_glove_model():
    """Load the GloVe model if it's not already loaded"""
    global glove_model
    
    if glove_model is None:
        # Check if word2vec format exists, if not convert from GloVe
        if not os.path.exists('glove.6B.50d.word2vec.txt'):
            if not os.path.exists('glove.6B.50d.txt'):
                raise FileNotFoundError("GloVe embeddings file 'glove.6B.50d.txt' not found")
            logger.info("Converting GloVe to Word2Vec format...")
            glove2word2vec('glove.6B.50d.txt', 'glove.6B.50d.word2vec.txt')
        
        logger.info("Loading GloVe model...")
        glove_model = KeyedVectors.load_word2vec_format('glove.6B.50d.word2vec.txt', binary=False)
        logger.info("GloVe model loaded successfully")
    
    return glove_model

# ----------------- SCORE SCALING (Final Adjustment) -----------------
def scale_score(score): 
    """Converts raw 0-1 score to 1-5 star rating scale"""
    return min(5.0, max(1.0, round(1.0 + score * 4, 1)))

# ----------------- CORE SIMILARITY METRICS -----------------
def text_to_vector(text):
    """Convert text to GloVe vector representation"""
    model = load_glove_model()
    words = [w for w in text.split() if w in model]
    return np.mean([model[w] for w in words], axis=0) if words else np.zeros(50)

def tfidf_similarity(t1, t2):
    """Calculate TF-IDF similarity between two texts"""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([t1, t2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def glove_similarity(t1, t2):
    """Calculate semantic similarity using GloVe embeddings"""
    return cosine_similarity([text_to_vector(t1)], [text_to_vector(t2)])[0][0]

def keyword_score(input_text, caption):
    """Calculate keyword match score with synonym detection"""
    try:
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
        vectorizer.fit([input_text])
        keywords = vectorizer.get_feature_names_out()[:5].tolist()
    except ValueError:
        return 0.0
    
    if not keywords:
        return 0.0
    
    caption_words = set(word_tokenize(caption.lower()))
    matches = sum(
        word in caption_words or
        any(lemma.name() in caption_words 
            for syn in wordnet.synsets(word) 
            for lemma in syn.lemmas())
        for word in keywords
    )
    return matches / len(keywords)

# ----------------- PLATFORM-SPECIFIC FEATURES -----------------
class PlatformEvaluator:
    def __init__(self, platform):
        self.platform = platform
        self.limits = {
            'twitter': lambda x: 1 - min(len(x)/280, 1),
            'medium': lambda x: min(len(word_tokenize(x))/500, 1)
        }
        
    def platform_features(self, caption):
        features = {
            'hashtag_count': len(re.findall(r'#\w+', caption)),
            'emoji_count': sum(1 for _ in emoji.analyze(caption)),
            'readability': textstat.flesch_reading_ease(caption)/100,
            'formality': 1 - TextBlob(caption).subjectivity
        }
        
        if self.platform in self.limits:
            features['length_score'] = self.limits[self.platform](caption)
        return features

# ----------------- SCORING ENGINE -----------------
PLATFORM_WEIGHTS = {
    'instagram': {
        'tfidf': 0.15, 'glove': 0.15, 'keywords': 0.15,
        'hashtag': 0.2, 'emoji': 0.15, 'readability': 0.1, 'sentiment': 0.1
    },
    'linkedin': {
        'tfidf': 0.2, 'glove': 0.2, 'keywords': 0.25,
        'hashtag': 0.05, 'formality': 0.2, 'readability': 0.1
    },
    'twitter': {
        'tfidf': 0.2, 'glove': 0.15, 'keywords': 0.15,
        'hashtag': 0.2, 'length': 0.2, 'readability': 0.1
    },
    'medium': {
        'tfidf': 0.25, 'glove': 0.25, 'keywords': 0.2,
        'length': 0.1, 'readability': 0.2
    }
}

def evaluate_for_platform(input_text, caption, platform='instagram'):
    """Evaluate caption for a given platform based on input text"""
    evaluator = PlatformEvaluator(platform)
    features = evaluator.platform_features(caption)
    
    metrics = {
        'tfidf': tfidf_similarity(input_text, caption),
        'glove': glove_similarity(input_text, caption),
        'keywords': keyword_score(input_text, caption),
        'sentiment': 1 - abs(TextBlob(input_text).sentiment.polarity - 
                             TextBlob(caption).sentiment.polarity)
    }
    
    total_score = sum(
        weight * features.get(factor, metrics.get(factor, 0))
        for factor, weight in PLATFORM_WEIGHTS[platform].items()
    )
    
    return scale_score(total_score)

# ----------------- API Models -----------------
class TextInput(BaseModel):
    input_text: str
    caption: str
    platforms: Optional[List[str]] = ["instagram", "linkedin", "twitter", "medium"]

class EvaluationResult(BaseModel):
    platform: str
    score: float

class EvaluationResponse(BaseModel):
    results: List[EvaluationResult]
    
# ----------------- API Endpoints -----------------
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("Starting the Semantic Evaluation API")
    # Download NLTK resources on startup
    download_nltk_resources()
    # Load GloVe model
    load_glove_model()
    logger.info("Initialization complete")

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "name": "Semantic Evaluation API",
        "version": "1.0.0",
        "description": "API for evaluating caption semantic similarity across different platforms"
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_caption(text_input: TextInput):
    """Evaluate caption against input text for specified platforms"""
    logger.info(f"Evaluating caption for platforms: {text_input.platforms}")
    
    results = []
    for platform in text_input.platforms:
        if platform not in PLATFORM_WEIGHTS:
            continue
        
        score = evaluate_for_platform(
            text_input.input_text, 
            text_input.caption, 
            platform
        )
        
        results.append(EvaluationResult(platform=platform, score=score))
    
    return EvaluationResponse(results=results)

# Example usage (for debugging purposes only)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)