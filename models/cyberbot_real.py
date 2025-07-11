import os

import json
import time
import logging
import random
import asyncio
import fastapi
import uvicorn
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session, relationship
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# HuggingFace Vietnamese Models
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration, pipeline, BitsAndBytesConfig
)
import torch
from llama_cpp import Llama

# --- Vector DB Cloud (Pinecone) ---
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# --- Knowledge Graph (Neo4j) ---
from neo4j import GraphDatabase

# --- Telegram Alert ---
import requests

# ========== Load ENV ==========
load_dotenv()

# ========== Database ==========
# ========== Database ==========
class Base(DeclarativeBase):
    pass
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship("ChatSession", back_populates="user")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    messages_count = Column(Integer, default=0)
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)
    session = relationship("ChatSession", back_populates="messages")

class UserBehavior(Base):
    __tablename__ = "user_behaviors"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    pattern = Column(JSON)
    last_updated = Column(DateTime, default=datetime.utcnow)

def create_db_engine(db_url: str):
    return create_engine(db_url)

def init_db(db_url: str):
    engine = create_db_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

# ========== Config ==========
class Config:
    def __init__(self):
        self.API_PORT = int(os.getenv("API_PORT", 8000))
        self.DB_URL = os.getenv("DATABASE_URL", "sqlite:///cyberbot.db")
        
        # HuggingFace Vietnamese Model Configuration - AWS G4dn.xlarge optimized
        self.USE_HUGGINGFACE = os.getenv("USE_HUGGINGFACE", "true").lower() == "true"
        
        # GPU Model Selection for G4dn.xlarge (T4 GPU, 16GB VRAM)
        self.HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "VietAI/vit5-large")  # Larger model for GPU
        self.HF_MODEL_TYPE = os.getenv("HF_MODEL_TYPE", "t5")  # t5, gpt, bert
        self.HF_USE_GPU = os.getenv("HF_USE_GPU", "true").lower() == "true"  # G4dn.xlarge has T4 GPU
        
        # Enhanced generation parameters for GPU performance
        self.HF_MAX_LENGTH = int(os.getenv("HF_MAX_LENGTH", 512))  # Much higher for GPU
        self.HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", 256))
        self.HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", 0.8))
        self.HF_TOP_P = float(os.getenv("HF_TOP_P", 0.9))
        self.HF_TOP_K = int(os.getenv("HF_TOP_K", 50))
        self.HF_DO_SAMPLE = os.getenv("HF_DO_SAMPLE", "true").lower() == "true"
        self.HF_NUM_BEAMS = int(os.getenv("HF_NUM_BEAMS", 4))  # Beam search for quality
        
        # AWS G4dn.xlarge specific optimizations (T4 GPU, 16GB VRAM, 16GB RAM, 4 vCPUs)
        self.HF_BATCH_SIZE = int(os.getenv("HF_BATCH_SIZE", 8))  # Higher batch for GPU
        self.HF_NUM_THREADS = int(os.getenv("HF_NUM_THREADS", 4))  # Match vCPU count
        self.HF_INTRAOP_THREADS = int(os.getenv("HF_INTRAOP_THREADS", 4))
        self.HF_INTEROP_THREADS = int(os.getenv("HF_INTEROP_THREADS", 2))
        self.MAX_MEMORY_USAGE = float(os.getenv("MAX_MEMORY_USAGE", 12.0))  # Leave 4GB for OS
        self.MAX_VRAM_USAGE = float(os.getenv("MAX_VRAM_USAGE", 14.0))  # Leave 2GB VRAM buffer
        
        # Multi-model support for advanced chatbot
        self.CONVERSATION_MODEL = os.getenv("CONVERSATION_MODEL", "microsoft/DialoGPT-medium")
        self.EMOTION_MODEL = os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
        self.SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        # Advanced features for customer service
        self.ENABLE_EMOTION_DETECTION = os.getenv("ENABLE_EMOTION_DETECTION", "true").lower() == "true"
        self.ENABLE_SENTIMENT_ANALYSIS = os.getenv("ENABLE_SENTIMENT_ANALYSIS", "true").lower() == "true"
        self.ENABLE_MULTI_MODEL_ENSEMBLE = os.getenv("ENABLE_MULTI_MODEL_ENSEMBLE", "true").lower() == "true"
        
        # Legacy Llama-cpp Configuration (fallback)
        self.MODEL_CONFIG = {
            'path': os.getenv("MODEL_PATH", "PhoGPT-4B-Chat-Q8_0.gguf"),
            'params': {
                'temperature': float(os.getenv("MODEL_TEMPERATURE", 0.7)),
                'max_length': int(os.getenv("MODEL_MAX_LENGTH", 80)),
                'n_ctx': int(os.getenv("MODEL_CTX_SIZE", 512)),
            }
        }
        
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chat-memory")
        self.NEO4J_URI = os.getenv("NEO4J_URI")
        self.NEO4J_USER = os.getenv("NEO4J_USER")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.TELEGRAM_CHAT_IDS = [x for x in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if x]
        self.API_KEY = os.getenv("API_KEY", "")

config = Config()
SessionLocal = init_db(config.DB_URL)
# ========== Pinecone Setup ==========
pc = None
pinecone_index = None
embedding_model = None

try:
    if config.PINECONE_API_KEY:
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        
        # Check existing indexes first
        existing_indexes = pc.list_indexes().names()
        if config.PINECONE_INDEX_NAME not in existing_indexes:
            print(f"🔧 Creating new Pinecone index: {config.PINECONE_INDEX_NAME}")
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Wait for index to be ready
            import time
            time.sleep(10)
        
        pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)
        
        # Load embedding model with error handling
        try:
            embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("✅ Pinecone and embedding model initialized successfully")
        except Exception as emb_e:
            print(f"⚠️  Embedding model failed, using alternative: {emb_e}")
            embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    else:
        print("⚠️  PINECONE_API_KEY not configured")
        
except Exception as e:
    print(f"⚠️  Pinecone initialization failed: {e}")
    print("🔧 CyberBot will run without vector memory until Pinecone is configured")
    pc = None
    pinecone_index = None
    embedding_model = None

def add_to_vector_memory_cloud(user_id, message_id, text):
    if not pinecone_index or not embedding_model:
        print("⚠️  Pinecone not available, skipping vector memory storage")
        return
    vector = embedding_model.encode([text])[0].tolist()
    pinecone_index.upsert([(f"{user_id}_{message_id}", vector, {"user_id": str(user_id), "text": text})])

def retrieve_context_cloud(user_id, query, top_k=3):
    if not pinecone_index or not embedding_model:
        print("⚠️  Pinecone not available, returning empty context")
        return []
    vector = embedding_model.encode([query])[0].tolist()
    res = pinecone_index.query(vector, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in res["matches"] if match["metadata"].get("user_id") == str(user_id)]

# ========== Neo4j Setup ==========
neo4j_driver = None

try:
    if config.NEO4J_URI and config.NEO4J_USER and config.NEO4J_PASSWORD:
        neo4j_driver = GraphDatabase.driver(
            config.NEO4J_URI, 
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60
        )
        # Test connection
        with neo4j_driver.session() as session:
            session.run("RETURN 1")
        print("✅ Neo4j connected successfully")
    else:
        print("⚠️  Neo4j credentials not fully configured")
except Exception as e:
    print(f"⚠️  Neo4j connection failed: {e}")
    print("🔧 CyberBot will run without knowledge graph until Neo4j is configured")
    neo4j_driver = None

def query_kg_facts(entity):
    if not neo4j_driver:
        return []
    try:
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (e1)-[r]->(e2) WHERE e1.name = $entity RETURN e1.name, type(r), e2.name LIMIT 5",
                entity=entity
            )
            return [(rec["e1.name"], rec["type(r)"], rec["e2.name"]) for rec in result]
    except Exception as e:
        print(f"⚠️  Neo4j query failed: {e}")
        return []

# ========== Function Calling ==========
function_schemas = [
    {
        "name": "get_order_status",
        "description": "Lấy trạng thái đơn hàng",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Mã đơn hàng khách cung cấp"}
            },
            "required": ["order_id"]
        }
    }
]

async def handle_function_call(function_name, arguments):
    if function_name == "get_order_status":
        return f"Đơn hàng {arguments['order_id']} đang trên đường giao."
    return "Function chưa được hỗ trợ."

def parse_function_call_openai(response_text):
    try:
        call = json.loads(response_text)
        if "function_call" in call:
            return call["function_call"]["name"], call["function_call"]["arguments"]
    except Exception:
        pass
    return None, None

# ========== Telegram Alert ==========
def send_telegram_alert(title, message):
    for chat_id in config.TELEGRAM_CHAT_IDS:
        try:
            requests.post(
                f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": f"{title}\n\n{message}",
                    "parse_mode": "HTML"
                }
            )
        except Exception as e:
            logging.error(f"Failed to send Telegram alert: {e}")

# ========== Behavior Tracking ==========
import re
from collections import Counter

def extract_behavior_features(message: str) -> dict:
    """Advanced behavior extraction for customer service AI with personality learning"""
    
    # Customer service specific patterns
    urgency_keywords = ["gấp", "khẩn cấp", "nhanh", "urgent", "asap", "help", "giúp đỡ"]
    complaint_keywords = ["lỗi", "hỏng", "không được", "tệ", "khiếu nại", "complain", "sai"]
    praise_keywords = ["tốt", "hay", "tuyệt", "xuất sắc", "perfect", "excellent", "cảm ơn"]
    question_keywords = ["sao", "thế nào", "như thế", "how", "what", "when", "where"]
    
    # Communication style analysis
    formal_indicators = ["dạ", "ạ", "kính", "quý", "xin", "please", "anh", "chị", "em"]
    casual_indicators = ["ok", "okay", "hi", "hello", "hey", "yo", "bro", "sis"]
    
    # Emotional state detection
    positive_emotions = ["vui", "hạnh phúc", "thích", "yêu", "happy", "love", "good", "great"]
    negative_emotions = ["buồn", "tức", "khó chịu", "stress", "tired", "angry", "sad", "bad"]
    neutral_emotions = ["bình thường", "ok", "được", "fine", "normal"]
    
    # Customer type patterns
    detailed_customer = len(message) > 100 and message.count('.') > 2
    quick_customer = len(message) < 30 and not message.count('?') > 1
    
    # Technical proficiency indicators
    tech_words = ["app", "software", "bug", "feature", "update", "api", "server", "database"]
    basic_words = ["không biết", "không hiểu", "làm sao", "help me", "giúp em"]
    
    # Personality traits analysis
    personality_traits = {
        "patient": any(word in message.lower() for word in ["từ từ", "chậm", "không sao", "take time"]),
        "impatient": any(word in message.lower() for word in urgency_keywords),
        "detail_oriented": detailed_customer,
        "direct": quick_customer,
        "tech_savvy": any(word in message.lower() for word in tech_words),
        "needs_guidance": any(word in message.lower() for word in basic_words),
        "polite": len([w for w in formal_indicators if w in message.lower()]) > 1,
        "casual": len([w for w in casual_indicators if w in message.lower()]) > 0,
    }
    
    return {
        # Basic features
        "message_length": len(message),
        "word_count": len(message.split()),
        "sentence_count": len([s for s in message.split('.') if s.strip()]),
        
        # Customer service patterns
        "urgency_level": len([w for w in urgency_keywords if w in message.lower()]),
        "complaint_indicators": len([w for w in complaint_keywords if w in message.lower()]),
        "praise_indicators": len([w for w in praise_keywords if w in message.lower()]),
        "question_count": message.count('?'),
        
        # Communication style
        "formality_score": len([w for w in formal_indicators if w in message.lower()]),
        "casualness_score": len([w for w in casual_indicators if w in message.lower()]),
        
        # Emotional state
        "positive_emotion": len([w for w in positive_emotions if w in message.lower()]),
        "negative_emotion": len([w for w in negative_emotions if w in message.lower()]),
        "neutral_emotion": len([w for w in neutral_emotions if w in message.lower()]),
        
        # Customer characteristics
        "tech_proficiency": len([w for w in tech_words if w in message.lower()]) - len([w for w in basic_words if w in message.lower()]),
        "detail_preference": 1 if detailed_customer else 0,
        "communication_brevity": 1 if quick_customer else 0,
        
        # Personality traits
        "personality_traits": personality_traits,
        
        # Legacy compatibility
        "emojis": re.findall(r'[^\w\s.,!?]', message),
        "keywords": [kw for kw in ["dạ", "ạ", "ok", "uhm", ":3", "^^", "thanks", "thks", "cảm ơn", "chị", "anh", "em", "boss"] if kw in message.lower()],
        "typo_count": sum(len(re.findall(p, message, flags=re.IGNORECASE)) for p in [r"ko\b", r"\bk\b", r"dc\b", r"bit\b", r"z\b", r"r\b"]),
        "ending": re.search(r'([.!?…]+)$', message.strip()).group(1) if re.search(r'([.!?…]+)$', message.strip()) else "",
        "length": len(message)
    }

def update_behavior_pattern(old_pattern: dict, new_features: dict, memory_size: int = 30) -> dict:
    """Advanced customer behavior pattern learning for personalized service"""
    history = old_pattern.get("history", [])
    history.append(new_features)
    if len(history) > memory_size:
        history = history[-memory_size:]
    
    # Calculate customer personality profile
    total_interactions = len(history)
    
    # Communication style analysis
    avg_formality = sum(h.get("formality_score", 0) for h in history) / total_interactions
    avg_casualness = sum(h.get("casualness_score", 0) for h in history) / total_interactions
    
    # Customer service preferences
    avg_urgency = sum(h.get("urgency_level", 0) for h in history) / total_interactions
    complaint_ratio = sum(1 for h in history if h.get("complaint_indicators", 0) > 0) / total_interactions
    praise_ratio = sum(1 for h in history if h.get("praise_indicators", 0) > 0) / total_interactions
    
    # Technical proficiency
    avg_tech_proficiency = sum(h.get("tech_proficiency", 0) for h in history) / total_interactions
    
    # Communication preferences
    prefers_detail = sum(h.get("detail_preference", 0) for h in history) / total_interactions > 0.3
    prefers_brevity = sum(h.get("communication_brevity", 0) for h in history) / total_interactions > 0.3
    
    # Emotional patterns
    emotional_positivity = sum(h.get("positive_emotion", 0) for h in history) / max(1, sum(h.get("negative_emotion", 0) for h in history))
    
    # Personality trait aggregation
    aggregated_traits = {}
    trait_keys = ["patient", "impatient", "detail_oriented", "direct", "tech_savvy", "needs_guidance", "polite", "casual"]
    
    for trait in trait_keys:
        trait_count = sum(1 for h in history if h.get("personality_traits", {}).get(trait, False))
        aggregated_traits[trait] = trait_count / total_interactions
    
    # Determine customer type
    customer_type = "unknown"
    if aggregated_traits.get("tech_savvy", 0) > 0.6:
        customer_type = "tech_expert"
    elif aggregated_traits.get("needs_guidance", 0) > 0.5:
        customer_type = "beginner"
    elif avg_urgency > 0.8:
        customer_type = "urgent_customer"
    elif aggregated_traits.get("detail_oriented", 0) > 0.6:
        customer_type = "detail_focused"
    elif aggregated_traits.get("direct", 0) > 0.6:
        customer_type = "quick_service"
    elif avg_formality > avg_casualness:
        customer_type = "formal_professional"
    else:
        customer_type = "casual_friendly"
    
    # Service preferences
    preferred_response_style = {
        "formality_level": "formal" if avg_formality > avg_casualness else "casual",
        "detail_level": "detailed" if prefers_detail else "concise",
        "technical_level": "advanced" if avg_tech_proficiency > 0 else "basic",
        "urgency_sensitivity": "high" if avg_urgency > 0.5 else "normal"
    }
    
    return {
        # Customer profile
        "customer_type": customer_type,
        "total_interactions": total_interactions,
        "preferred_response_style": preferred_response_style,
        
        # Communication analysis
        "avg_formality_score": avg_formality,
        "avg_casualness_score": avg_casualness,
        "avg_message_length": sum(h.get("message_length", 0) for h in history) / total_interactions,
        
        # Service patterns
        "urgency_tendency": avg_urgency,
        "complaint_ratio": complaint_ratio,
        "praise_ratio": praise_ratio,
        "emotional_positivity": emotional_positivity,
        
        # Technical proficiency
        "tech_proficiency_level": avg_tech_proficiency,
        
        # Personality traits (aggregated)
        "personality_profile": aggregated_traits,
        
        # Preferences
        "prefers_detailed_responses": prefers_detail,
        "prefers_quick_responses": prefers_brevity,
        
        # Legacy compatibility
        "emoji_top": Counter([e for h in history for e in h.get("emojis", [])]).most_common(3),
        "keyword_top": Counter([k for h in history for k in h.get("keywords", [])]).most_common(5),
        "avg_typos": sum(h.get("typo_count", 0) for h in history) / total_interactions,
        "ending_top": Counter([h.get("ending", "") for h in history if h.get("ending")]).most_common(2),
        "avg_length": sum(h.get("length", 0) for h in history) / total_interactions,
        "history": history
    }

def save_user_behavior(db: Session, user_id: int, features: dict):
    behavior = db.query(UserBehavior).filter_by(user_id=user_id).first()
    if not behavior:
        behavior = UserBehavior(user_id=user_id, pattern={})
        db.add(behavior)
        db.commit()
    new_pattern = update_behavior_pattern(behavior.pattern or {}, features)
    behavior.pattern = new_pattern
    behavior.last_updated = datetime.utcnow()
    db.commit()
    return behavior.pattern

def load_user_behavior(db: Session, user_id: int) -> dict:
    behavior = db.query(UserBehavior).filter_by(user_id=user_id).first()
    return behavior.pattern if behavior else {}

def mimic_user_style(bot_message: str, user_pattern: dict) -> str:
    """Advanced customer service style adaptation based on user personality"""
    if not user_pattern:
        return bot_message
    
    customer_type = user_pattern.get("customer_type", "unknown")
    preferred_style = user_pattern.get("preferred_response_style", {})
    
    # Adapt based on customer type
    if customer_type == "tech_expert":
        # Technical customers prefer precise, detailed responses
        if not any(word in bot_message.lower() for word in ["chi tiết", "cụ thể", "technical", "spec"]):
            bot_message = "Chi tiết: " + bot_message
            
    elif customer_type == "beginner":
        # Beginners need guidance and simple language
        if not any(word in bot_message.lower() for word in ["hướng dẫn", "từng bước", "đơn giản"]):
            bot_message = bot_message + " (Tôi sẽ hướng dẫn bạn từng bước nhé!)"
            
    elif customer_type == "urgent_customer":
        # Urgent customers need immediate acknowledgment
        if not bot_message.startswith(("Ngay lập tức", "Tôi hiểu", "Được rồi")):
            bot_message = "Tôi hiểu tình huống khẩn cấp. " + bot_message
            
    elif customer_type == "detail_focused":
        # Detail-oriented customers want comprehensive answers
        if len(bot_message) < 100:  # Short response
            bot_message += " Để cung cấp thông tin đầy đủ hơn, tôi có thể giải thích chi tiết từng phần nếu bạn cần."
            
    elif customer_type == "quick_service":
        # Quick customers prefer concise responses
        if len(bot_message) > 150:  # Too long
            sentences = bot_message.split('. ')
            bot_message = sentences[0] + ('. ' + sentences[1] if len(sentences) > 1 else '') + '.'
    
    # Adapt formality level
    formality_level = preferred_style.get("formality_level", "formal")
    if formality_level == "formal":
        # Add formal language
        if not any(word in bot_message.lower() for word in ["anh", "chị", "dạ", "ạ"]):
            bot_message = "Dạ, " + bot_message.lower()
            bot_message = bot_message[0].upper() + bot_message[1:]  # Capitalize first letter
        
        # Add polite endings
        if not bot_message.endswith(("ạ.", "ạ!", "nhé.", "ạ?")):
            if bot_message.endswith('.'):
                bot_message = bot_message[:-1] + " ạ."
    else:
        # Casual style
        bot_message = bot_message.replace("Dạ, ", "").replace(" ạ", "")
        if not any(word in bot_message for word in ["nhé", "nha", "đó"]):
            if random.random() < 0.4:
                bot_message += " nhé!"
    
    # Adapt technical level
    tech_level = preferred_style.get("technical_level", "basic")
    if tech_level == "basic" and any(word in bot_message.lower() for word in ["api", "server", "database", "config"]):
        # Simplify technical terms
        bot_message = bot_message.replace("API", "ứng dụng")
        bot_message = bot_message.replace("server", "máy chủ")
        bot_message = bot_message.replace("database", "cơ sở dữ liệu")
    
    # Add empathy for negative emotions
    if user_pattern.get("emotional_positivity", 1) < 0.5:  # User seems frustrated
        empathy_phrases = [
            "Tôi hiểu cảm giác của bạn. ",
            "Tôi thấu hiểu tình huống này. ",
            "Tôi rất tiếc vì điều này. "
        ]
        if not any(phrase.lower() in bot_message.lower() for phrase in empathy_phrases):
            bot_message = random.choice(empathy_phrases) + bot_message
    
    # Add appreciation for positive customers
    elif user_pattern.get("praise_ratio", 0) > 0.3:  # Customer often gives praise
        if random.random() < 0.3:
            bot_message += " Cảm ơn bạn rất nhiều!"
    
    # Legacy compatibility for basic style mimicking
    if user_pattern.get("emoji_top"):
        em = user_pattern["emoji_top"][0][0]
        if em and em not in bot_message and random.random() < 0.3:
            bot_message += " " + em
    
    return bot_message

def process_user_message(db: Session, user_id: int, message: str):
    features = extract_behavior_features(message)
    pattern = save_user_behavior(db, user_id, features)
    return pattern

# ========== Typing Delay, Engagement, Anti-Spam ==========
class EngagementTracker:
    def __init__(self):
        self.last_message_time: Optional[float] = None
        self.message_frequency: int = 0
        self.cooldown_until: Optional[float] = None
        self.engagement_level: float = 0.5

    def update(self, now: float, message: str):
        if self.last_message_time and (now - self.last_message_time) < 5:
            self.message_frequency += 1
            if self.message_frequency > 6:
                self.cooldown_until = now + random.randint(30, 60)
                self.engagement_level = max(0.1, self.engagement_level - 0.3)
        else:
            self.message_frequency = max(0, self.message_frequency - 1)
        self.last_message_time = now
        self.engagement_level = min(1.0, self.engagement_level + len(message)/500)
        if any(word in message.lower() for word in ["bận", "thôi", "goodbye", "bye"]):
            self.engagement_level = max(0.1, self.engagement_level - 0.3)

    def is_in_cooldown(self, now: float):
        return self.cooldown_until and now < self.cooldown_until

    def typing_delay(self, length: int) -> float:
        base = 0.05 + min(length, 100) * 0.005
        if self.engagement_level < 0.3: base *= 1.5
        return random.uniform(base * 0.7, base * 1.3)

# ========== Enhanced GPU AI Engine for G4dn.xlarge ==========
import gc
import psutil
from contextlib import contextmanager

class AdvancedAIEngine:
    """Enhanced AI engine optimized for AWS G4dn.xlarge with T4 GPU and multiple specialized models"""
    
    def __init__(self, config_obj):
        self.config = config_obj
        
        # Model containers
        self.conversation_model = None
        self.conversation_tokenizer = None
        self.conversation_pipeline = None
        
        self.emotion_model = None
        self.emotion_pipeline = None
        
        self.sentiment_model = None
        self.sentiment_pipeline = None
        
        # Legacy support
        self.hf_model = None
        self.hf_tokenizer = None
        self.hf_pipeline = None
        self.llm = None
        
        # Status flags
        self.model_loaded = False
        self.hf_model_loaded = False
        
        # Performance tracking
        self.model_stats = {
            "total_inferences": 0,
            "avg_response_time": 0,
            "memory_usage": 0,
            "gpu_usage": 0
        }
        
        # Per-user engagement tracking
        self.engagement: Dict[str, EngagementTracker] = {}
        
        # GPU device setup
        self.device = self._setup_gpu_device()
        
    def _setup_gpu_device(self):
        """Setup optimal GPU device for G4dn.xlarge"""
        if torch.cuda.is_available() and self.config.HF_USE_GPU:
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"🎮 GPU Detected: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
            print(f"🚀 Using GPU acceleration for AI models")
            
            # Optimize GPU memory usage
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory'):
                try:
                    torch.cuda.memory.set_per_process_memory_fraction(0.9)  # Use 90% of VRAM
                except:
                    pass  # Fallback if not available
                    
            return device
        else:
            print("🖥️ GPU not available, falling back to CPU")
            return torch.device("cpu")
    
    @contextmanager
    def _gpu_memory_manager(self):
        """Context manager for GPU memory optimization"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        yield
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    
    def _load_hf_model(self):
        """Load HuggingFace Vietnamese model optimized for AWS G4dn.xlarge (T4 GPU, 16GB VRAM)"""
        if self.hf_model_loaded:
            return
            
        try:
            print(f"🔄 Loading HuggingFace model: {self.config.HF_MODEL_NAME}")
            print("🎮 Optimizing for AWS G4dn.xlarge (T4 GPU, 16GB VRAM)...")
            
            # Set PyTorch threading for G4dn.xlarge
            torch.set_num_threads(self.config.HF_NUM_THREADS)
            torch.set_num_interop_threads(self.config.HF_INTEROP_THREADS)
            
            print(f"🧵 CPU Threads: {self.config.HF_NUM_THREADS} (intra), {self.config.HF_INTEROP_THREADS} (inter)")
            
            with self._gpu_memory_manager():
                # Check available memory
                available_memory = psutil.virtual_memory().available / (1024**3)
                print(f"💾 Available RAM: {available_memory:.1f}GB")
                
                if self.device.type == "cuda":
                    available_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
                    print(f"🎮 Available VRAM: {available_vram:.1f}GB")
                
                # G4dn.xlarge optimized configuration (T4 GPU)
                model_config = {
                    "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                    "use_cache": True,
                    "device_map": "auto" if self.device.type == "cuda" else None,
                }
                
                if self.device.type == "cuda":
                    # GPU optimizations for T4
                    model_config.update({
                        "max_memory": {0: f"{self.config.MAX_VRAM_USAGE}GB"},
                        "offload_folder": "/tmp/model_offload",
                    })
                else:
                    # CPU fallback
                    model_config.update({
                        "offload_folder": "/tmp/model_offload",
                        "max_memory": f"{self.config.MAX_MEMORY_USAGE}GB",
                    })
                
                print(f"🔧 Using device: {self.device}")
                
                # Load model based on type with G4dn.xlarge optimizations
                if self.config.HF_MODEL_TYPE.lower() == "t5":
                    # T5 models optimized for GPU
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(
                        self.config.HF_MODEL_NAME,
                        model_max_length=self.config.HF_MAX_LENGTH,
                        padding_side="left",
                        truncation_side="left"
                    )
                    
                    print("🧠 Loading T5 model with G4dn.xlarge GPU optimizations...")
                    self.hf_model = T5ForConditionalGeneration.from_pretrained(
                        self.config.HF_MODEL_NAME, **model_config
                    )
                    
                    # GPU-optimized pipeline for G4dn.xlarge
                    self.hf_pipeline = pipeline(
                        "text2text-generation",
                        model=self.hf_model,
                        tokenizer=self.hf_tokenizer,
                        device=0 if self.device.type == "cuda" else -1,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        max_length=self.config.HF_MAX_LENGTH,
                        max_new_tokens=self.config.HF_MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=self.config.HF_TEMPERATURE,
                        top_p=self.config.HF_TOP_P,
                        top_k=self.config.HF_TOP_K,
                        num_beams=self.config.HF_NUM_BEAMS if self.device.type == "cuda" else 1,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                        batch_size=self.config.HF_BATCH_SIZE,
                    )
                    
                elif self.config.HF_MODEL_TYPE.lower() == "gpt":
                    # GPT models with G4dn.xlarge optimizations
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(
                        self.config.HF_MODEL_NAME,
                        padding_side="left"
                    )
                    if self.hf_tokenizer.pad_token is None:
                        self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
                    
                    print("🧠 Loading GPT model with G4dn.xlarge GPU optimizations...")
                    self.hf_model = AutoModelForCausalLM.from_pretrained(
                        self.config.HF_MODEL_NAME, **model_config
                    )
                    
                    # GPU-optimized pipeline for GPT
                    self.hf_pipeline = pipeline(
                        "text-generation",
                        model=self.hf_model,
                        tokenizer=self.hf_tokenizer,
                        device=0 if self.device.type == "cuda" else -1,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        max_new_tokens=self.config.HF_MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=self.config.HF_TEMPERATURE,
                        top_p=self.config.HF_TOP_P,
                        top_k=self.config.HF_TOP_K,
                        repetition_penalty=1.15,
                        pad_token_id=self.hf_tokenizer.pad_token_id,
                        eos_token_id=self.hf_tokenizer.eos_token_id,
                        return_full_text=False,
                        batch_size=self.config.HF_BATCH_SIZE,
                    )
                
                # Load additional specialized models for customer service
                if self.config.ENABLE_EMOTION_DETECTION:
                    # Note: These will be loaded on first use to avoid async issues
                    pass
                
                if self.config.ENABLE_SENTIMENT_ANALYSIS:
                    # Note: These will be loaded on first use to avoid async issues  
                    pass
                
                self.hf_model_loaded = True
                print("✅ Main HuggingFace model loaded successfully")
                self._print_memory_stats()
                
        except Exception as e:
            print(f"❌ Failed to load HuggingFace model: {e}")
            print("🔄 Falling back to CPU with reduced model size...")
            # Fallback strategy
            try:
                self.config.HF_MODEL_NAME = "VietAI/vit5-base"  # Smaller model
                self.config.HF_MAX_LENGTH = 256
                self.config.HF_BATCH_SIZE = 1
                self.device = torch.device("cpu")
                self._load_hf_model()  # Retry with smaller model
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                print("⚠️ Running without HuggingFace models")
    
    async def _load_emotion_model(self):
        """Load emotion detection model for enhanced customer service"""
        try:
            print(f"😊 Loading emotion model: {self.config.EMOTION_MODEL}")
            
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.config.EMOTION_MODEL,
                device=0 if self.device.type == "cuda" else -1,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                batch_size=self.config.HF_BATCH_SIZE,
            )
            
            print("✅ Emotion detection model loaded")
            
        except Exception as e:
            print(f"⚠️ Emotion model loading failed: {e}")
            self.emotion_pipeline = None
    
    async def _load_sentiment_model(self):
        """Load sentiment analysis model for customer satisfaction tracking"""
        try:
            print(f"📊 Loading sentiment model: {self.config.SENTIMENT_MODEL}")
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.SENTIMENT_MODEL,
                device=0 if self.device.type == "cuda" else -1,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                batch_size=self.config.HF_BATCH_SIZE,
            )
            
            print("✅ Sentiment analysis model loaded")
            
        except Exception as e:
            print(f"⚠️ Sentiment model loading failed: {e}")
            self.sentiment_pipeline = None
    
    def _load_emotion_model_sync(self):
        """Load emotion detection model synchronously"""
        if self.emotion_pipeline:
            return
            
        try:
            print(f"😊 Loading emotion model: {self.config.EMOTION_MODEL}")
            
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.config.EMOTION_MODEL,
                device=0 if self.device.type == "cuda" else -1,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                batch_size=self.config.HF_BATCH_SIZE,
            )
            
            print("✅ Emotion detection model loaded")
            
        except Exception as e:
            print(f"⚠️ Emotion model loading failed: {e}")
            self.emotion_pipeline = None
    
    def _load_sentiment_model_sync(self):
        """Load sentiment analysis model synchronously"""
        if self.sentiment_pipeline:
            return
            
        try:
            print(f"📊 Loading sentiment model: {self.config.SENTIMENT_MODEL}")
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.SENTIMENT_MODEL,
                device=0 if self.device.type == "cuda" else -1,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                batch_size=self.config.HF_BATCH_SIZE,
            )
            
            print("✅ Sentiment analysis model loaded")
            
        except Exception as e:
            print(f"⚠️ Sentiment model loading failed: {e}")
            self.sentiment_pipeline = None
    
    def _print_memory_stats(self):
        """Print current memory usage statistics"""
        # System memory
        memory = psutil.virtual_memory()
        print(f"💾 System RAM: {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent:.1f}%)")
        
        # GPU memory
        if self.device.type == "cuda":
            try:
                gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"🎮 GPU VRAM: {gpu_memory_used:.1f}GB allocated / {gpu_memory_reserved:.1f}GB reserved / {gpu_memory_total:.1f}GB total")
            except Exception as e:
                print(f"⚠️ Could not get GPU memory stats: {e}")
    
    def _load_llama_model(self):
        """Legacy Llama model loading (fallback)"""
        if self.model_loaded:
            return
        try:
            print(f"🦙 Loading Llama model: {self.config.MODEL_CONFIG['path']}")
            # Your existing llama loading code here if needed
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Failed to load Llama model: {e}")
    
    def _generate_with_hf(self, prompt: str) -> str:
        """Generate customer service response optimized for AWS T3.large"""
        if not self.hf_pipeline:
            return None
            
        try:
            # Memory management for T3.large
            import gc
            gc.collect()
            
            if self.config.HF_MODEL_TYPE.lower() == "t5":
                # T5 models with customer service focus
                formatted_prompt = f"""Bạn là nhân viên chăm sóc khách hàng chuyên nghiệp và thân thiện.
Hãy trả lời một cách chu đáo, giải quyết vấn đề hiệu quả và thể hiện sự quan tâm đến khách hàng.
Sử dụng ngôn ngữ phù hợp với tính cách của khách hàng.

{prompt}"""
                
                # Single generation for T3.large efficiency
                result = self.hf_pipeline(
                    formatted_prompt, 
                    max_length=self.config.HF_MAX_LENGTH,
                    do_sample=True,
                    temperature=0.8,
                    num_return_sequences=1,  # Single sequence for memory efficiency
                    clean_up_tokenization_spaces=True,
                )
                
                if result and result[0]["generated_text"]:
                    return result[0]["generated_text"].strip()
                
            elif self.config.HF_MODEL_TYPE.lower() in ["gpt", "bert"]:
                # GPT-style models with customer service prompt
                enhanced_prompt = f"""[Vai trò: Nhân viên chăm sóc khách hàng chuyên nghiệp]
[Mục tiêu: Giải quyết vấn đề và tạo trải nghiệm tích cực]
{prompt}

[Hướng dẫn: Trả lời thân thiện, chuyên nghiệp, phù hợp với tính cách khách hàng]
Trả lời:"""
                
                # Generate with T3.large optimized parameters
                result = self.hf_pipeline(
                    enhanced_prompt, 
                    max_new_tokens=self.config.HF_MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.85,
                    top_p=0.92,
                    num_return_sequences=1,  # Memory efficient
                    pad_token_id=self.hf_tokenizer.pad_token_id,
                    clean_up_tokenization_spaces=True,
                )
                
                if result and result[0]["generated_text"]:
                    generated_text = result[0]["generated_text"]
                    # Clean up the response
                    if "Trả lời:" in generated_text:
                        response = generated_text.split("Trả lời:")[-1].strip()
                    elif generated_text.startswith(enhanced_prompt):
                        response = generated_text[len(enhanced_prompt):].strip()
                    else:
                        response = generated_text.strip()
                    
                    return response
                    
        except Exception as e:
            print(f"⚠️ HuggingFace generation failed on T3.large: {e}")
            
            # Retry with minimal parameters for T3.large
            try:
                if self.config.HF_MODEL_TYPE.lower() == "t5":
                    result = self.hf_pipeline(
                        prompt, 
                        max_length=min(64, self.config.HF_MAX_LENGTH // 2),
                        do_sample=False,  # Greedy for speed
                        temperature=0.7,
                    )
                    if result and result[0]["generated_text"]:
                        return result[0]["generated_text"].strip()
                        
            except Exception as retry_e:
                print(f"⚠️ Retry also failed: {retry_e}")
                return None
                
        finally:
            # Always clean up memory on T3.large
            import gc
            gc.collect()

    def _generate_with_llama(self, prompt: str) -> str:
        """Generate response using llama-cpp model"""
        if not self.llm:
            return None
            
        try:
            response = self.llm(
                prompt,
                max_tokens=self.config.MODEL_CONFIG['params']['max_length'],
                temperature=self.config.MODEL_CONFIG['params']['temperature'],
                stop=["</s>", "<|endoftext|>", "###", "User:", "Assistant:", "Khách:", "Bot:"]
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"⚠️  Llama-cpp generation failed: {e}")
            return None

    async def generate_response(self, message: str, db: Session = None, user_id: int = None) -> str:
        """Legacy method for backward compatibility"""
        user_pattern = load_user_behavior(db, user_id) if db and user_id else {}
        response_data = await self.enhanced_generate(message, "", user_pattern)
        return response_data.get("response", "Xin lỗi, đã có lỗi xảy ra.")
    
    async def enhanced_generate(self, message: str, context: str = "", user_pattern: dict = None) -> dict:
        """Generate enhanced AI response with emotion, sentiment analysis and customer adaptation"""
        start_time = time.time()
        
        # Load models if not already loaded
        if not self.hf_model_loaded:
            await asyncio.get_event_loop().run_in_executor(None, self._load_hf_model)
        
        with self._gpu_memory_manager():
            try:
                # Load specialized models on first use
                if self.config.ENABLE_EMOTION_DETECTION and not self.emotion_pipeline:
                    self._load_emotion_model_sync()
                
                if self.config.ENABLE_SENTIMENT_ANALYSIS and not self.sentiment_pipeline:
                    self._load_sentiment_model_sync()
                
                # Analyze emotion and sentiment for customer service insights
                emotion_analysis = None
                sentiment_analysis = None
                
                if self.emotion_pipeline:
                    try:
                        emotion_result = self.emotion_pipeline(message)
                        emotion_analysis = emotion_result[0] if emotion_result else None
                    except Exception as e:
                        print(f"⚠️ Emotion analysis failed: {e}")
                
                if self.sentiment_pipeline:
                    try:
                        sentiment_result = self.sentiment_pipeline(message)
                        sentiment_analysis = sentiment_result[0] if sentiment_result else None
                    except Exception as e:
                        print(f"⚠️ Sentiment analysis failed: {e}")
                
                # Create enhanced prompt with customer service context
                enhanced_prompt = self._create_customer_service_prompt(
                    message, context, user_pattern, emotion_analysis, sentiment_analysis
                )
                
                # Generate main response using the optimized pipeline
                generated_text = "Xin lỗi, hệ thống AI đang gặp vấn đề kỹ thuật."
                
                if self.hf_pipeline:
                    try:
                        response_result = self.hf_pipeline(enhanced_prompt)
                        if isinstance(response_result, list) and len(response_result) > 0:
                            generated_text = response_result[0].get('generated_text', enhanced_prompt)
                            # Extract only the new generated part
                            if generated_text.startswith(enhanced_prompt):
                                generated_text = generated_text[len(enhanced_prompt):].strip()
                            elif "Trợ lý AI:" in generated_text:
                                # Extract response after the AI marker
                                parts = generated_text.split("Trợ lý AI:")
                                if len(parts) > 1:
                                    generated_text = parts[-1].strip()
                        
                        # Clean up the response
                        generated_text = self._clean_response(generated_text)
                        
                    except Exception as e:
                        print(f"⚠️ Pipeline generation failed: {e}")
                        generated_text = "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Bạn có thể thử lại không?"
                
                # Apply customer-specific style adaptation
                if user_pattern:
                    generated_text = mimic_user_style(generated_text, user_pattern)
                
                response_time = time.time() - start_time
                
                # Update performance statistics
                self.model_stats["total_inferences"] += 1
                self.model_stats["avg_response_time"] = (
                    self.model_stats["avg_response_time"] * (self.model_stats["total_inferences"] - 1) + response_time
                ) / self.model_stats["total_inferences"]
                
                return {
                    "response": generated_text,
                    "emotion_analysis": emotion_analysis,
                    "sentiment_analysis": sentiment_analysis,
                    "response_time": response_time,
                    "model_stats": self.model_stats.copy(),
                    "device_used": str(self.device)
                }
                
            except Exception as e:
                print(f"❌ Error in enhanced generation: {e}")
                return {
                    "response": "Xin lỗi, đã có lỗi xảy ra trong hệ thống AI. Vui lòng thử lại sau.",
                    "error": str(e),
                    "response_time": time.time() - start_time,
                    "device_used": str(self.device)
                }
    
    def get_model_status(self) -> dict:
        """Get current model loading status"""
        return {
            "hf_model_loaded": self.hf_model_loaded,
            "model_loaded": self.model_loaded,
            "device": str(self.device),
            "total_inferences": self.model_stats["total_inferences"],
            "avg_response_time": self.model_stats["avg_response_time"]
        }
    
    def _create_customer_service_prompt(self, message: str, context: str, user_pattern: dict, 
                                      emotion_analysis: dict, sentiment_analysis: dict) -> str:
        """Create enhanced customer service prompt with emotional and contextual awareness"""
        
        # Base customer service context for TikTok
        system_prompt = "Bạn là trợ lý AI chăm sóc khách hàng chuyên nghiệp cho TikTok Việt Nam. "
        system_prompt += "Hãy trả lời một cách thân thiện, hữu ích và chính xác. "
        
        # Add customer personality adaptation
        if user_pattern:
            customer_type = user_pattern.get("customer_type", "unknown")
            if customer_type == "tech_expert":
                system_prompt += "Khách hàng là người hiểu biết kỹ thuật, hãy trả lời chi tiết và chuyên sâu. "
            elif customer_type == "beginner":
                system_prompt += "Khách hàng cần hướng dẫn đơn giản, hãy giải thích từng bước một cách dễ hiểu. "
            elif customer_type == "urgent_customer":
                system_prompt += "Khách hàng cần giải quyết khẩn cấp, hãy ưu tiên giải pháp nhanh chóng và hiệu quả. "
            elif customer_type == "detail_focused":
                system_prompt += "Khách hàng thích thông tin chi tiết, hãy cung cấp đầy đủ thông tin cần thiết. "
            elif customer_type == "quick_service":
                system_prompt += "Khách hàng muốn giải quyết nhanh, hãy đưa ra câu trả lời ngắn gọn và hiệu quả. "
        
        # Add emotional awareness for better customer service
        if emotion_analysis:
            emotion = emotion_analysis.get("label", "").lower()
            confidence = emotion_analysis.get("score", 0)
            
            if confidence > 0.7:  # High confidence emotion detection
                if emotion in ["anger", "frustration"]:
                    system_prompt += "Khách hàng đang tức giận hoặc bực bội, hãy thể hiện sự đồng cảm và kiên nhẫn giải quyết vấn đề. "
                elif emotion in ["joy", "happiness"]:
                    system_prompt += "Khách hàng đang vui vẻ, hãy duy trì không khí tích cực và năng lượng này. "
                elif emotion in ["sadness", "disappointment"]:
                    system_prompt += "Khách hàng đang buồn hoặc thất vọng, hãy an ủi và tìm giải pháp tích cực. "
                elif emotion in ["fear", "anxiety"]:
                    system_prompt += "Khách hàng đang lo lắng, hãy trấn an và đưa ra hướng dẫn rõ ràng. "
        
        # Add sentiment awareness for service quality
        if sentiment_analysis:
            sentiment = sentiment_analysis.get("label", "").lower()
            confidence = sentiment_analysis.get("score", 0)
            
            if confidence > 0.6:
                if "negative" in sentiment:
                    system_prompt += "Khách hàng có cảm xúc tiêu cực, hãy thể hiện sự quan tâm và hỗ trợ tích cực để cải thiện trải nghiệm. "
                elif "positive" in sentiment:
                    system_prompt += "Khách hàng có cảm xúc tích cực, hãy duy trì và tăng cường sự hài lòng này. "
        
        # Prepare conversation context
        context_text = ""
        if context:
            context_text = f"\n\nLịch sử hội thoại gần đây:\n{context}"
        
        # Create the full prompt
        full_prompt = f"{system_prompt}{context_text}\n\nKhách hàng: {message}\n\nTrợ lý AI:"
        
        return full_prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean and validate AI response"""
        if not response:
            return "Xin lỗi, tôi không thể xử lý yêu cầu này. Bạn có thể thử lại không?"
        
        # Remove unwanted patterns
        response = response.strip()
        
        # Remove duplicate phrases
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in cleaned_lines:
                cleaned_lines.append(line)
        
        response = ' '.join(cleaned_lines)
        
        # Ensure proper Vietnamese punctuation
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Limit response length for better user experience
        if len(response) > 800:
            sentences = response.split('. ')
            response = '. '.join(sentences[:3]) + '.'
        
        return response

# ========== FastAPI ==========
app = FastAPI(title="AI Chat System Pro with Vietnamese HuggingFace Models")
# Initialize the enhanced AI engine for G4dn.xlarge
ai_engine = AdvancedAIEngine(config)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def api_key_auth(x_api_key: str = Header(None)):
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.post("/chat")
async def chat(
    message: str,
    user_id: str,
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Header(None)
):
    api_key_auth(x_api_key)
    try:
        user = db.query(User).filter(User.username == user_id).first()
        if not user:
            user = User(username=user_id)
            db.add(user)
            db.commit()
        session = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user.id)
            .filter(ChatSession.end_time.is_(None))
            .first()
        )
        if not session:
            session = ChatSession(user_id=user.id)
            db.add(session)
            db.commit()
        process_user_message(db, user.id, message)
        response_data = await ai_engine.enhanced_generate(
            message=message,
            context="",
            user_pattern=load_user_behavior(db, user.id)
        )
        response_text = response_data.get("response", "Xin lỗi, đã có lỗi xảy ra.")
        chat_message = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=response_text,
            metrics={}
        )
        db.add(chat_message)
        session.messages_count += 1
        db.commit()
        return {
            'response': response_text,  # Changed from 'content' to 'response'
            'content': response_text,   # Keep both for compatibility
            'engagement_level': ai_engine.engagement.get(str(user.id), EngagementTracker()).engagement_level,
        }
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        send_telegram_alert("❗Lỗi /chat", str(e))
        raise HTTPException(status_code=429 if "cooldown" in str(e).lower() else 500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get CyberBot status and model information"""
    try:
        model_info = {
            "huggingface_enabled": config.USE_HUGGINGFACE,
            "huggingface_loaded": ai_engine.hf_model_loaded if ai_engine else False,
            "llama_loaded": ai_engine.model_loaded if ai_engine else False,
            "active_model": None,
            "model_type": None,
            "fallback_available": False
        }
        
        if ai_engine:
            if config.USE_HUGGINGFACE and ai_engine.hf_model_loaded:
                model_info.update({
                    "active_model": config.HF_MODEL_NAME,
                    "model_type": config.HF_MODEL_TYPE,
                    "parameters": config.HF_MAX_LENGTH,
                    "temperature": config.HF_TEMPERATURE
                })
            elif ai_engine.model_loaded:
                model_info.update({
                    "active_model": config.MODEL_CONFIG["path"],
                    "model_type": "llama-cpp",
                    "fallback_available": True
                })
            else:
                model_info.update({
                    "active_model": "simple_fallback",
                    "model_type": "fallback"
                })
        
        # System info
        import psutil
        system_info = {
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 1),
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1)
        }
        
        # Database info
        db_info = {
            "database_url": config.DB_URL,
            "pinecone_enabled": pc is not None,
            "neo4j_enabled": config.NEO4J_URI is not None
        }
        
        return {
            "status": "active",
            "model": model_info,
            "system": system_info,
            "database": db_info,
            "api_version": "2.0",
            "vietnamese_support": True,
            "features": [
                "vietnamese_language_models",
                "vector_memory",
                "knowledge_graph",
                "behavior_tracking",
                "anti_spam",
                "telegram_alerts"
            ]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "model": {"active_model": "unknown"},
            "api_version": "2.0"
        }

@app.post("/batch_chat")
async def batch_chat(
    messages: List[str],
    user_id: str,
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Header(None),
    background_tasks: BackgroundTasks = None
):
    api_key_auth(x_api_key)
    async def process_all():
        results = []
        for msg in messages:
            response_data = await ai_engine.enhanced_generate(
                message=msg,
                context="",
                user_pattern=load_user_behavior(db, user_id)
            )
            res = response_data.get("response", "Lỗi xử lý tin nhắn.")
            results.append(res)
        return results
    results = await process_all()
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.API_PORT)