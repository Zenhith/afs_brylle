import os
import re
import json
import time
import logging
import threading
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from sqlalchemy import func

# Data Processing
import requests
import feedparser
import sqlite3
import hashlib
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

# Machine Learning and NLP Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Deep Learning
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# Web Scraping and Data Collection
import scrapy
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup

# Dashboard and Visualization
import dash
from dash import dcc, html, dash_table, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from dash_bootstrap_templates import load_figure_template

# Database
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base
Base = declarative_base()
from sqlalchemy.orm import sessionmaker, relationship

# Constants
VICTIM_TYPES = [
	"Politicians", "CEOs/Executives", "Celebrities", 
	"Financial Institutions", "Government Agencies",
	"Healthcare Organizations", "Small Businesses", 
	"Elderly Individuals", "General Public",
	"Tech Professionals", "Social Media Influencers",
	"Academic Institutions", "Military Personnel"
]

RISK_LEVELS = ["Critical", "High", "Medium", "Low", "Minimal"]

"""
Helper functions for handling datetime objects in your forecast functions.
Add these to your utility functions or include them in the relevant classes.
"""

def safe_datetime_operation(date_obj, days_to_add=0):
	"""
	Safely add days to a datetime object with proper type checking
	
	Args:
		date_obj: A datetime object
		days_to_add: Number of days to add (can be negative)
	
	Returns:
		A new datetime object with days added/subtracted
	"""
	from datetime import datetime, timedelta
	
	# Make sure we're working with a datetime object
	if not isinstance(date_obj, datetime):
		# Try to convert to datetime if it's a string
		if isinstance(date_obj, str):
			try:
				date_obj = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
			except ValueError:
				# If conversion fails, return current datetime
				return datetime.now()
		else:
			# If it's neither a datetime nor a string, return current datetime
			return datetime.now()
	
	# Add days using timedelta
	return date_obj + timedelta(days=days_to_add)

def ensure_datetime_list(date_list):
	"""
	Ensure all items in a list are datetime objects
	
	Args:
		date_list: List of datetime objects or strings
	
	Returns:
		List with all items converted to datetime objects
	"""
	from datetime import datetime
	
	result = []
	for date_item in date_list:
		if isinstance(date_item, datetime):
			result.append(date_item)
		elif isinstance(date_item, str):
			try:
				# Try to parse ISO format
				dt = datetime.fromisoformat(date_item.replace('Z', '+00:00'))
				result.append(dt)
			except ValueError:
				# If parsing fails, use current datetime
				result.append(datetime.now())
		else:
			# For any other type, use current datetime
			result.append(datetime.now())
	
	return result

def sort_datetime_data(dates, values, lower_bounds=None, upper_bounds=None):
	"""
	Sort data by datetime while maintaining corresponding values
	
	Args:
		dates: List of datetime objects
		values: List of corresponding values
		lower_bounds: Optional list of lower bounds
		upper_bounds: Optional list of upper bounds
	
	Returns:
		Tuple of sorted lists (dates, values, [lower_bounds], [upper_bounds])
	"""
	# Make sure all dates are datetime objects
	dates = ensure_datetime_list(dates)
	
	# Create pairs for sorting
	data = list(zip(dates, values))
	
	# Sort by date
	sorted_data = sorted(data, key=lambda x: x[0])
	
	# Unzip the sorted data
	sorted_dates, sorted_values = zip(*sorted_data) if data else ([], [])
	
	# If bounds are provided, sort them too
	if lower_bounds is not None and upper_bounds is not None:
		# Create triples with bounds
		bound_data = list(zip(dates, lower_bounds, upper_bounds))
		
		# Sort by date
		sorted_bound_data = sorted(bound_data, key=lambda x: x[0])
		
		# Unzip the sorted bounds
		_, sorted_lower, sorted_upper = zip(*sorted_bound_data) if bound_data else ([], [], [])
		
		return sorted_dates, sorted_values, sorted_lower, sorted_upper
	
	return sorted_dates, sorted_values

class FraudReport(Base):
	"""Database model for fraud reports"""
	__tablename__ = 'fraud_reports'
	
	id = Column(Integer, primary_key=True)
	source = Column(String(100), nullable=False)
	title = Column(String(500), nullable=False)
	description = Column(Text, nullable=True)
	url = Column(String(1000), nullable=True)
	category = Column(String(100), nullable=False)
	timestamp = Column(DateTime, default=datetime.now)
	risk_level = Column(String(20), nullable=True)
	impact_score = Column(Float, nullable=True)
	verified = Column(Integer, default=0)  # 0: unverified, 1: verified, 2: false positive
	
	# Relationships
	victims = relationship("VictimAssociation", back_populates="report")
	tools = relationship("ToolAssociation", back_populates="report")
	
	def __repr__(self):
		return f"<FraudReport(id={self.id}, category='{self.category}', risk='{self.risk_level}')>"

class Victim(Base):
	"""Database model for victim types"""
	__tablename__ = 'victims'
	
	id = Column(Integer, primary_key=True)
	type = Column(String(100), nullable=False, unique=True)
	description = Column(Text, nullable=True)
	vulnerability_score = Column(Float, nullable=True)
	
	# Relationships
	reports = relationship("VictimAssociation", back_populates="victim")
	
	def __repr__(self):
		return f"<Victim(id={self.id}, type='{self.type}')>"

class Tool(Base):
	"""Database model for fraud tools"""
	__tablename__ = 'tools'
	
	id = Column(Integer, primary_key=True)
	name = Column(String(100), nullable=False, unique=True)
	description = Column(Text, nullable=True)
	sophistication_level = Column(Float, nullable=True)  # 0-10 scale
	first_observed = Column(DateTime, nullable=True)
	
	# Relationships
	reports = relationship("ToolAssociation", back_populates="tool")
	
	def __repr__(self):
		return f"<Tool(id={self.id}, name='{self.name}')>"

class VictimAssociation(Base):
	"""Association table for many-to-many relationship between reports and victims"""
	__tablename__ = 'victim_associations'
	
	id = Column(Integer, primary_key=True)
	report_id = Column(Integer, ForeignKey('fraud_reports.id'))
	victim_id = Column(Integer, ForeignKey('victims.id'))
	confidence = Column(Float, default=1.0)  # Confidence score (0-1)
	
	# Relationships
	report = relationship("FraudReport", back_populates="victims")
	victim = relationship("Victim", back_populates="reports")

class ToolAssociation(Base):
	"""Association table for many-to-many relationship between reports and tools"""
	__tablename__ = 'tool_associations'
	
	id = Column(Integer, primary_key=True)
	report_id = Column(Integer, ForeignKey('fraud_reports.id'))
	tool_id = Column(Integer, ForeignKey('tools.id'))
	confidence = Column(Float, default=1.0)  # Confidence score (0-1)
	
	# Relationships
	report = relationship("FraudReport", back_populates="tools")
	tool = relationship("Tool", back_populates="reports")



class AdvancedFraudFilter:
	"""
	Sophisticated filtering system to identify genuine fraud reports
	with additional victim and tool tracking capabilities
	"""
	def __init__(self):
		"""Initialize enhanced fraud filtering criteria"""
		# Comprehensive fraud-related keywords
		self.fraud_indicators = {
			'strong_indicators': [
				# Direct fraud terms
				'fraud', 'scam', 'phishing', 'identity theft', 
				'cybercrime', 'data breach', 'hack', 'stolen',
				
				# AI-specific fraud indicators
				'deepfake', 'synthetic identity', 'ai-powered fraud',
				'machine learning scam', 'generative ai fraud',
				'voice cloning', 'synthetic media fraud',
				
				# Financial fraud terms
				'financial fraud', 'monetary theft', 'investment scam',
				'digital wallet fraud', 'cryptocurrency scam',
				
				# Social engineering
				'social engineering', 'credential theft', 
				'impersonation', 'fake profile'
			],
			
			'weak_indicators': [
				'security risk', 'privacy concern', 'potential exploit',
				'vulnerability', 'cybersecurity', 'data manipulation',
				'credentials', 'fake', 'fraud detection', 'digital identity',
				'biometric', 'artificial intelligence', 'ml', 'deception', 
				'unauthorized access', 'spoofing', 'synthetic voice',
				'profile', 'verification', 'authentication', 'trust',
				'deepfake detection', 'scam prevention', 'digital safety'
			],
			
			# Terms to explicitly filter out
			'exclusion_terms': [
				'patch tuesday', 'software update', 'new feature',
				'company announcement', 'product launch', 'tech news',
				'team graduation', 'conference', 'research paper'
			]
		}
		
		# Victim type indicators (terms associated with victim types)
		self.victim_indicators = {
			'Politicians': [
				'politician', 'lawmaker', 'senator', 'congressman', 
				'parliament', 'elected official', 'candidate', 
				'campaign', 'political', 'government official'
			],
			'CEOs/Executives': [
				'ceo', 'executive', 'cfo', 'cio', 'cto', 'board member',
				'director', 'business leader', 'corporate', 'company executive'
			],
			'Celebrities': [
				'celebrity', 'actor', 'actress', 'singer', 'entertainer',
				'influencer', 'public figure', 'famous', 'star', 'athlete'
			],
			'Financial Institutions': [
				'bank', 'financial institution', 'credit union', 'investment firm',
				'insurance company', 'hedge fund', 'broker', 'trading platform'
			],
			'Government Agencies': [
				'government agency', 'federal', 'department', 'state agency',
				'regulatory body', 'intelligence agency', 'military', 'police'
			],
			'Healthcare Organizations': [
				'hospital', 'clinic', 'healthcare provider', 'medical center',
				'pharmacy', 'insurance provider', 'patient data'
			],
			'Small Businesses': [
				'small business', 'startup', 'local business', 'entrepreneur',
				'shop owner', 'retailer', 'small enterprise'
			],
			'Elderly Individuals': [
				'elderly', 'senior citizen', 'retired', 'aging', 'older adult',
				'pensioner', 'retirement'
			],
			'General Public': [
				'public', 'consumer', 'citizen', 'user', 'customer',
				'individual', 'general population', 'people'
			],
			'Tech Professionals': [
				'developer', 'engineer', 'it professional', 'security researcher',
				'programmer', 'system administrator', 'tech worker'
			],
			'Social Media Influencers': [
				'influencer', 'content creator', 'youtuber', 'streamer',
				'social media personality', 'online personality'
			],
			'Academic Institutions': [
				'university', 'college', 'school', 'academic', 'education',
				'student', 'faculty', 'researcher', 'professor'
			],
			'Military Personnel': [
				'military', 'soldier', 'veteran', 'armed forces', 'defense',
				'navy', 'army', 'air force', 'marine'
			]
		}
		
		# Tool indicators (terms associated with fraud tools)
		self.tool_indicators = {
			'Deepfake Generation Software': [
				'deepfake', 'synthetic media', 'face swap', 'ai video',
				'video manipulation', 'synthetic generation'
			],
			'Voice Synthesis Tools': [
				'voice clone', 'voice synthesis', 'speech synthesis',
				'voice replication', 'audio deepfake', 'voice mimicry'
			],
			'Phishing Kits': [
				'phishing kit', 'credential harvester', 'spoofed site',
				'fake login', 'credential theft'
			],
			'Identity Forgery Tools': [
				'identity forgery', 'id generator', 'synthetic identity',
				'fake persona', 'profile generator'
			],
			'Social Engineering Frameworks': [
				'social engineering', 'pretexting', 'psychological manipulation',
				'trust exploitation', 'persuasion framework'
			],
			'Botnet Infrastructure': [
				'botnet', 'command and control', 'zombie network',
				'distributed attack', 'bot herder'
			],
			'Cryptocurrency Scam Tools': [
				'crypto scam', 'blockchain exploit', 'wallet drainer',
				'defi hack', 'crypto theft'
			],
			'Automated Disinformation Systems': [
				'disinformation', 'fake news generator', 'propaganda tool',
				'automated narrative', 'synthetic text'
			],
			'Malware Frameworks': [
				'malware', 'ransomware', 'spyware', 'trojan',
				'backdoor', 'virus', 'worm'
			],
			'LLM-Based Scam Generators': [
				'llm', 'gpt', 'ai writer', 'language model',
				'text generation', 'chatbot scam'
			]
		}
	
	def is_fraud_report(self, text):
		"""
		Determine if the text represents a genuine fraud report
		
		Args:
			text: Text to analyze
		
		Returns:
			Boolean indicating if it's a fraud report
		"""
		# Convert to lowercase for case-insensitive matching
		text_lower = text.lower()
		
		# Check for exclusion terms first (quick filter)
		if any(term in text_lower for term in self.fraud_indicators['exclusion_terms']):
			return False
		
		# Count strong and weak indicators
		strong_matches = sum(1 for term in self.fraud_indicators['strong_indicators'] 
							 if term in text_lower)
		weak_matches = sum(1 for term in self.fraud_indicators['weak_indicators'] 
						   if term in text_lower)
		
		# Scoring mechanism - strong indicators have more weight
		fraud_score = strong_matches * 2 + weak_matches
		
		# Adjust thresholds as needed
		return fraud_score >= 1  # At least 1 points from strong or weak indicators
	# Add these additional keywords to the fraud_indicators
	additional_strong_indicators = [
	'ai-generated scam', 'large language model fraud', 'chatgpt scam',
	'ai fraud', 'llm-based fraud', 'generative ai misuse', 'claude scam',
	'bard fraud', 'gpt fraud', 'synthetic fraud', 'automated fraud',
	'digital deception', 'voice synthesis fraud', 'neural fake',
	'ai impersonation', 'prompt injection', 'ai deception', 
	'model manipulation', 'data poisoning'
	]
	
	def identify_victims(self, text):
		"""
		Identify potential victim types mentioned in the text
		
		Args:
			text: Text to analyze
		
		Returns:
			Dictionary mapping victim types to confidence scores
		"""
		text_lower = text.lower()
		victims = {}
		
		for victim_type, indicators in self.victim_indicators.items():
			# Count matches
			matches = sum(1 for indicator in indicators if indicator in text_lower)
			
			# Calculate confidence based on number of matches
			if matches > 0:
				# Normalize confidence between 0.5 and 1.0
				confidence = 0.5 + min(matches / len(indicators) * 0.5, 0.5)
				victims[victim_type] = confidence
		
		return victims
	
	def identify_tools(self, text):
		"""
		Identify potential tools mentioned in the text
		
		Args:
			text: Text to analyze
		
		Returns:
			Dictionary mapping tool types to confidence scores
		"""
		text_lower = text.lower()
		tools = {}
		
		for tool_name, indicators in self.tool_indicators.items():
			# Count matches
			matches = sum(1 for indicator in indicators if indicator in text_lower)
			
			# Calculate confidence based on number of matches
			if matches > 0:
				# Normalize confidence between 0.5 and 1.0
				confidence = 0.5 + min(matches / len(indicators) * 0.5, 0.5)
				tools[tool_name] = confidence
		
		return tools
	
	def assess_risk_level(self, text, victims, tools):
		"""
		Assess the risk level of a fraud report
		
		Args:
			text: Text to analyze
			victims: Identified victim types with confidence scores
			tools: Identified tools with confidence scores
		
		Returns:
			Risk level (Critical, High, Medium, Low, Minimal)
		"""
		# Base score
		risk_score = 0
		
		# Victim impact factor - more sensitive victims increase risk
		high_impact_victims = {'Politicians', 'CEOs/Executives', 'Government Agencies', 
							   'Financial Institutions', 'Healthcare Organizations'}
		
		for victim, confidence in victims.items():
			impact_multiplier = 2.0 if victim in high_impact_victims else 1.0
			risk_score += confidence * impact_multiplier
		
		# Tool sophistication factor
		high_sophistication_tools = {'Deepfake Generation Software', 'Voice Synthesis Tools', 
									'LLM-Based Scam Generators', 'Automated Disinformation Systems'}
		
		for tool, confidence in tools.items():
			sophistication_multiplier = 2.0 if tool in high_sophistication_tools else 1.0
			risk_score += confidence * sophistication_multiplier
		
		# Scale and categorize
		if risk_score >= 5.0:
			return "Critical"
		elif risk_score >= 3.0:
			return "High"
		elif risk_score >= 1.5:
			return "Medium"
		elif risk_score >= 0.5:
			return "Low"
		else:
			return "Minimal"
	
	def clean_and_prepare_text(self, text):
		"""
		Clean and prepare text for analysis
		
		Args:
			text: Input text
		
		Returns:
			Cleaned and prepared text
		"""
		# Remove extra whitespace
		text = re.sub(r'\s+', ' ', text).strip()
		
		# Remove special characters and normalize
		text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
		
		return text

class AdvancedFraudClassifier:
	"""
	Sophisticated AI fraud classification system with advanced neural models
	and improved categorical classification
	"""
	def __init__(self, model_path='models'):
		"""
		Initialize the advanced fraud classifier
		
		Args:
			model_path: Path to store/load models
		"""
		# Logging setup
		logging.basicConfig(
			level=logging.INFO,
			format='%(asctime)s - %(levelname)s: %(message)s'
		)
		self.logger = logging.getLogger(__name__)
		self.model_path = model_path
		
		# Ensure model directory exists
		os.makedirs(self.model_path, exist_ok=True)
		
		# Predefined fraud categories with expanded descriptions
		self.fraud_categories = {
			'Deepfake Fraud': [
				'synthetic media manipulation',
				'AI-generated fake videos',
				'identity impersonation through video',
				'false visual representation',
				'manipulated video content',
				'synthetic facial mapping',
				'AI-generated face replacement',
				'video identity theft'
			],
			'Voice Cloning Scam': [
				'artificial voice replication',
				'audio fraud',
				'voice impersonation',
				'synthetic speech fraud',
				'audio deepfake',
				'voice synthesis scam',
				'speech cloning attack',
				'telephony-based voice fraud'
			],
			'AI Phishing': [
				'advanced social engineering',
				'AI-powered email scams',
				'machine learning phishing',
				'intelligent credential theft',
				'contextual phishing attacks',
				'personalized scam messages',
				'AI-generated spear phishing',
				'automated social engineering'
			],
			'Synthetic Identity Theft': [
				'AI-generated fake identities',
				'machine learning identity creation',
				'algorithmic persona fabrication',
				'artificial identity fraud',
				'composite identity theft',
				'blended identity creation',
				'fabricated credential fraud',
				'synthetic profile scams'
			],
			'Generative AI Misinformation': [
				'AI-generated false content',
				'machine learning disinformation',
				'synthetic text manipulation',
				'algorithmic fake news generation',
				'automated propaganda creation',
				'neural text generation for deception',
				'synthetic media propaganda',
				'AI-fabricated false narratives'
			],
			'Financial Fraud': [
				'AI-powered financial scams',
				'machine learning investment fraud',
				'algorithmic financial manipulation',
				'intelligent transaction fraud',
				'automated banking scams',
				'synthetic financial document fraud',
				'AI-enhanced market manipulation',
				'algorithmic credit fraud'
			],
			'Social Media Manipulation': [
				'AI-driven social engineering',
				'synthetic social network profiles',
				'machine learning influence operations',
				'algorithmic reputation manipulation',
				'coordinated inauthentic behavior',
				'automated social media fraud',
				'synthetic engagement generation',
				'AI-powered sentiment manipulation'
			],
			'Automated Impersonation': [
				'bot-based identity fraud',
				'automated credential misuse',
				'systematic impersonation attacks',
				'scalable identity spoofing',
				'programmatic account takeover',
				'AI-driven behavior mimicry',
				'automated session hijacking',
				'coordinated impersonation campaigns'
			],
			'Advanced Ransomware': [
				'AI-optimized ransom demands',
				'neural targeting of vulnerabilities',
				'machine learning encryption attacks',
				'automated data exfiltration',
				'intelligent ransomware propagation',
				'adaptive evasion techniques',
				'context-aware encryption targeting',
				'predictive vulnerability exploitation'
			]
		}
		
		# Initialize models
		try:
			# Transformer-based classification model
			self.transformer_model = self._load_transformer_model()
			
			# Zero-shot classification model
			self.zero_shot_model = SentenceTransformer('all-MiniLM-L6-v2')
			
			# ML Classifier
			self._train_initial_classifier()
		
		except Exception as e:
			self.logger.error(f"Model initialization failed: {e}")
			self.transformer_model = None
			self.zero_shot_model = None
	
	def _load_transformer_model(self):
		"""
		Load pretrained transformer model or use sentence transformer as fallback
		
		Returns:
			Loaded model
		"""
		try:
			# Try to load pretrained model
			model_path = os.path.join(self.model_path, "fraud_transformer")
			if os.path.exists(model_path):
				model = AutoModelForSequenceClassification.from_pretrained(model_path)
				tokenizer = AutoTokenizer.from_pretrained(model_path)
				return {"model": model, "tokenizer": tokenizer}
			else:
				# Use sentence transformer as fallback
				return None
		except Exception as e:
			self.logger.error(f"Error loading transformer model: {e}")
			return None
	
	def _train_initial_classifier(self):
		"""
		Train an initial machine learning classifier
		with predefined category descriptions
		"""
		# Create sample training data
		texts = []
		labels = []
		
		# Add predefined category descriptions
		for category, descriptions in self.fraud_categories.items():
			texts.extend(descriptions)
			labels.extend([category] * len(descriptions))
		
		# Prepare data
		X_train, X_test, y_train, y_test = train_test_split(
			texts, labels, test_size=0.2, random_state=42
		)
		
		# Create enhanced classification pipeline with RandomForest
		self.ml_classifier = Pipeline([
			('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
			('clf', RandomForestClassifier(n_estimators=100, random_state=42))
		])
		
		# Train classifier
		self.ml_classifier.fit(X_train, y_train)
	
	def transformer_classification(self, text):
		"""
		Classify using fine-tuned transformer model
		
		Args:
			text: Text to classify
		
		Returns:
			Predicted category and confidence score
		"""
		if not self.transformer_model:
			return None, 0.0
		
		try:
			# Tokenize and get predictions
			inputs = self.transformer_model["tokenizer"](
				text, return_tensors="pt", truncation=True, max_length=512
			)
			
			with torch.no_grad():
				outputs = self.transformer_model["model"](**inputs)
			
			# Get predicted class and confidence
			probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
			pred_class = torch.argmax(probs, dim=-1).item()
			confidence = probs[0, pred_class].item()
			
			# Map to category names
			id2label = self.transformer_model["model"].config.id2label
			category = id2label[pred_class]
			
			return category, confidence
		
		except Exception as e:
			self.logger.error(f"Transformer classification error: {e}")
			return None, 0.0
	
	def zero_shot_classification(self, text):
		"""
		Perform zero-shot classification using sentence embeddings
		
		Args:
			text: Text to classify
		
		Returns:
			Most likely category and confidence score
		"""
		if not self.zero_shot_model:
			return None, 0.0
		
		try:
			# Get embeddings for text and categories
			text_embedding = self.zero_shot_model.encode(text)
			category_embeddings = self.zero_shot_model.encode(
				list(self.fraud_categories.keys())
			)
			
			# Calculate cosine similarities
			similarities = [
				np.dot(text_embedding, cat_emb) / 
				(np.linalg.norm(text_embedding) * np.linalg.norm(cat_emb))
				for cat_emb in category_embeddings
			]
			
			# Get the most similar category
			most_similar_index = np.argmax(similarities)
			categories = list(self.fraud_categories.keys())
			max_similarity = similarities[most_similar_index]
			
			# Return if similarity is above threshold
			if max_similarity > 0.5:
				return categories[most_similar_index], max_similarity
			
			return None, 0.0
		
		except Exception as e:
			self.logger.error(f"Zero-shot classification error: {e}")
			return None, 0.0
	
	def ml_classification(self, text):
		"""
		Classify using machine learning classifier
		
		Args:
			text: Text to classify
		
		Returns:
			Predicted category and confidence score
		"""
		try:
			# Get prediction
			prediction = self.ml_classifier.predict([text])[0]
			
			# Get probability estimates
			proba = self.ml_classifier.predict_proba([text])[0]
			confidence = max(proba)
			
			return prediction, confidence
		except Exception as e:
			self.logger.error(f"ML classification error: {e}")
			return None, 0.0
	
	def classify_text(self, text):
		"""
		Comprehensive text classification method using ensemble approach
		
		Args:
			text: Text to classify
		
		Returns:
			Best classified fraud category and confidence score
		"""
		results = []
		
		# Try transformer classification
		transformer_result, transformer_conf = self.transformer_classification(text)
		if transformer_result:
			results.append((transformer_result, transformer_conf, 1.2))  # Higher weight
		
		# Try zero-shot classification
		zero_shot_result, zero_shot_conf = self.zero_shot_classification(text)
		if zero_shot_result:
			results.append((zero_shot_result, zero_shot_conf, 1.0))
		
		# Try ML classification
		ml_result, ml_conf = self.ml_classification(text)
		if ml_result:
			results.append((ml_result, ml_conf, 0.9))  # Lower weight
		
		# Combine results using weighted confidence
		if results:
			weighted_results = [(cat, conf * weight) for cat, conf, weight in results]
			best_category = max(weighted_results, key=lambda x: x[1])
			return best_category[0], best_category[1]
		
		# Fallback to keyword-based classification
		for category, keywords in self.fraud_categories.items():
			if any(keyword.lower() in text.lower() for keyword in keywords):
				return category, 0.6  # Default confidence for keyword matching
		
		return 'Unclassified AI Fraud', 0.3

			
class RiskProfiler:
	"""
	Advanced risk profiling for AI fraud threats including financial loss consideration
	"""
	def __init__(self, db_session):
		"""
		Initialize risk profiler
		
		Args:
			db_session: Database session
		"""
		self.db_session = db_session
		
		# Risk factors weights - updated to include financial impact
		self.risk_factors = {
			'victim_sensitivity': 0.20,
			'tool_sophistication': 0.20,
			'incident_frequency': 0.15,
			'financial_impact': 0.25,    # New factor with highest weight
			'trend_direction': 0.10,
			'verification_status': 0.10
		}
		
		# Victim type sensitivity scores (0-10)
		self.victim_sensitivity = {
			'Politicians': 9.0,
			'CEOs/Executives': 8.5,
			'Government Agencies': 9.5,
			'Financial Institutions': 8.0,
			'Healthcare Organizations': 8.5,
			'Military Personnel': 9.0,
			'Celebrities': 6.5,
			'Small Businesses': 6.0,
			'Elderly Individuals': 7.0,
			'General Public': 5.0,
			'Tech Professionals': 5.5,
			'Social Media Influencers': 6.0,
			'Academic Institutions': 7.0
		}
		
		# Tool sophistication scores (0-10)
		self.tool_sophistication = {
			'Deepfake Generation Software': 9.0,
			'Voice Synthesis Tools': 8.5,
			'LLM-Based Scam Generators': 8.5,
			'Automated Disinformation Systems': 8.0,
			'Social Engineering Frameworks': 7.5,
			'Identity Forgery Tools': 7.0,
			'Phishing Kits': 6.5,
			'Cryptocurrency Scam Tools': 7.0,
			'Botnet Infrastructure': 7.0,
			'Malware Frameworks': 7.5
		}
		
		# Average financial loss by category (in millions of dollars)
		self.financial_loss_by_category = {
			'Deepfake Fraud': 2.8,
			'Voice Cloning Scam': 1.5,
			'AI Phishing': 3.2,
			'Synthetic Identity Theft': 4.7,
			'Generative AI Misinformation': 1.2,
			'Financial Fraud': 5.9,
			'Social Media Manipulation': 0.9,
			'Automated Impersonation': 2.3,
			'Advanced Ransomware': 7.8
		}
	
	def calculate_risk_score(self, report, victims, tools):
		"""
		Calculate comprehensive risk score for a fraud report
		
		Args:
			report: Fraud report data
			victims: Identified victim types with confidence scores
			tools: Identified tools with confidence scores
		
		Returns:
			Risk score (0-10) and risk level
		"""
		# 1. Victim sensitivity score
		victim_score = 0
		for victim_type, confidence in victims.items():
			sensitivity = self.victim_sensitivity.get(victim_type, 5.0)
			victim_score += sensitivity * confidence
		
		# Normalize victim score (0-10)
		max_possible_victim_score = sum(self.victim_sensitivity.values())
		normalized_victim_score = min(10, (victim_score / max_possible_victim_score) * 20)
		
		# 2. Tool sophistication score
		tool_score = 0
		for tool_name, confidence in tools.items():
			sophistication = self.tool_sophistication.get(tool_name, 5.0)
			tool_score += sophistication * confidence
		
		# Normalize tool score (0-10)
		max_possible_tool_score = sum(self.tool_sophistication.values())
		normalized_tool_score = min(10, (tool_score / max_possible_tool_score) * 20)
		
		# 3. Incident frequency score
		# Check how many similar incidents occurred in last 30 days
		thirty_days_ago = datetime.now() - timedelta(days=30)
		similar_incidents = self.db_session.query(FraudReport).filter(
			FraudReport.category == report.get('category'),
			FraudReport.timestamp >= thirty_days_ago
		).count()
		
		# Convert count to score (0-10)
		frequency_score = min(10, similar_incidents / 5)
		
		# 4. Financial impact score (NEW)
		category = report.get('category', 'Unclassified AI Fraud')
		
		# Get estimated financial loss for this category
		estimated_loss = self.financial_loss_by_category.get(category, 2.0)  # Default $2M if unknown
		
		# Scale to 0-10 score (maximum is around $8M in our data)
		financial_impact_score = min(10, (estimated_loss / 8) * 10)
		
		# 5. Trend direction score
		# Check if incidents are increasing or decreasing
		sixty_days_ago = datetime.now() - timedelta(days=60)
		last_30_count = self.db_session.query(FraudReport).filter(
			FraudReport.category == report.get('category'),
			FraudReport.timestamp >= thirty_days_ago
		).count()
		
		previous_30_count = self.db_session.query(FraudReport).filter(
			FraudReport.category == report.get('category'),
			FraudReport.timestamp >= sixty_days_ago,
			FraudReport.timestamp < thirty_days_ago
		).count()
		
		# Calculate trend score (0-10)
		if previous_30_count == 0:
			trend_score = 5.0  # Neutral if no previous data
		else:
			growth_rate = (last_30_count - previous_30_count) / previous_30_count
			trend_score = 5.0 + min(5.0, max(-5.0, growth_rate * 10))
		
		# 6. Verification status (default for new reports)
		verification_score = 5.0  # Neutral for new reports
		
		# Calculate weighted risk score
		weighted_score = (
			normalized_victim_score * self.risk_factors['victim_sensitivity'] +
			normalized_tool_score * self.risk_factors['tool_sophistication'] +
			frequency_score * self.risk_factors['incident_frequency'] +
			financial_impact_score * self.risk_factors['financial_impact'] +  # New factor
			trend_score * self.risk_factors['trend_direction'] +
			verification_score * self.risk_factors['verification_status']
		)
		
		# Determine risk level
		if weighted_score >= 8.0:
			risk_level = "Critical"
		elif weighted_score >= 6.0:
			risk_level = "High"
		elif weighted_score >= 4.0:
			risk_level = "Medium"
		elif weighted_score >= 2.0:
			risk_level = "Low"
		else:
			risk_level = "Minimal"
		
		# Return the risk score, risk level, and the estimated financial loss
		return weighted_score, risk_level, estimated_loss
	
	
class AdvancedSourceScraper:
	"""
	Enhanced real-time data ingestion with multiple sources
	and advanced classification
	"""
	def __init__(self, db_session, fraud_classifier):
		"""
		Initialize the advanced source scraper
		
		Args:
			db_session: Database session
			fraud_classifier: Classification system to use
		"""
		self.logger = logging.getLogger(__name__)
		self.db_session = db_session
		self.fraud_classifier = fraud_classifier
		self.fraud_filter = AdvancedFraudFilter()
		self.risk_profiler = RiskProfiler(db_session)
		
		

# Replace your current sources list with this expanded version
		self.sources = [
			# Cybersecurity News Sources
			{
				'name': 'Krebs on Security',
				'url': 'https://krebsonsecurity.com/feed/',
				'type': 'rss'
			},
			{
				'name': 'The Hacker News',
				'url': 'https://feeds.feedburner.com/TheHackersNews',
				'type': 'rss'
			},
			{
				'name': 'Dark Reading',
				'url': 'https://www.darkreading.com/rss.xml',
				'type': 'rss'
			},
			{
				'name': 'Bleeping Computer',
				'url': 'https://www.bleepingcomputer.com/feed/',
				'type': 'rss'
			},
			{
				'name': 'SC Magazine',
				'url': 'https://www.scmagazine.com/rss-feeds/news',
				'type': 'rss'
			},
			{
				'name': 'Security Week',
				'url': 'https://feeds.feedburner.com/securityweek',
				'type': 'rss'
			},
			{
				'name': 'Infosecurity Magazine',
				'url': 'https://www.infosecurity-magazine.com/rss/news/',
				'type': 'rss'
			},
			{
				'name': 'Help Net Security',
				'url': 'https://www.helpnetsecurity.com/feed/',
				'type': 'rss'
			},
			
			# Threat Intelligence Sources
			{
				'name': 'US-CERT Alerts',
				'url': 'https://us-cert.cisa.gov/ncas/alerts.xml',
				'type': 'rss'
			},
			{
				'name': 'SANS Internet Storm Center',
				'url': 'https://isc.sans.edu/rssfeed.xml',
				'type': 'rss'
			},
			{
				'name': 'Microsoft Security Blog',
				'url': 'https://www.microsoft.com/en-us/security/blog/feed/',
				'type': 'rss'
			},
			{
				'name': 'Recorded Future',
				'url': 'https://www.recordedfuture.com/feed/',
				'type': 'rss'
			},
			
			# Security Blogs and Research
			{
				'name': 'Schneier on Security',
				'url': 'https://www.schneier.com/feed/atom/',
				'type': 'rss'
			},
			{
				'name': 'Troy Hunt',
				'url': 'https://feeds.feedburner.com/TroyHunt',
				'type': 'rss'
			},
			{
				'name': 'Google Security Blog',
				'url': 'https://security.googleblog.com/feeds/posts/default',
				'type': 'rss'
			},
			{
				'name': 'Naked Security',
				'url': 'https://nakedsecurity.sophos.com/feed/',
				'type': 'rss'
			},
			
			# AI and Security Focused
			{
				'name': 'AI Incident Database',
				'url': 'https://incidentdatabase.ai/feed/',
				'type': 'rss'
			},
			{
				'name': 'NIST Cybersecurity',
				'url': 'https://www.nist.gov/blogs/cybersecurity-insights/rss.xml',
				'type': 'rss'
			},
			{
				'name': 'OpenAI Blog',
				'url': 'https://openai.com/blog/rss/',
				'type': 'rss'
			},
			
			# Government and Regulatory
			{
				'name': 'FTC Scam Alerts',
				'url': 'https://www.consumer.ftc.gov/taxonomy/term/870/feed',
				'type': 'rss'
			},
			{
				'name': 'FBI News',
				'url': 'https://www.fbi.gov/feeds/news-and-features.rss',
				'type': 'rss'
			},
			{
				'name': 'UK NCSC',
				'url': 'https://www.ncsc.gov.uk/api/1/services/v1/report-rss-feed.xml',
				'type': 'rss'
			},
			
			
			{
				'name': 'CSO Online',
				'url': 'https://www.csoonline.com/index.rss',
				'type': 'rss'
			},
			{
				'name': 'Cyber Defense Magazine',
				'url': 'https://cyberdefensemagazine.com/feed/',
				'type': 'rss'
			},
			{
				'name': 'Infosecurity Magazine',
				'url': 'https://www.infosecurity-magazine.com/rss/news/',
				'type': 'rss'
			},
			{
				'name': 'Security Weekly',
				'url': 'https://securityweekly.com/feed/',
				'type': 'rss'
			},
			{
				'name': 'Security Affairs',
				'url': 'https://securityaffairs.com/feed',
				'type': 'rss'
			},
			
			# Government and Official Cybersecurity Sources
			{
				'name': 'NIST Cybersecurity',
				'url': 'https://www.nist.gov/blogs/cybersecurity-insights/rss.xml',
				'type': 'rss'
			},
			{
				'name': 'Australian Signals Directorate',
				'url': 'https://www.asd.gov.au/rss.xml',
				'type': 'rss'
			},
			{
				'name': 'UK National Cyber Security Centre',
				'url': 'https://www.ncsc.gov.uk/api/1/services/v1/report-rss-feed.xml',
				'type': 'rss'
			},
			
			# Technology and AI Security
			{
				'name': 'Wired Security',
				'url': 'https://www.wired.com/category/security/feed/',
				'type': 'rss'
			},
			{
				'name': 'TechCrunch Security',
				'url': 'https://techcrunch.com/category/security/feed/',
				'type': 'rss'
			},
			{
				'name': 'MIT Technology Review',
				'url': 'https://www.technologyreview.com/topic/artificial-intelligence/rss/',
				'type': 'rss'
			},
			
			# Fraud and Financial Security
			{
				'name': 'ACFE Fraud Magazine',
				'url': 'https://www.acfe.com/fraud-news.aspx?feed=rss',
				'type': 'rss'
			},
			{
				'name': 'Identity Theft Resource Center',
				'url': 'https://identitytheftresourcecenter.org/feed/',
				'type': 'rss'
			},
			
			# AI and Machine Learning Insights
			{
				'name': 'AI Trends',
				'url': 'https://www.aitrends.com/feed/',
				'type': 'rss'
			},
			{
				'name': 'VentureBeat AI',
				'url': 'https://venturebeat.com/category/ai/feed/',
				'type': 'rss'
			},
			
			# Specialized Cybersecurity Sources
			{
				'name': 'Recorded Future',
				'url': 'https://www.recordedfuture.com/feed/',
				'type': 'rss'
			},
			{
				'name': 'FireEye Threat Research',
				'url': 'https://www.fireeye.com/blog/threat-research/rss.xml',
				'type': 'rss'
			},
			{
				'name': 'Checkpoint Research',
				'url': 'https://research.checkpoint.com/feed/',
				'type': 'rss'
			},
			
			# International Cybersecurity Sources
			{
				'name': 'ITNews Australia',
				'url': 'https://www.itnews.com.au/rss/news',
				'type': 'rss'
			},
			{
				'name': 'Japan Computer Emergency Response Team',
				'url': 'https://www.jpcert.or.jp/rss/jpcert.rss',
				'type': 'rss'
			},
			
			# Academic and Research Sources
			{
				'name': 'ACM Digital Library',
				'url': 'https://dl.acm.org/action/showFeed?type=etoc&feed=rss&jc=toit',
				'type': 'rss'
			},
			{
				'name': 'arXiv Cybersecurity',
				'url': 'https://arxiv.org/rss/cs.CR',
				'type': 'rss'
			},
			
			# Cloud and Enterprise Security
			{
				'name': 'AWS Security Blog',
				'url': 'https://aws.amazon.com/blogs/security/feed/',
				'type': 'rss'
			},
			{
				'name': 'Cisco Security Blog',
				'url': 'https://blogs.cisco.com/security/rss',
				'type': 'rss'
			},
			
			# Emerging Technology Security
			{
				'name': 'ZDNet Security',
				'url': 'https://www.zdnet.com/topic/security/rss.xml',
				'type': 'rss'
			},
			{
				'name': 'Ars Technica Security',
				'url': 'https://arstechnica.com/security/feed/',
				'type': 'rss'
			},
		]
		
	def _safe_fetch_rss(self, source):
		"""
		Safely fetch RSS feed with advanced handling and better content extraction
		
		Args:
			source: Source dictionary
		
		Returns:
			List of parsed entries
		"""
		try:
			self.logger.info(f"Fetching feed from {source['name']}")
			
			# Add User-Agent to avoid being blocked
			headers = {
				'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
				'Accept': 'application/rss+xml, application/xml, text/xml, */*'
			}
			
			# First try using requests to get around potential network issues
			try:
				response = requests.get(source['url'], headers=headers, timeout=15)
				if response.status_code == 200:
					feed_content = response.content
					feed = feedparser.parse(feed_content)
				else:
					self.logger.warning(f"HTTP error {response.status_code} from {source['name']}")
					return []
			except requests.RequestException:
				# Fallback to direct parsing if requests fails
				feed = feedparser.parse(source['url'])
			
			# Check for feed errors
			if hasattr(feed, 'bozo') and feed.bozo:
				self.logger.warning(f"Feed error from {source['name']}: {feed.bozo_exception}")
			
			# Check if feed has entries
			if not hasattr(feed, 'entries') or len(feed.entries) == 0:
				self.logger.warning(f"No entries found in {source['name']} feed")
				return []
			
			# Process entries - increase to 50 for more content
			processed_entries = []
			for entry in feed.entries[:50]:
				try:
					# Skip entries without title
					if not entry.get('title'):
						continue
					
					# Extract entry date
					published_date = None
					if hasattr(entry, 'published_parsed') and entry.published_parsed:
						published_date = datetime(*entry.published_parsed[:6])
					elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
						published_date = datetime(*entry.updated_parsed[:6])
					else:
						published_date = datetime.now()
					
					# Skip entries that are too old (older than January 2025)
					if published_date < datetime(2025, 1, 1):
						continue
					
					# Get the best content available
					content = ""
					
					# Try to get content from different possible fields
					if hasattr(entry, 'content') and entry.content:
						if isinstance(entry.content, list):
							content = entry.content[0].value
						else:
							content = entry.content
					elif hasattr(entry, 'summary'):
						content = entry.summary
					elif hasattr(entry, 'description'):
						content = entry.description
					
					# Combine title and content for analysis
					full_text = f"{entry.get('title', '')} {content}"
					
					# Clean and prepare text
					cleaned_text = self.fraud_filter.clean_and_prepare_text(full_text)
					
					# Check if it's a genuine fraud report
					if not self.fraud_filter.is_fraud_report(cleaned_text):
						continue
					
					# Classify text
					try:
						# Classify text
						detected_category, confidence = self.fraud_classifier.classify_text(cleaned_text)
						
						# Lower the threshold for accepting reports
						if detected_category == 'Unclassified AI Fraud' and confidence < 0.4:
							continue
						
						# Identify potential victims
						victims = self.fraud_filter.identify_victims(cleaned_text)
						
						# Identify potential tools
						tools = self.fraud_filter.identify_tools(cleaned_text)
						
						# Calculate risk level
						risk_level = self.fraud_filter.assess_risk_level(cleaned_text, victims, tools)
						
						# Create entry
						processed_entry = {
							'source': source['name'],
							'title': entry.get('title', 'Untitled'),
							'description': content[:5000],  # Limit size but capture more
							'url': entry.get('link', ''),
							'category': detected_category,
							'confidence': confidence,
							'timestamp': published_date.isoformat(),
							'victims': victims,
							'tools': tools,
							'risk_level': risk_level
						}
						
						# Get detailed risk profile
						try:
							risk_score, risk_level, estimated_loss  = self.risk_profiler.calculate_risk_score(
								processed_entry, victims, tools
							)
							processed_entry['risk_score'] = risk_score
							processed_entry['risk_level'] = risk_level
							processed_entry['financial_loss'] = estimated_loss
							
						except Exception as e:
							self.logger.error(f"Risk profiling error: {e}")
							processed_entry['risk_score'] = 5.0  # Default risk score
							processed_entry['risk_level'] = "Medium"  # Default risk level
							processed_entry['financial_loss'] = 1.0  # Default financial loss in millions
						processed_entries.append(processed_entry)
					
					except Exception as e:
						self.logger.error(f"Classification error for {entry.get('title', 'Unknown')}: {e}")
				
				except Exception as e:
					self.logger.error(f"Error processing entry in {source['name']}: {e}")
			
			self.logger.info(f"Processed {len(feed.entries)} entries, found {len(processed_entries)} fraud reports from {source['name']}")
			return processed_entries
		
		except Exception as e:
			self.logger.error(f"Error fetching {source['name']} feed: {e}")
			return []
	
	def save_reports_to_db(self, reports):
		"""
		Save reports to database
		
		Args:
			reports: List of processed reports
		"""
		try:
			for report_data in reports:
				# Check if report already exists (avoid duplicates)
				existing = self.db_session.query(FraudReport).filter(
					FraudReport.title == report_data['title'],
					FraudReport.source == report_data['source']
				).first()
				
				if existing:
					continue
				
				# Create report
				report = FraudReport(
					source=report_data['source'],
					title=report_data['title'],
					description=report_data['description'],
					url=report_data['url'],
					category=report_data['category'],
					timestamp=datetime.now(),
					risk_level=report_data['risk_level'],
					impact_score=report_data.get('risk_score', 5.0)
				)
				
				self.db_session.add(report)
				self.db_session.flush()  # Get ID
				
				# Add victim associations
				for victim_type, confidence in report_data['victims'].items():
					# Get or create victim
					victim = self.db_session.query(Victim).filter(
						Victim.type == victim_type
					).first()
					
					if not victim:
						victim = Victim(
							type=victim_type,
							description=f"Victims classified as {victim_type}",
							vulnerability_score=self.risk_profiler.victim_sensitivity.get(victim_type, 5.0)
						)
						self.db_session.add(victim)
						self.db_session.flush()
					
					# Create association
					victim_assoc = VictimAssociation(
						report_id=report.id,
						victim_id=victim.id,
						confidence=confidence
					)
					self.db_session.add(victim_assoc)
				
				# Add tool associations
				for tool_name, confidence in report_data['tools'].items():
					# Get or create tool
					tool = self.db_session.query(Tool).filter(
						Tool.name == tool_name
					).first()
					
					if not tool:
						tool = Tool(
							name=tool_name,
							description=f"Tool identified as {tool_name}",
							sophistication_level=self.risk_profiler.tool_sophistication.get(tool_name, 5.0),
							first_observed=datetime.now()
						)
						self.db_session.add(tool)
						self.db_session.flush()
					
					# Create association
					tool_assoc = ToolAssociation(
						report_id=report.id,
						tool_id=tool.id,
						confidence=confidence
					)
					self.db_session.add(tool_assoc)
				
				self.db_session.commit()
		
		except Exception as e:
			self.logger.error(f"Error saving reports to database: {e}")
			self.db_session.rollback()
	
	def scrape_all_sources(self):
		"""
		Scrape all configured sources and save to database
		
		Returns:
			List of scraped and classified reports
		"""
		all_reports = []
		
		for source in self.sources:
			try:
				if source['type'] == 'rss':
					reports = self._safe_fetch_rss(source)
					all_reports.extend(reports)
					
					# Save immediately to avoid data loss
					self.save_reports_to_db(reports)
			except Exception as e:
				self.logger.error(f"Error processing {source['name']}: {e}")
		
		return all_reports


"""
Modified HistoricalDataCollector that targets real data from 2024-2025
"""

class HistoricalDataCollector:
	"""
	Specialized collector for historical AI fraud data from 2024-2025
	"""
	def __init__(self, data_manager):
		"""
		Initialize historical data collector
		
		Args:
			data_manager: RealTimeDataIngestionManager instance
		"""
		self.data_manager = data_manager
		self.db_session = data_manager.db_session
		self.fraud_classifier = data_manager.fraud_classifier
		self.fraud_filter = data_manager.source_scraper.fraud_filter
		self.risk_profiler = data_manager.source_scraper.risk_profiler
		self.logger = logging.getLogger(__name__)
		
		# Archives and historical sources with real 2024-2025 content
		self.historical_sources = [
			# Security blogs with existing archives
			{
				'name': 'Krebs on Security Archive',
				'url': 'https://krebsonsecurity.com/archives/',
				'type': 'archive',
				'date_format': '%B %Y'  # Archives by month
			},
			{
				'name': 'The Hacker News',
				'url': 'https://thehackernews.com/',
				'type': 'archive'
			},
			{
				'name': 'Bleeping Computer',
				'url': 'https://www.bleepingcomputer.com/news/security/',
				'type': 'archive'
			},
			{
				'name': 'Dark Reading',
				'url': 'https://www.darkreading.com/threat-intelligence',
				'type': 'archive'
			},
			{
				'name': 'SecurityWeek',
				'url': 'https://www.securityweek.com/category/security-infrastructure/',
				'type': 'archive'
			},
			{
				'name': 'ZDNet Security',
				'url': 'https://www.zdnet.com/topic/security/',
				'type': 'archive'
			},
			
			# Threat databases with historical records
			{
				'name': 'CISA Known Exploited Vulnerabilities',
				'url': 'https://www.cisa.gov/known-exploited-vulnerabilities-catalog',
				'type': 'database'
			},
			{
				'name': 'FTC Consumer Alerts',
				'url': 'https://consumer.ftc.gov/consumer-alerts',
				'type': 'archive'
			},
			
			# Security blogs with full archives
			{
				'name': 'Schneier on Security',
				'url': 'https://www.schneier.com/blog/',
				'type': 'archive'
			},
			
			# AI-focused sources
			{
				'name': 'OpenAI Blog',
				'url': 'https://openai.com/blog/',
				'type': 'archive'
			},
			{
				'name': 'Anthropic Blog',
				'url': 'https://www.anthropic.com/blog',
				'type': 'archive'
			},
			{
				'name': 'AI Safety Blog',
				'url': 'https://www.alignmentforum.org/',
				'type': 'archive'
			}
		]
	
	def collect_historical_data(self):
		"""
		Collect historical data from 2024-2025
		
		Returns:
			Number of new reports collected
		"""
		self.logger.info("Starting historical data collection from 2024-2025")
		
		# Add date range for filtering
		self.start_date = datetime(2024, 1, 1)
		
		# Add specialized request handling for historical data
		total_reports = 0
		
		for source in self.historical_sources:
			try:
				self.logger.info(f"Processing historical source: {source['name']}")
				
				if source['type'] == 'archive':
					# For archive pages, we need to use BeautifulSoup to extract links and dates
					reports = self._process_archive(source)
					
				elif source['type'] == 'database':
					# For database sources, we need specialized handling
					reports = self._process_database(source)
				
				else:
					reports = []
				
				# Add source name to all reports
				for report in reports:
					report['source'] = source['name']
				
				# Save reports to database
				saved = self.save_historical_reports(reports)
				total_reports += saved
				
				self.logger.info(f"Saved {saved} reports from {source['name']}")
				
			except Exception as e:
				self.logger.error(f"Error processing {source['name']}: {e}")
		
		self.logger.info(f"Completed historical data collection. Total new reports: {total_reports}")
		return total_reports
	
	def _process_archive(self, source):
		"""
		Process an archive page to extract historical articles
		
		Args:
			source: Source dictionary
		
		Returns:
			List of processed reports
		"""
		try:
			# Add User-Agent to avoid being blocked
			headers = {
				'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
				'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
			}
			
			# Get archive page
			response = requests.get(source['url'], headers=headers, timeout=30)
			if response.status_code != 200:
				self.logger.warning(f"HTTP error {response.status_code} from {source['name']}")
				return []
			
			# Parse HTML
			soup = BeautifulSoup(response.text, 'html.parser')
			
			# Extract links to articles (this is source-specific and may need customization)
			# We'll look for common article patterns
			links = []
			
			# Try different selectors commonly used for archives
			for selector in ['article', '.post', '.entry', '.archive-item', 'h2 a', '.title a', '.entry-title a', '.card a', '.post-title a']:
				elements = soup.select(selector)
				if elements:
					for element in elements:
						# If the element is an article container
						if element.name in ['article', 'div', 'section']:
							# Look for links inside
							article_links = element.find_all('a')
							for link in article_links:
								if link.get('href') and not link.get('href').startswith('#'):
									links.append({
										'url': link.get('href') if link.get('href').startswith('http') else f"{source['url'].split('/')[0]}//{source['url'].split('/')[2]}{link.get('href')}",
										'title': link.text.strip()
									})
						# If the element is a direct link
						elif element.name == 'a':
							if element.get('href') and not element.get('href').startswith('#'):
								links.append({
									'url': element.get('href') if element.get('href').startswith('http') else f"{source['url'].split('/')[0]}//{source['url'].split('/')[2]}{element.get('href')}",
									'title': element.text.strip()
								})
			
			# Process each article
			processed_articles = []
			for link in links[:20]:  # Limit to 20 articles per archive to avoid overloading
				try:
					# Skip if title is empty
					if not link.get('title') or len(link.get('title', '').strip()) < 5:
						continue
						
					# Check if this URL is already in our database
					existing = self.db_session.query(FraudReport).filter(
						FraudReport.url == link['url']
					).first()
					
					if existing:
						continue
					
					# Get article content
					try:
						article_response = requests.get(link['url'], headers=headers, timeout=20)
						if article_response.status_code != 200:
							continue
						
						article_soup = BeautifulSoup(article_response.text, 'html.parser')
						
						# Extract date (look for common date patterns)
						date_str = None
						date_obj = None
						
						# Try different date selectors
						for date_selector in ['.date', '.entry-date', '.post-date', 'time', '.meta time', '.post-meta', '.published', '.post-published']:
							date_element = article_soup.select_one(date_selector)
							if date_element:
								date_str = date_element.text.strip()
								break
						
						# If a date element was found, try to parse it
						if date_str:
							try:
								# Try standard date parsing first
								from dateutil import parser
								date_obj = parser.parse(date_str, fuzzy=True)
							except:
								# If that fails, try dateparser which handles more formats
								try:
									import dateparser
									date_obj = dateparser.parse(date_str)
								except:
									date_obj = None
						
						# If no date found or parsing failed, look for date in URL or path
						if not date_obj:
							import re
							date_patterns = [
								r'(\d{4}[-/]\d{2}[-/]\d{2})',  # YYYY-MM-DD or YYYY/MM/DD
								r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # YYYY-M-D or YYYY/M/D
								r'(\d{2}[-/]\d{2}[-/]\d{4})',  # MM-DD-YYYY or MM/DD/YYYY
							]
							
							for pattern in date_patterns:
								matches = re.findall(pattern, link['url'])
								if matches:
									try:
										from dateutil import parser
										date_str = matches[0]
										date_obj = parser.parse(date_str, fuzzy=True)
										break
									except:
										continue
						
						# If still no date, use current date (as a last resort)
						if not date_obj:
							date_obj = datetime.now()
						
						# Skip if the article is from before 2024
						if date_obj < self.start_date:
							continue
						
						# Extract article content
						content = ""
						
						# Try different content selectors
						for content_selector in ['.entry-content', '.post-content', 'article', '.content', '.entry', '.article-content', '.post-body']:
							content_element = article_soup.select_one(content_selector)
							if content_element:
								# Remove script and style elements
								for script in content_element.find_all(['script', 'style']):
									script.decompose()
								content = content_element.text.strip()
								break
						
						# If no content found, use the whole page text
						if not content:
							# Remove script and style elements
							for script in article_soup.find_all(['script', 'style']):
								script.decompose()
							content = article_soup.text
						
						# Combine title and content
						full_text = f"{link['title']} {content}"
						
						# Check if it's fraud related
						if not self.fraud_filter.is_fraud_report(full_text):
							continue
						
						# Classify
						category, confidence = self.fraud_classifier.classify_text(full_text)
						
						# Skip unclassified with low confidence
						if category == 'Unclassified AI Fraud' and confidence < 0.4:
							continue
						
						# Process for victims and tools
						victims = self.fraud_filter.identify_victims(full_text)
						tools = self.fraud_filter.identify_tools(full_text)
						risk_level = self.fraud_filter.assess_risk_level(full_text, victims, tools)
						
						# Create entry
						processed_entry = {
							'title': link['title'],
							'description': content[:5000],  # Limit size
							'url': link['url'],
							'category': category,
							'confidence': confidence,
							'timestamp': date_obj.isoformat(),
							'victims': victims,
							'tools': tools,
							'risk_level': risk_level
						}
						
						# Get detailed risk profile
						try:
							risk_score, risk_level, estimated_loss = self.risk_profiler.calculate_risk_score(
								processed_entry, victims, tools
							)
							processed_entry['risk_score'] = risk_score
							processed_entry['risk_level'] = risk_level
							processed_entry['financial_loss'] = estimated_loss
							
						except Exception as e:
							self.logger.error(f"Risk profiling error: {e}")
							processed_entry['risk_score'] = 5.0  # Default risk score
							processed_entry['risk_level'] = "Medium"  # Default risk level
							processed_entry['financial_loss'] = 1.0  # Default financial loss in millions
						
						processed_articles.append(processed_entry)
						
					except Exception as e:
						self.logger.warning(f"Error processing article {link['url']}: {e}")
						continue
				
				except Exception as e:
					self.logger.warning(f"Error processing link {link.get('url', 'unknown')}: {e}")
			
			return processed_articles
			
		except Exception as e:
			self.logger.error(f"Error processing archive {source['name']}: {e}")
			return []
	
	def _process_database(self, source):
		"""
		Process a database source that contains historical incidents
		
		Args:
			source: Source dictionary
		
		Returns:
			List of processed reports
		"""
		# This is highly specific to each database
		# We'll implement a generic approach that can be customized
		try:
			headers = {
				'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
			}
			
			response = requests.get(source['url'], headers=headers, timeout=30)
			if response.status_code != 200:
				self.logger.warning(f"HTTP error {response.status_code} from {source['name']}")
				return []
			
			# For CISA Known Exploited Vulnerabilities
			if 'cisa.gov' in source['url']:
				return self._process_cisa_kev(response.text)
			
			# Default processing
			soup = BeautifulSoup(response.text, 'html.parser')
			
			# Look for incident items
			incidents = []
			for selector in ['.incident', '.vulnerability', '.alert', '.article', '.item', '.entry', '.card', '.post']:
				items = soup.select(selector)
				if items:
					for item in items:
						# Extract title
						title_element = item.find(['h2', 'h3', 'h4', '.title', '.heading'])
						title = title_element.text.strip() if title_element else "Unknown Title"
						
						# Extract link
						link_element = item.find('a')
						link = link_element.get('href') if link_element else ""
						if link and not link.startswith('http'):
							link = f"{source['url'].split('/')[0]}//{source['url'].split('/')[2]}{link}"
						
						# Extract description
						desc_element = item.find(['.description', '.summary', '.content', 'p'])
						description = desc_element.text.strip() if desc_element else ""
						
						# Extract date
						date_element = item.find(['.date', 'time', '.meta'])
						date_str = date_element.text.strip() if date_element else ""
						
						date_obj = None
						if date_str:
							try:
								from dateutil import parser
								date_obj = parser.parse(date_str, fuzzy=True)
							except:
								try:
									import dateparser
									date_obj = dateparser.parse(date_str)
								except:
									date_obj = None
						
						# Try to extract date from the content
						if not date_obj:
							import re
							date_patterns = [
								r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
								r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
								r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}'  # Month DD, YYYY
							]
							
							for pattern in date_patterns:
								matches = re.findall(pattern, str(item))
								if matches:
									try:
										from dateutil import parser
										date_str = matches[0]
										date_obj = parser.parse(date_str, fuzzy=True)
										break
									except:
										continue
						
						# Skip if date is before 2024
						if date_obj and date_obj < self.start_date:
							continue
						
						# If no date found, use current date
						if not date_obj:
							date_obj = datetime.now()
						
						# Combine title and description for analysis
						full_text = f"{title} {description}"
						
						# Check if fraud-related
						if not self.fraud_filter.is_fraud_report(full_text):
							continue
						
						# Classify
						category, confidence = self.fraud_classifier.classify_text(full_text)
						
						# Process victims and tools
						victims = self.fraud_filter.identify_victims(full_text)
						tools = self.fraud_filter.identify_tools(full_text)
						risk_level = self.fraud_filter.assess_risk_level(full_text, victims, tools)
						
						incidents.append({
							'title': title,
							'description': description,
							'url': link,
							'category': category,
							'confidence': confidence,
							'timestamp': date_obj.isoformat(),
							'victims': victims,
							'tools': tools,
							'risk_level': risk_level,
							'risk_score': 7.0  # Default value
						})
			
			return incidents
			
		except Exception as e:
			self.logger.error(f"Error processing database {source['name']}: {e}")
			return []
	
	def _process_cisa_kev(self, html_content):
		"""
		Process CISA Known Exploited Vulnerabilities
		
		Args:
			html_content: HTML content from CISA
		
		Returns:
			List of processed vulnerabilities
		"""
		vulnerabilities = []
		try:
			soup = BeautifulSoup(html_content, 'html.parser')
			
			# Find vulnerability table
			table = soup.find('table')
			if not table:
				return []
			
			# Get table rows
			rows = table.find_all('tr')
			
			# Skip header row
			for row in rows[1:]:
				try:
					# Extract cells
					cells = row.find_all('td')
					if len(cells) < 4:
						continue
					
					# Extract data
					cve_id = cells[0].text.strip()
					vuln_name = cells[1].text.strip()
					description = cells[2].text.strip() if len(cells) > 2 else ""
					date_str = cells[3].text.strip() if len(cells) > 3 else ""
					
					# Parse date
					date_obj = None
					if date_str:
						try:
							from dateutil import parser
							date_obj = parser.parse(date_str, fuzzy=True)
						except:
							try:
								import dateparser
								date_obj = dateparser.parse(date_str)
							except:
								date_obj = None
					
					# Skip if date is before 2024
					if date_obj and date_obj < self.start_date:
						continue
					
					# If no date found, use current date
					if not date_obj:
						date_obj = datetime.now()
					
					# Create title
					title = f"CISA KEV Alert: {vuln_name} ({cve_id})"
					
					# Combine for analysis
					full_text = f"{title} {description}"
					
					# Check if AI-related
					if not any(term in full_text.lower() for term in ['ai', 'machine learning', 'deepfake', 'neural', 'llm', 'gpt']):
						continue
					
					# Process for standard fields
					category = "Synthetic Identity Theft"  # Most CVEs related to AI fraud fall here
					confidence = 0.7
					
					# Process victims and tools
					victims = self.fraud_filter.identify_victims(full_text)
					tools = self.fraud_filter.identify_tools(full_text)
					
					# If no victims identified, add defaults
					if not victims:
						victims = {"Technical Systems": 0.9}
					
					vulnerabilities.append({
						'title': title,
						'description': description,
						'url': f"https://nvd.nist.gov/vuln/detail/{cve_id}",
						'category': category,
						'confidence': confidence,
						'timestamp': date_obj.isoformat(),
						'victims': victims,
						'tools': tools,
						'risk_level': "High",
						'risk_score': 7.5
					})
				
				except Exception as e:
					self.logger.warning(f"Error processing vulnerability: {e}")
			
			return vulnerabilities
			
		except Exception as e:
			self.logger.error(f"Error processing CISA KEV: {e}")
			return []
	
	def save_historical_reports(self, reports):
		"""
		Save historical reports to database with proper handling of relationships
		
		Args:
			reports: List of report dictionaries
		
		Returns:
			Number of reports saved
		"""
		saved_count = 0
		
		try:
			for report_data in reports:
				# Check if report already exists
				existing = self.db_session.query(FraudReport).filter(
					FraudReport.title == report_data['title'],
					FraudReport.source == report_data['source']
				).first()
				
				if existing:
					continue
				
				# Create timestamp from string if needed
				timestamp = report_data.get('timestamp')
				if isinstance(timestamp, str):
					try:
						timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
					except ValueError:
						timestamp = datetime.now()
				elif not timestamp:
					timestamp = datetime.now()
				
				# Create report
				report = FraudReport(
					source=report_data['source'],
					title=report_data['title'],
					description=report_data['description'],
					url=report_data.get('url', ''),
					category=report_data['category'],
					timestamp=timestamp,
					risk_level=report_data['risk_level'],
					impact_score=report_data.get('risk_score', 5.0)
				)
				
				self.db_session.add(report)
				self.db_session.flush()  # Get ID
				
				# Add victim associations
				for victim_type, confidence in report_data.get('victims', {}).items():
					# Get or create victim
					victim = self.db_session.query(Victim).filter(
						Victim.type == victim_type
					).first()
					
					if not victim:
						continue  # Skip if victim type not found
					
					# Create association
					victim_assoc = VictimAssociation(
						report_id=report.id,
						victim_id=victim.id,
						confidence=confidence
					)
					self.db_session.add(victim_assoc)
				
				# Add tool associations
				for tool_name, confidence in report_data.get('tools', {}).items():
					# Get or create tool
					tool = self.db_session.query(Tool).filter(
						Tool.name == tool_name
					).first()
					
					if not tool:
						continue  # Skip if tool not found
					
					# Create association
					tool_assoc = ToolAssociation(
						report_id=report.id,
						tool_id=tool.id,
						confidence=confidence
					)
					self.db_session.add(tool_assoc)
				
				saved_count += 1
				
				# Commit in batches to avoid memory issues
				if saved_count % 10 == 0:
					self.db_session.commit()
			
			# Final commit
			self.db_session.commit()
			
		except Exception as e:
			self.logger.error(f"Error saving historical reports: {e}")
			self.db_session.rollback()
		
		return saved_count
	

class RealTimeDataIngestionManager:
	"""
	Orchestrate data collection, analysis, forecasting and dashboard integration
	"""
	def __init__(self, database_path='ai_fraud_reports.db', dashboard=None):
		"""
		Initialize data ingestion with database and dashboard integration
		
		Args:
			database_path: Path to SQLite database
			dashboard: Dashboard instance to update
		"""
		# Logging setup
		logging.basicConfig(level=logging.INFO)
		self.logger = logging.getLogger(__name__)
		
		# Initialize database
		self.db_engine = create_engine(f'sqlite:///{database_path}')
		Base.metadata.create_all(self.db_engine)
		Session = sessionmaker(bind=self.db_engine)
		self.db_session = Session()
		
		# Initialize components
		self.fraud_classifier = AdvancedFraudClassifier()
		self.source_scraper = AdvancedSourceScraper(self.db_session, self.fraud_classifier)

		
		self.novelty_detector = NoveltyDetector(self.db_session, self.fraud_classifier)
		
		# Dashboard and status
		self.dashboard = dashboard
		self.is_running = False
		self.last_collection_time = None

	
	def initialize_database(self):
		"""
		Initialize database with default victim and tool data
		"""
		try:
			# Add default victim types
			for victim_type in VICTIM_TYPES:
				existing = self.db_session.query(Victim).filter(
					Victim.type == victim_type
				).first()
				
				if not existing:
					victim = Victim(
						type=victim_type,
						description=f"Victims classified as {victim_type}",
						vulnerability_score=5.0  # Default score
					)
					self.db_session.add(victim)
			
			# Add default tools
			for tool_name in self.source_scraper.fraud_filter.tool_indicators.keys():
				existing = self.db_session.query(Tool).filter(
					Tool.name == tool_name
				).first()
				
				if not existing:
					tool = Tool(
						name=tool_name,
						description=f"Tool identified as {tool_name}",
						sophistication_level=5.0,  # Default score
						first_observed=datetime.now()
					)
					self.db_session.add(tool)
			
			self.db_session.commit()
			self.logger.info("Database initialized with default data")
		
		except Exception as e:
			self.logger.error(f"Database initialization error: {e}")
			self.db_session.rollback()
	
	
		
	def collect_data(self):
		"""
		Collect and process data from all sources
		"""
		try:
			print(f"\n[{datetime.now()}] Collecting new data from sources...")
			# Collect reports
			new_reports = self.source_scraper.scrape_all_sources()
			
			# Update dashboard if available
			if self.dashboard:
				self.dashboard.update_data()
			
			# Log collection summary
			categories = {}
			for report in new_reports:
				category = report.get('category', 'Unclassified')
				categories[category] = categories.get(category, 0) + 1
			
			summary = ", ".join([f"{count} {cat}" for cat, count in categories.items()])
			self.logger.info(f"Collected {len(new_reports)} reports: {summary}")
			
			self.last_collection_time = datetime.now()
			
			return new_reports
		
		except Exception as e:
			self.logger.error(f"Data collection error: {e}")
			return []
	
	def start_periodic_collection(self, collection_interval_minutes=15):
		"""
		Start periodic data collection and forecasting
		
		Args:
			collection_interval_minutes: Minutes between collection attempts
			forecast_interval_hours: Hours between forecast updates
		"""
		self.is_running = True
	
		collection_interval_seconds = collection_interval_minutes * 60
		
		# Initialize database with default data
		self.initialize_database()
		
		# Initial collection and forecast
		self.collect_data()
	   
		

	def get_victim_statistics(self, timeframe_days=30):
		"""
		Get victim statistics for dashboard
		
		Args:
			timeframe_days: Number of days to analyze
		
		Returns:
			Dictionary with victim statistics
		"""
		try:
			# Start date
			start_date = datetime.now() - timedelta(days=timeframe_days)
			
			# Get victim associations
			results = self.db_session.query(
				Victim.type, 
				func.count(VictimAssociation.id).label('count')
			).join(
				VictimAssociation, Victim.id == VictimAssociation.victim_id
			).join(
				FraudReport, FraudReport.id == VictimAssociation.report_id
			).filter(
				FraudReport.timestamp >= start_date
			).group_by(
				Victim.type
			).order_by(
				func.count(VictimAssociation.id).desc()
			).all()
			
			# Format results
			victim_stats = {row[0]: row[1] for row in results}
			
			return victim_stats
		
		except Exception as e:
			self.logger.error(f"Error getting victim statistics: {e}")
			return {}
	
	def get_tool_statistics(self, timeframe_days=30):
		"""
		Get tool statistics for dashboard
		
		Args:
			timeframe_days: Number of days to analyze
		
		Returns:
			Dictionary with tool statistics
		"""
		try:
			# Start date
			start_date = datetime.now() - timedelta(days=timeframe_days)
			
			# Get tool associations
			results = self.db_session.query(
				Tool.name, 
				func.count(ToolAssociation.id).label('count')
			).join(
				ToolAssociation, Tool.id == ToolAssociation.tool_id
			).join(
				FraudReport, FraudReport.id == ToolAssociation.report_id
			).filter(
				FraudReport.timestamp >= start_date
			).group_by(
				Tool.name
			).order_by(
				func.count(ToolAssociation.id).desc()
			).all()
			
			# Format results
			tool_stats = {row[0]: row[1] for row in results}
			
			return tool_stats
		
		except Exception as e:
			self.logger.error(f"Error getting tool statistics: {e}")
			return {}
	
	def get_category_statistics(self, timeframe_days=90, monthly=True):
		"""
		Get category statistics for dashboard
		
		Args:
			timeframe_days: Number of days to analyze
			monthly: Group by month
		
		Returns:
			DataFrame with category statistics
		"""
		try:
			# Start date
			start_date = datetime.now() - timedelta(days=timeframe_days)
			
			# Get reports
			reports = self.db_session.query(
				FraudReport.category,
				FraudReport.timestamp
			).filter(
				FraudReport.timestamp >= start_date
			).all()
			
			# Convert to DataFrame
			df = pd.DataFrame([(r.category, r.timestamp) for r in reports], 
							 columns=['category', 'timestamp'])
			
			if df.empty:
				return pd.DataFrame()
			
			# Group by month or day
			if monthly:
				df['period'] = df['timestamp'].dt.strftime('%Y-%m')
			else:
				df['period'] = df['timestamp'].dt.strftime('%Y-%m-%d')
			
			# Count by category and period
			pivot = pd.pivot_table(
				df, 
				values='timestamp', 
				index='period', 
				columns='category', 
				aggfunc='count',
				fill_value=0
			)
			
			return pivot
		
		except Exception as e:
			self.logger.error(f"Error getting category statistics: {e}")
			return pd.DataFrame()
	
	def get_incident_count_since_january(self):
		"""
		Get incident count since January 2025
		
		Returns:
			Dictionary with counts by category
		"""
		try:
			# Start date (January 1, 2025)
			start_date = datetime(2025, 1, 1)
			
			# Get counts by category
			results = self.db_session.query(
				FraudReport.category,
				func.count(FraudReport.id).label('count')
			).filter(
				FraudReport.timestamp >= start_date
			).group_by(
				FraudReport.category
			).all()
			
			# Format results
			counts = {row[0]: row[1] for row in results}
			
			# Add total
			counts['Total'] = sum(counts.values())
			
			return counts
		
		except Exception as e:
			self.logger.error(f"Error getting incident counts: {e}")
			return {'Total': 0}

class EnhancedAIFraudDashboard:
	"""
	Advanced interactive dashboard for AI fraud detection with rich visualizations
	and detailed analytics
	"""
	def __init__(self, data_manager):
		"""
		Initialize the dashboard with data manager for real-time analytics
		
		Args:
			data_manager: RealTimeDataIngestionManager instance
		"""

		
		self.app = dash.Dash(
			__name__,
			external_stylesheets=[dbc.themes.DARKLY],
			meta_tags=[
				{"name": "viewport", "content": "width=device-width, initial-scale=1"}
			]
		)
		self.app.title = "AI Fraud Intelligence Dashboard"
		self.data_manager = data_manager
		
		# Load Plotly template
		load_figure_template("darkly")
		
		# Setup initial dashboard layout
		self.setup_layout()
		self.register_callbacks()
				# Apply the Onyx color palette to the dashboard

		# Color palette from the provided image
		ONYX_PALETTE = {
			'onyx': '#222526',         # Main background (dark)
			'graphite': '#353A3E',     # Secondary background (dark gray)
			'platinum': '#E0E0E0',     # Light gray background
			'jet_black': '#1A1A1A',    # Deepest black
			'ash': '#BFBFBF',          # Light ash color
			'white': '#FFFFFF',        # White text
			'accent_blue': '#0071e3',  # Apple blue accent
		}
	


	def setup_layout(self):
		"""
		Create Apple-inspired dashboard layout with Onyx color palette
		Args:
			ONYX_PALETTE: Dictionary of color values
			custom_styles: Dictionary of predefined styles
		"""
		# Custom CSS for styling
		app_css = {
			'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'
		}
		
		if hasattr(self.app, 'css'):
			self.app.css.append_css(app_css)
		
		# Onyx palette
		ONYX_PALETTE = {
			'onyx': '#222526',
			'graphite': '#353A3E',
			'platinum': '#E0E0E0',
			'jet_black': '#1A1A1A',
			'ash': '#BFBFBF',
			'white': '#FFFFFF',
			'accent_blue': '#0071e3',
		}
		
		# Custom styles
		custom_styles = {
			'container': {
				'backgroundColor': ONYX_PALETTE['onyx'],
				'fontFamily': 'SF Pro Display, SF Pro Icons, Helvetica Neue, Helvetica, Arial, sans-serif',
				'color': ONYX_PALETTE['platinum'],
				'padding': '0',
				'maxWidth': '100%'
			},
			'header': {
				'backgroundColor': ONYX_PALETTE['jet_black'],
				'padding': '24px 32px',
				'borderBottom': f'1px solid {ONYX_PALETTE["graphite"]}',
				'marginBottom': '24px'
			},
			'title': {
				'fontSize': '32px',
				'fontWeight': '600',
				'marginBottom': '4px',
				'color': ONYX_PALETTE['platinum']
			},
			'subtitle': {
				'fontSize': '17px',
				'fontWeight': '400',
				'color': ONYX_PALETTE['ash']
			},
			'card': {
				'borderRadius': '16px',
				'border': f'1px solid {ONYX_PALETTE["graphite"]}',
				'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.15)',
				'backgroundColor': ONYX_PALETTE['graphite'],
				'marginBottom': '24px',
				'overflow': 'hidden'
			},
			'card_header': {
				'borderBottom': f'1px solid {ONYX_PALETTE["onyx"]}',
				'padding': '16px 20px',
				'backgroundColor': ONYX_PALETTE['graphite']
			},
			'card_title': {
				'fontSize': '18px',
				'fontWeight': '600',
				'color': ONYX_PALETTE['platinum'],
				'margin': '0'
			},
			'card_body': {
				'padding': '20px'
			},
			'button': {
				'backgroundColor': ONYX_PALETTE['accent_blue'],
				'color': ONYX_PALETTE['white'],
				'borderRadius': '20px',
				'border': 'none',
				'padding': '10px 20px',
				'fontWeight': '500',
				'fontSize': '15px',
				'cursor': 'pointer'
			}
		}
		
		self.app.layout = dbc.Container([
			# Header Section
			dbc.Row([
				dbc.Col([
					html.Div([
						html.H1("AI Fraud Intelligence", style=custom_styles['title']),
						html.P("Monitor and analyze AI-powered fraud incidents in real-time", style=custom_styles['subtitle'])
					])
				], width=12)
			], style=custom_styles['header']),
			
			# Filter Controls
			dbc.Row([
				dbc.Col([
					html.Div([
						html.Div([
							html.Label("Time Range", style={'fontWeight': '600', 'marginBottom': '8px', 'color': ONYX_PALETTE['platinum']}),
							dcc.Dropdown(
								id='time-range-selector',
								options=[
									{'label': '7 Days', 'value': 7},
									{'label': '30 Days', 'value': 30},
									{'label': '90 Days', 'value': 90},
									{'label': 'Since Jan 2025', 'value': 'year'}
								],
								value=30,
								clearable=False,
								style={
									'borderRadius': '8px',
									'backgroundColor': ONYX_PALETTE['graphite'],
									'color': ONYX_PALETTE['onyx']
								}
							)
						], style={'flex': '1', 'marginRight': '16px'}),
						html.Div([
							html.Label("Category", style={'fontWeight': '600', 'marginBottom': '8px', 'color': ONYX_PALETTE['platinum']}),
							dcc.Dropdown(
								id='category-filter',
								multi=True,
								placeholder="All Categories",
								style={
									'borderRadius': '8px',
									'backgroundColor': ONYX_PALETTE['graphite'],
									'color': ONYX_PALETTE['onyx']
								}
							)
						], style={'flex': '1', 'marginRight': '16px'}),
						html.Div([
							html.Label("Risk Level", style={'fontWeight': '600', 'marginBottom': '8px', 'color': ONYX_PALETTE['platinum']}),
							dcc.Dropdown(
								id='risk-level-filter',
								options=[
									{'label': 'All Risks', 'value': 'all'},
									{'label': 'Critical', 'value': 'Critical'},
									{'label': 'High', 'value': 'High'},
									{'label': 'Medium', 'value': 'Medium'},
									{'label': 'Low', 'value': 'Low'}
								],
								value='all',
								clearable=False,
								style={
									'borderRadius': '8px',
									'backgroundColor': ONYX_PALETTE['graphite'],
									'color': ONYX_PALETTE['onyx']
								}
							)
						], style={'flex': '1'})
					], style={'display': 'flex', 'marginBottom': '24px'})
				], width=12)
			], style={'padding': '0 32px'}),
			
			# Stats Cards
			dbc.Row([
				dbc.Col([
					html.Div(id="stats-summary", style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '16px'})
				], width=12)
			], style={'padding': '0 32px', 'marginBottom': '24px'}),
			
			# Trend Analysis & Categories
			dbc.Row([
				# Trend Analysis (taking 8 columns)
				dbc.Col([
					html.Div([
						html.Div([
							html.H5("Fraud Trend Analysis", style=custom_styles['card_title'])
						], style=custom_styles['card_header']),
						html.Div([
							dcc.Graph(
								id="trend-graph",
								config={'displayModeBar': False}
							)
						], style=custom_styles['card_body'])
					], style=custom_styles['card'])
				], width=8),
				
				# Categories (taking 4 columns) - Replaced pie chart with numeric display
				dbc.Col([
					html.Div([
						html.Div([
							html.H5("Fraud Categories", style=custom_styles['card_title'])
						], style=custom_styles['card_header']),
						html.Div([
							html.Div(id="category-card-container", style={
								'height': '100%',
								'overflow': 'auto'
							})
						], style={
							'padding': '12px',
							'height': 'calc(100% - 53px)'  # Adjusting for header
						})
					], style={**custom_styles['card'], 'height': '100%'})
				], width=4)
			], style={'padding': '0 32px'}),
			
			# Victim & Tools Analysis
			dbc.Row([
				# Victim Analysis
				dbc.Col([
					html.Div([
						html.Div([
							html.H5("Victim Analysis", style=custom_styles['card_title'])
						], style=custom_styles['card_header']),
						html.Div([
							dcc.Graph(
								id="victim-analysis-chart",
								config={'displayModeBar': False}
							)
						], style=custom_styles['card_body'])
					], style=custom_styles['card'])
				], width=6),
				
				# Tools Analysis
				dbc.Col([
					html.Div([
						html.Div([
							html.H5("Tools & Techniques", style=custom_styles['card_title'])
						], style=custom_styles['card_header']),
						html.Div([
							dcc.Graph(
								id="tools-analysis-chart",
								config={'displayModeBar': False}
							)
						], style=custom_styles['card_body'])
					], style=custom_styles['card'])
				], width=6)
			], style={'padding': '0 32px'}),
			
			# Recent Incidents Table
			dbc.Row([
				dbc.Col([
					html.Div([
						html.Div([
							html.H5("Recent Fraud Incidents", style=custom_styles['card_title'])
						], style=custom_styles['card_header']),
						html.Div([
							dash_table.DataTable(
								id='fraud-reports-table',
								columns=[
									{'name': 'Date', 'id': 'timestamp', 'type': 'text'},
									{'name': 'Source', 'id': 'source', 'type': 'text'},
									{'name': 'Title', 'id': 'title', 'type': 'text'},
									{'name': 'Category', 'id': 'category', 'type': 'text'},
									{'name': 'Risk Level', 'id': 'risk_level', 'type': 'text'},
									{'name': 'Victim Types', 'id': 'victims', 'type': 'text'},
									{'name': 'Tools Used', 'id': 'tools', 'type': 'text'}
								],
								data=[{
									'timestamp': '-',
									'source': '-',
									'title': 'Loading data...',
									'category': '-',
									'risk_level': '-',
									'victims': '-',
									'tools': '-'
								}],
								page_current=0,
								page_size=10,
								page_action='native',
								sort_action='native',
								filter_action='native',
								fixed_rows={'headers': True},
								style_as_list_view=False,
								style_table={
									'overflowX': 'auto',
									'borderRadius': '12px',
									'border': f'1px solid {ONYX_PALETTE["graphite"]}',
									'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.15)'
								},
								style_header={
									'backgroundColor': ONYX_PALETTE['jet_black'],
									'fontWeight': '600',
									'textAlign': 'left',
									'padding': '14px 16px',
									'borderBottom': f'1px solid {ONYX_PALETTE["graphite"]}',
									'fontFamily': 'SF Pro Display, SF Pro Icons, Helvetica Neue, Helvetica, Arial, sans-serif',
									'fontSize': '14px',
									'color': ONYX_PALETTE['platinum'],
									'height': '60px'
								},
								style_cell={
									'textAlign': 'left',
									'padding': '14px 16px',
									'minWidth': '140px',
									'width': 'auto',
									'maxWidth': '300px',
									'whiteSpace': 'normal',
									'overflow': 'hidden',
									'textOverflow': 'ellipsis',
									'height': 'auto',
									'backgroundColor': ONYX_PALETTE['graphite'],
									'color': ONYX_PALETTE['platinum'],
									'borderBottom': f'1px solid {ONYX_PALETTE["onyx"]}',
									'fontFamily': 'SF Pro Display, SF Pro Icons, Helvetica Neue, Helvetica, Arial, sans-serif',
									'fontSize': '14px'
								},
								style_data={
									'backgroundColor': ONYX_PALETTE['graphite'],
									'color': ONYX_PALETTE['platinum'],
									'height': 'auto',
									'minHeight': '60px'
								},
								style_data_conditional=[
									{
										'if': {'filter_query': '{risk_level} contains "Critical"'},
										'backgroundColor': 'rgba(255, 59, 48, 0.15)',
										'color': '#ff3b30',
										'fontWeight': '500'
									},
									{
										'if': {'filter_query': '{risk_level} contains "High"'},
										'backgroundColor': 'rgba(255, 149, 0, 0.15)',
										'color': '#ff9500',
										'fontWeight': '500'
									},
									{
										'if': {'filter_query': '{risk_level} contains "Medium"'},
										'backgroundColor': 'rgba(255, 204, 0, 0.15)',
										'color': '#ffc300',
										'fontWeight': '500'
									},
									{
										'if': {'filter_query': '{risk_level} contains "Low"'},
										'backgroundColor': 'rgba(52, 199, 89, 0.15)',
										'color': '#34c759',
										'fontWeight': '500'
									},
									{
										'if': {'row_index': 'odd'},
										'backgroundColor': ONYX_PALETTE['onyx']
									}
								],
								css=[
									{'selector': '.dash-spreadsheet', 'rule': f'font-family: SF Pro Display, SF Pro Icons, Helvetica Neue, Helvetica, Arial, sans-serif !important; background-color: {ONYX_PALETTE["graphite"]};'},
									{'selector': '.dash-spreadsheet-menu', 'rule': f'margin-top: 8px; background-color: {ONYX_PALETTE["graphite"]}; padding: 8px; border-radius: 8px;'},
									{'selector': '.dash-spreadsheet-menu *', 'rule': f'color: {ONYX_PALETTE["platinum"]} !important;'},
									{'selector': '.dash-filter', 'rule': 'margin-bottom: 8px;'},
									{'selector': '.dash-filter input', 'rule': f'border-radius: 8px; border: 1px solid {ONYX_PALETTE["graphite"]}; padding: 8px; width: 100%; background-color: {ONYX_PALETTE["onyx"]}; color: {ONYX_PALETTE["platinum"]};'},
									{'selector': '.dash-table-container', 'rule': 'overflow: visible !important;'},
									{'selector': '.dash-cell', 'rule': 'padding: 16px !important; text-overflow: ellipsis; overflow: hidden;'},
									{'selector': '.dash-pagination-links', 'rule': f'display: flex; justify-content: center; margin-top: 12px; font-family: SF Pro Display, Helvetica, Arial, sans-serif; color: {ONYX_PALETTE["platinum"]};'},
									{'selector': '.dash-pagination-link', 'rule': f'color: {ONYX_PALETTE["accent_blue"]} !important; border: none !important; background: none !important; padding: 6px 12px; border-radius: 8px;'},
									{'selector': '.dash-pagination-link--current', 'rule': f'background-color: {ONYX_PALETTE["accent_blue"]} !important; color: {ONYX_PALETTE["white"]} !important; border-radius: 8px;'}
								]
							)
						], style=custom_styles['card_body'])
					], style=custom_styles['card'])
				], width=12)
			], style={'padding': '0 32px', 'marginBottom': '32px'}),
			
			# Footer
			dbc.Row([
				dbc.Col([
					html.Div([
						html.P("AI Fraud Intelligence Dashboard • " + datetime.now().strftime("%B %Y"),
							style={'color': ONYX_PALETTE['ash'], 'fontSize': '14px'})
					], style={"textAlign": "center", "padding": "20px 0"})
				], width=12)
			], style={'backgroundColor': ONYX_PALETTE['jet_black'], 'marginTop': '24px'}),
			
			# Auto-refresh
			dcc.Interval(
				id='auto-refresh',
				interval=5 * 60 * 1000,  # 5 minutes
				n_intervals=0
			)
		], fluid=True, style=custom_styles['container'])



	def register_callbacks(self):
		"""
		Register dashboard update callbacks with modified category display
		"""
		@self.app.callback(
			[
				Output("stats-summary", "children"),
				Output("trend-graph", "figure"),
				Output("category-card-container", "children"),  # Updated to use card container
				Output("victim-analysis-chart", "figure"),
				Output("tools-analysis-chart", "figure"),
				
				Output("fraud-reports-table", "data"),
				Output("category-filter", "options")
			],
			[
				Input("time-range-selector", "value"),
				Input("category-filter", "value"),
				Input("risk-level-filter", "value"),
				Input("auto-refresh", "n_intervals")
			]
		)
		def update_dashboard(time_range, category_filter, risk_level, n_intervals):
			"""
			Update dashboard based on selected filters
			
			Args:
				time_range: Number of days to analyze or 'year' for since Jan 2025
				category_filter: List of selected categories or None for all
				risk_level: Selected risk level or 'all' for all levels
				n_intervals: Auto-refresh counter
			
			Returns:
				Updated visualizations and data
			"""
			# Get data from database based on filters
			if time_range == 'year':
				# Calculate days since January 1, 2025
				start_date = datetime(2025, 1, 1)
				days = (datetime.now() - start_date).days
				effective_time_range = max(1, days)  # Ensure at least 1 day
			else:
				effective_time_range = time_range
			
			# Get recent reports for table
			reports = self._get_recent_reports(effective_time_range, category_filter, risk_level)
			
			# Get categories for dropdown
			categories = self._get_all_categories()
			category_options = [{'label': cat, 'value': cat} for cat in categories]
			
			# 1. Statistics Summary
			stats_summary = self._create_stats_summary(effective_time_range)
			
			# 2. Trend Analysis
			trend_figure = self._create_trend_graph(effective_time_range, category_filter)
			
			# 3. Category Cards (replacing pie chart)
			category_cards = self._create_category_cards(effective_time_range, category_filter, risk_level)
			
			# 4. Victim Analysis
			victim_figure = self._create_victim_analysis(effective_time_range, category_filter, risk_level)
			
			# 5. Tools Analysis
			tools_figure = self._create_tools_analysis(effective_time_range, category_filter, risk_level)
			
		
			
			# Format table data
			table_data = self._format_table_data(reports)
			
			return [
				stats_summary,
				trend_figure,
				category_cards,  # Now returning HTML layout instead of figure
				victim_figure,
				tools_figure,
				
				table_data,
				category_options
			]
		
		@self.app.callback(
			Output("collection-result", "children"),
			Input("collect-data-button", "n_clicks"),
			prevent_initial_call=True
		)
		
		def manual_data_collection(n_clicks):
			if n_clicks:
				try:
					print("Manual data collection triggered")
					new_reports = self.data_manager.collect_data()
					return html.Div([
						html.P(f"Collected {len(new_reports)} new reports at {datetime.now().strftime('%H:%M:%S')}",
							className="text-success", style={"color": "#34c759"})
					])
				except Exception as e:
					print(f"Error during manual collection: {e}")
					return html.Div([
						html.P(f"Error: {str(e)}", className="text-danger", style={"color": "#ff3b30"})
					])

	def _create_category_cards(self, timeframe_days, category_filter, risk_level):
		"""
		Create numeric category display with two cards per row
		
		Args:
			timeframe_days: Number of days to analyze
			category_filter: List of selected categories or None for all
			risk_level: Selected risk level or 'all' for all levels
		
		Returns:
			HTML layout for category cards
		"""
		# Onyx palette with Jet Black and Ash from the images
		ONYX_PALETTE = {
			'onyx': '#1A1A1A',      # Jet Black
			'graphite': '#353A3E',
			'platinum': '#E0E0E0',
			'ash': '#BFBFBF',        # Ash color
			'accent_blue': '#0071e3'
		}
		
		# Color mapping for different fraud categories
		CATEGORY_COLORS = {
			'Automated Impersonation': '#ff3b30',  # Red
			'Advanced Ransomware': '#ffd60a',      # Yellow
			'AI Phishing': '#34c759',              # Green
			'Generative AI Misinformation': '#af52de',  # Purple
			'Deepfake Fraud': '#5856d6',           # Indigo
			'Voice Cloning Scam': '#ff9500',       # Orange
			'Synthetic Identity Theft': '#007aff', # Blue
			'Financial Fraud': '#ff2d55'           # Pink
		}
		
		# Average financial loss by category (in millions of dollars)
		# These are example values - you'd need to replace with real data
		FINANCIAL_LOSS = {
			'Deepfake Fraud': 2.8,
			'Voice Cloning Scam': 1.5,
			'AI Phishing': 3.2,
			'Synthetic Identity Theft': 4.7,
			'Generative AI Misinformation': 1.2,
			'Financial Fraud': 5.9,
			'Social Media Manipulation': 0.9,
			'Automated Impersonation': 2.3,
			'Advanced Ransomware': 7.8
		}
		
		# Calculate start date
		start_date = datetime.now() - timedelta(days=timeframe_days)
		
		# Build query
		query = self.data_manager.db_session.query(
			FraudReport.category,
			func.count(FraudReport.id).label('count')
		).filter(
			FraudReport.timestamp >= start_date
		)
		
		# Add risk level filter if needed
		if risk_level != 'all':
			query = query.filter(FraudReport.risk_level == risk_level)
		
		# Add category filter if needed
		if category_filter:
			query = query.filter(FraudReport.category.in_(category_filter))
		
		# Group by category and execute
		results = query.group_by(FraudReport.category).order_by(func.count(FraudReport.id).desc()).all()
		
		if not results:
			# Return empty state message
			return html.Div([
				html.Div([
					html.P("No category data available for the selected filters", 
						style={'color': ONYX_PALETTE['ash'], 'fontSize': '16px', 'textAlign': 'center'})
				], style={'padding': '40px 0'})
			])
		
		# Calculate total for percentages
		total_incidents = sum(row[1] for row in results)
		
		# Create layout with two cards per row
		rows = []
		for i in range(0, len(results), 2):
			row_cards = []
			
			# First card
			if i < len(results):
				category, count = results[i]
				percentage = (count / total_incidents) * 100 if total_incidents > 0 else 0
				color = CATEGORY_COLORS.get(category, ONYX_PALETTE['accent_blue'])
				
				# Calculate estimated financial loss
				est_loss_per_incident = FINANCIAL_LOSS.get(category, 1.0)  # Default to $1M if unknown
				total_loss = est_loss_per_incident * count
				
				first_card = html.Div([
					html.Div([
						html.Div(f"{percentage:.1f}%", style={
							'fontSize': '12px',
							'fontWeight': '500',
							'color': color,
							'textAlign': 'right',
							'marginBottom': '4px'
						}),
						html.Div(str(count), style={
							'fontSize': '24px',
							'fontWeight': '700',
							'color': ONYX_PALETTE['platinum'],
							'textAlign': 'right',
							'marginBottom': '8px'
						}),
						html.Div(category, style={
							'fontSize': '13px',
							'color': ONYX_PALETTE['ash'],
							'textAlign': 'right',
							'marginBottom': '8px'
						}),
						html.Div([
							html.Span("Est. Loss: ", style={
								'fontSize': '12px',
								'color': ONYX_PALETTE['ash'],
							}),
							html.Span(f"${total_loss:.1f}M", style={
								'fontSize': '13px',
								'fontWeight': '600',
								'color': '#ff9500',  # Orange for financial losses
							})
						], style={
							'textAlign': 'right'
						})
					], style={
						'padding': '12px',
						'borderLeft': f'3px solid {color}'
					})
				], style={
					'backgroundColor': 'rgba(191, 191, 191, 0.1)',  # Ash with low opacity
					'borderRadius': '8px',
					'flex': '1',
					'marginRight': '8px'
				})
				row_cards.append(first_card)
			
			# Second card
			if i + 1 < len(results):
				category, count = results[i + 1]
				percentage = (count / total_incidents) * 100 if total_incidents > 0 else 0
				color = CATEGORY_COLORS.get(category, ONYX_PALETTE['accent_blue'])
				
				# Calculate estimated financial loss
				est_loss_per_incident = FINANCIAL_LOSS.get(category, 1.0)  # Default to $1M if unknown
				total_loss = est_loss_per_incident * count
				
				second_card = html.Div([
					html.Div([
						html.Div(f"{percentage:.1f}%", style={
							'fontSize': '12px',
							'fontWeight': '500',
							'color': color,
							'textAlign': 'right',
							'marginBottom': '4px'
						}),
						html.Div(str(count), style={
							'fontSize': '24px',
							'fontWeight': '700',
							'color': ONYX_PALETTE['platinum'],
							'textAlign': 'right',
							'marginBottom': '8px'
						}),
						html.Div(category, style={
							'fontSize': '13px',
							'color': ONYX_PALETTE['ash'],
							'textAlign': 'right',
							'marginBottom': '8px'
						}),
						html.Div([
							html.Span("Est. Loss: ", style={
								'fontSize': '12px',
								'color': ONYX_PALETTE['ash'],
							}),
							html.Span(f"${total_loss:.1f}M", style={
								'fontSize': '13px',
								'fontWeight': '600',
								'color': '#ff9500',  # Orange for financial losses
							})
						], style={
							'textAlign': 'right'
						})
					], style={
						'padding': '12px',
						'borderLeft': f'3px solid {color}'
					})
				], style={
					'backgroundColor': 'rgba(191, 191, 191, 0.1)',  # Ash with low opacity
					'borderRadius': '8px',
					'flex': '1'
				})
				row_cards.append(second_card)
			
			# Create row with cards
			row = html.Div(row_cards, style={
				'display': 'flex',
				'marginBottom': '12px'
			})
			rows.append(row)
		
		# Container with Jet Black background
		return html.Div(rows, style={
			'backgroundColor': ONYX_PALETTE['onyx'],
			'padding': '12px',
			'borderRadius': '12px'
		})


		
	def _create_stats_summary(self, timeframe_days):
		"""
		Create statistics summary cards
		
		Args:
			timeframe_days: Number of days to analyze
		
		Returns:
			HTML layout for statistics summary
		"""
		# Get incident counts since January
		incident_counts = self.data_manager.get_incident_count_since_january()
		
		# Create stats cards
		stats_cards = dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H2(incident_counts.get('Total', 0), className="card-title text-center"),
						html.P("Total Incidents", className="card-text text-center text-muted")
					])
				], color="primary", outline=True)
			], width=3),
			
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H2(
							incident_counts.get('Deepfake Fraud', 0) + 
							incident_counts.get('Voice Cloning Scam', 0),
							className="card-title text-center"
						),
						html.P("Synthetic Media Frauds", className="card-text text-center text-muted")
					])
				], color="danger", outline=True)
			], width=3),
			
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H2(
							incident_counts.get('AI Phishing', 0) + 
							incident_counts.get('Synthetic Identity Theft', 0),
							className="card-title text-center"
						),
						html.P("Identity Frauds", className="card-text text-center text-muted")
					])
				], color="warning", outline=True)
			], width=3),
			
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H2(
							incident_counts.get('Financial Fraud', 0),
							className="card-title text-center"
						),
						html.P("Financial Frauds", className="card-text text-center text-muted")
					])
				], color="success", outline=True)
			], width=3)
		])
		
		# Add daily average card
		avg_per_day = round(incident_counts.get('Total', 0) / max(1, timeframe_days), 1)
		avg_card = dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.Div([
							html.Span(f"{avg_per_day} ", style={"fontSize": "1.5rem", "fontWeight": "bold"}),
							html.Span("incidents per day on average for the data scrapped(datascrapping on going)", style={"fontSize": "1rem"})
						], className="d-flex align-items-center justify-content-center")
					])
				], className="mt-3")
			], width=12)
		])
		
		return html.Div([stats_cards, avg_card])
	
	def _create_trend_graph(self, timeframe_days, category_filter):
		"""
		Create trend analysis graph
		
		Args:
			timeframe_days: Number of days to analyze
			category_filter: List of selected categories or None for all
		
		Returns:
			Plotly figure
		"""
		# Get category statistics (monthly if more than 30 days)
		monthly = timeframe_days > 30
		pivot_df = self.data_manager.get_category_statistics(timeframe_days, monthly)
		
		if pivot_df.empty:
			# Return empty figure
			return go.Figure().update_layout(
				title="No trend data available",
				xaxis_title="Time Period",
				yaxis_title="Incident Count"
			)
		
		# Filter categories if needed
		if category_filter:
			columns = [col for col in pivot_df.columns if col in category_filter]
			if columns:
				pivot_df = pivot_df[columns]
		
		# Create figure
		fig = go.Figure()
		
		# Color mapping for categories
		colors = px.colors.qualitative.Plotly
		
		# Add traces for each category
		for i, category in enumerate(pivot_df.columns):
			color = colors[i % len(colors)]
			fig.add_trace(go.Scatter(
				x=pivot_df.index,
				y=pivot_df[category],
				mode='lines+markers',
				name=category,
				line=dict(color=color, width=2),
				marker=dict(size=8),
				hovertemplate="%{y} incidents<extra>%{x}</extra>"
			))
		
		# Update layout
		period_type = "Month" if monthly else "Day"
		fig.update_layout(
			title=f"Incident Trends by {period_type}",
			xaxis_title=f"Time ({period_type})",
			yaxis_title="Incident Count",
			legend_title="Fraud Categories",
			hovermode="x unified",
			legend=dict(
				orientation="h",
				yanchor="bottom",
				y=1.02,
				xanchor="center",
				x=0.5
			)
		)
		
		return fig
	
	def _create_category_pie_chart(self, timeframe_days, category_filter, risk_level):
		"""
		Create category distribution pie chart
		
		Args:
			timeframe_days: Number of days to analyze
			category_filter: List of selected categories or None for all
			risk_level: Selected risk level or 'all' for all levels
		
		Returns:
			Plotly figure
		"""
		# Calculate start date
		start_date = datetime.now() - timedelta(days=timeframe_days)
		
		# Build query
		query = self.data_manager.db_session.query(
			FraudReport.category,
			func.count(FraudReport.id).label('count')
		).filter(
			FraudReport.timestamp >= start_date
		)
		
		# Add risk level filter if needed
		if risk_level != 'all':
			query = query.filter(FraudReport.risk_level == risk_level)
		
		# Add category filter if needed
		if category_filter:
			query = query.filter(FraudReport.category.in_(category_filter))
		
		# Group by category and execute
		results = query.group_by(FraudReport.category).all()
		
		# Format results
		categories = [row[0] for row in results]
		counts = [row[1] for row in results]
		
		# Create pie chart
		if not categories:
			# Return empty figure
			return go.Figure().update_layout(
				title="No category data available",
				showlegend=False
			)
		
		fig = px.pie(
			names=categories,
			values=counts,
			hole=0.4,
			color_discrete_sequence=px.colors.qualitative.Bold
		)
		
		fig.update_layout(
			title="Fraud Categories Distribution",
			legend_title="Categories",
			legend=dict(
				orientation="h",
				yanchor="bottom",
				y=-0.3,
				xanchor="center",
				x=0.5
			)
		)
		
		fig.update_traces(
			textposition='inside',
			textinfo='percent+label',
			hovertemplate='%{label}<br>%{value} incidents (%{percent})<extra></extra>'
		)
		
		return fig
	
	def _create_victim_analysis(self, timeframe_days, category_filter, risk_level):
		"""
		Create victim analysis chart
		
		Args:
			timeframe_days: Number of days to analyze
			category_filter: List of selected categories or None for all
			risk_level: Selected risk level or 'all' for all levels
		
		Returns:
			Plotly figure
		"""
		# Get victim statistics
		victim_stats = self.data_manager.get_victim_statistics(timeframe_days)
		
		if not victim_stats:
			# Return empty figure
			return go.Figure().update_layout(
				title="No victim data available",
				showlegend=False
			)
		
		# Sort by count (descending)
		sorted_victims = sorted(victim_stats.items(), key=lambda x: x[1], reverse=True)
		
		# Top 10 for readability
		top_victims = sorted_victims[:10]
		
		# Create horizontal bar chart
		fig = go.Figure()
		
		fig.add_trace(go.Bar(
			y=[item[0] for item in top_victims],
			x=[item[1] for item in top_victims],
			orientation='h',
			marker_color='rgba(55, 128, 191, 0.7)',
			hovertemplate='%{y}: %{x} incidents<extra></extra>'
		))
		
		fig.update_layout(
			title="Most Targeted Victim Types",
			xaxis_title="Incident Count",
			yaxis_title="Victim Type",
			yaxis=dict(autorange="reversed")  # Highest count at top
		)
		
		return fig
	
	def _create_tools_analysis(self, timeframe_days, category_filter, risk_level):
		"""
		Create tools analysis chart
		
		Args:
			timeframe_days: Number of days to analyze
			category_filter: List of selected categories or None for all
			risk_level: Selected risk level or 'all' for all levels
		
		Returns:
			Plotly figure
		"""
		# Get tool statistics
		tool_stats = self.data_manager.get_tool_statistics(timeframe_days)
		
		if not tool_stats:
			# Return empty figure
			return go.Figure().update_layout(
				title="No tool data available",
				showlegend=False
			)
		
		# Sort by count (descending)
		sorted_tools = sorted(tool_stats.items(), key=lambda x: x[1], reverse=True)
		
		# Top 10 for readability
		top_tools = sorted_tools[:10]
		
		# Create horizontal bar chart
		fig = go.Figure()
		
		fig.add_trace(go.Bar(
			y=[item[0] for item in top_tools],
			x=[item[1] for item in top_tools],
			orientation='h',
			marker_color='rgba(255, 128, 0, 0.7)',
			hovertemplate='%{y}: %{x} incidents<extra></extra>'
		))
		
		fig.update_layout(
			title="Most Common Fraud Tools & Techniques",
			xaxis_title="Incident Count",
			yaxis_title="Tool/Technique",
			yaxis=dict(autorange="reversed")  # Highest count at top
		)
		
		return fig
	
	def _get_recent_reports(self, timeframe_days, category_filter, risk_level):
		"""
		Get recent reports from database
		
		Args:
			timeframe_days: Number of days to analyze
			category_filter: List of selected categories or None for all
			risk_level: Selected risk level or 'all' for all levels
		
		Returns:
			List of report objects
		"""
		try:
			# Calculate start date
			start_date = datetime.now() - timedelta(days=timeframe_days)
			
			# Build query
			query = self.data_manager.db_session.query(FraudReport).filter(
				FraudReport.timestamp >= start_date
			)
			
			# Add risk level filter if needed
			if risk_level != 'all':
				query = query.filter(FraudReport.risk_level == risk_level)
			
			# Add category filter if needed
			if category_filter:
				query = query.filter(FraudReport.category.in_(category_filter))
			
			# Get reports (sorted by timestamp, newest first)
			reports = query.order_by(FraudReport.timestamp.desc()).limit(100).all()
			
			return reports
		
		except Exception as e:
			self.data_manager.logger.error(f"Error getting recent reports: {e}")
			return []
	
	def _format_table_data(self, reports):
		"""
		Format reports for data table
		
		Args:
			reports: List of report objects
		
		Returns:
			List of dictionaries for data table
		"""
		table_data = []
		
		try:
			for report in reports:
				# Get victim types
				victims = self.data_manager.db_session.query(
					Victim.type
				).join(
					VictimAssociation, 
					VictimAssociation.victim_id == Victim.id
				).filter(
					VictimAssociation.report_id == report.id
				).all()
				
				victim_types = ", ".join([v[0] for v in victims])
				
				# Get tools
				tools = self.data_manager.db_session.query(
					Tool.name
				).join(
					ToolAssociation, 
					ToolAssociation.tool_id == Tool.id
				).filter(
					ToolAssociation.report_id == report.id
				).all()
				
				tool_names = ", ".join([t[0] for t in tools])
				
				# Format timestamp
				formatted_time = report.timestamp.strftime("%Y-%m-%d %H:%M")
				
				# Prepare report data for risk profiling
				processed_entry = {
					'category': report.category,
					'timestamp': report.timestamp,
					'title': report.title
				}
				
				# Calculate risk score
				risk_score = report.impact_score if report.impact_score is not None else 5.0
				risk_level = report.risk_level if report.risk_level else "Medium"
				
				# Try to get detailed risk profile if possible
				try:
					# Identify victims and tools if needed
					victims_dict = {}
					if victim_types:
						for vtype in victim_types.split(", "):
							victims_dict[vtype] = 1.0
					
					tools_dict = {}
					if tool_names:
						for tname in tool_names.split(", "):
							tools_dict[tname] = 1.0
					
					risk_profiler = RiskProfiler(self.data_manager.db_session)
					results = risk_profiler.calculate_risk_score(
						processed_entry, victims_dict, tools_dict
					)
					
					# Handle different possible return formats
					if len(results) == 3:
						risk_score, risk_level, _ = results
					elif len(results) == 2:
						risk_score, risk_level = results
				except Exception as risk_error:
					self.data_manager.logger.warning(f"Risk profiling error: {risk_error}")
				
				# Create table row
				table_row = {
					'timestamp': formatted_time,
					'source': report.source,
					'title': report.title,
					'category': report.category,
					'risk_level': risk_level,
					'victims': victim_types,
					'tools': tool_names
				}
				
				table_data.append(table_row)
		
		except Exception as e:
			self.data_manager.logger.error(f"Error formatting table data: {e}")
		
		return table_data
	
	def _get_all_categories(self):
		"""
		Get all available categories
		
		Returns:
			List of categories
		"""
		try:
			categories = self.data_manager.db_session.query(
				FraudReport.category
			).distinct().all()
			
			return [cat[0] for cat in categories]
		
		except Exception as e:
			self.data_manager.logger.error(f"Error getting categories: {e}")
			return []
	
	def update_data(self):
		"""
		Update dashboard with new data
		(No specific action needed as data is fetched from database in callbacks)
		"""
		pass
	
	def run(self, debug=False, port=8050):
		"""
		Run the Dash application
		
		Args:
			debug: Enable debug mode
			port: Port to run the dashboard on
		"""
		self.app.run_server(debug=debug, port=port)
def setup_multi_page_dashboard(dashboard, data_manager):
	"""
	Setup a multi-page dashboard with URL-based routing
	
	Args:
		dashboard: EnhancedAIFraudDashboard instance
		data_manager: RealTimeDataIngestionManager instance
	"""
	# Onyx palette colors for consistency
	ONYX_PALETTE = {
		'onyx': '#222526',
		'graphite': '#353A3E',
		'platinum': '#E0E0E0',
		'jet_black': '#1A1A1A',
		'ash': '#BFBFBF',
		'white': '#FFFFFF',
		'accent_blue': '#0071e3',
	}
	
	# Save the original dashboard layout
	main_dashboard_layout = dashboard.app.layout
	
	# Create the novelty detection page layout
	novelty_page_layout = create_novelty_page_layout(data_manager)
	
	# Create a navigation bar
	navbar = dbc.Navbar(
		dbc.Container(
			[
				dbc.NavbarBrand("AI Fraud Intelligence", style={"color": ONYX_PALETTE['platinum']}),
				dbc.Nav(
					[
						dbc.NavItem(dbc.NavLink("Dashboard", href="/", active="exact", 
											   style={"color": ONYX_PALETTE['platinum']})),
						dbc.NavItem(dbc.NavLink("Novelty Detection", href="/novelty", active="exact",
											   style={"color": ONYX_PALETTE['platinum']})),
					],
					navbar=True,
				),
			]
		),
		color=ONYX_PALETTE['jet_black'],
		dark=True,
		className="mb-4",
	)
	
	# Create the app layout with URL-based routing
	dashboard.app.layout = html.Div([
		dcc.Location(id='url', refresh=False),
		navbar,
		html.Div(id='page-content')
	])
	
	# Register callback to handle routing
	@dashboard.app.callback(
		Output('page-content', 'children'),
		[Input('url', 'pathname')]
	)
	def display_page(pathname):
		if pathname == '/novelty':
			return novelty_page_layout
		else:
			return main_dashboard_layout.children  # Return the children of the main layout
	
	return dashboard

def create_novelty_page_layout(data_manager):
	"""
	Create the layout for the novelty detection page
	
	Args:
		data_manager: RealTimeDataIngestionManager instance
	
	Returns:
		Dash layout for the novelty detection page
	"""
	# Onyx palette for consistent styling
	ONYX_PALETTE = {
		'onyx': '#222526',
		'graphite': '#353A3E',
		'platinum': '#E0E0E0',
		'jet_black': '#1A1A1A',
		'ash': '#BFBFBF',
		'white': '#FFFFFF',
		'accent_blue': '#0071e3',
	}
	
	# Get novelty detector
	novelty_detector = data_manager.novelty_detector
	
	# Get novel incidents
	try:
		novel_incidents = novelty_detector.get_novel_incidents(timeframe_days=30, min_score=6.0)
		
		# Calculate some statistics
		total_incidents = len(novel_incidents)
		avg_score = 0
		highest_category = "None"
		
		if total_incidents > 0:
			avg_score = sum(item['novelty_data']['novelty_score'] for item in novel_incidents) / total_incidents
			
			# Get the most common category
			category_counts = {}
			for item in novel_incidents:
				category = item['report'].category
				category_counts[category] = category_counts.get(category, 0) + 1
			
			if category_counts:
				highest_category = max(category_counts.items(), key=lambda x: x[1])[0]
		
		# Format table data
		table_data = []
		for item in novel_incidents:
			report = item['report']
			novelty_data = item['novelty_data']
			
			# Format timestamp
			formatted_time = report.timestamp.strftime("%Y-%m-%d %H:%M")
			
			table_data.append({
				'timestamp': formatted_time,
				'title': report.title,
				'novelty_score': novelty_data['novelty_score'],
				'risk_level': novelty_data['risk_level'],
				'primary_classification': novelty_data['primary_classification'],
				'novel_terms': ", ".join(novelty_data['novel_terms'])
			})
	except Exception as e:
		print(f"Error getting novel incidents: {e}")
		total_incidents = 0
		avg_score = 0
		highest_category = "Error"
		table_data = []
	
	# Create layout
	return html.Div([
		# Header
		dbc.Row([
			dbc.Col([
				html.H1("Novel Fraud Pattern Detection", 
						style={'color': ONYX_PALETTE['platinum'], 'fontSize': '32px', 'fontWeight': '600'}),
				html.P("Identify emerging and unusual fraud patterns with advanced AI analysis",
					  style={'color': ONYX_PALETTE['ash'], 'fontSize': '17px'})
			])
		], style={'backgroundColor': ONYX_PALETTE['jet_black'], 'padding': '24px 32px', 'marginBottom': '24px'}),
		
		# Statistics Cards
		dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H2(total_incidents, className="card-title text-center", 
								style={'fontSize': '36px', 'fontWeight': '700'}),
						html.P("Novel Incidents Detected", className="card-text text-center text-muted")
					])
				], color="primary", outline=True)
			], width=4),
			
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H2(f"{avg_score:.1f}", className="card-title text-center", 
								style={'fontSize': '36px', 'fontWeight': '700', 'color': '#ff9500'}),
						html.P("Average Novelty Score", className="card-text text-center text-muted")
					])
				], color="warning", outline=True)
			], width=4),
			
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H2(highest_category, className="card-title text-center", 
								style={'fontSize': '28px', 'fontWeight': '600', 'color': '#ff3b30'}),
						html.P("Highest Risk Category", className="card-text text-center text-muted")
					])
				], color="danger", outline=True)
			], width=4)
		], style={'padding': '0 32px', 'marginBottom': '24px'}),
		
		# Methodology Explanation
		dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardHeader("Understanding Novelty Detection", 
								  style={'backgroundColor': ONYX_PALETTE['graphite'], 'color': ONYX_PALETTE['platinum']}),
					dbc.CardBody([
						html.P("The Novelty Detection system identifies potentially new fraud patterns that may not fit established categories, helping in early detection of emerging threats. Analysis includes:"),
						html.Ul([
							html.Li(html.Strong("Semantic Novelty: "),"How different the incident is semantically from known patterns (60% of score)"),
							html.Li(html.Strong("Vocabulary Novelty: "), "Presence of unusual or new terminology compared to existing incidents (40% of score)"),
							html.Li(html.Strong("Classification Confidence: "), "Low confidence in classification can indicate novel patterns"),
						]),
						html.P("Novelty scores range from 1 (similar to known patterns) to 10 (highly novel). Scores above 7 indicate potentially new fraud types that warrant investigation.")
					], style={'backgroundColor': ONYX_PALETTE['graphite'], 'color': ONYX_PALETTE['platinum']})
				])
			], width=12)
		], style={'padding': '0 32px', 'marginBottom': '24px'}),
		
		# Novel Incidents Table
		dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardHeader("Novel Fraud Incidents", 
								  style={'backgroundColor': ONYX_PALETTE['graphite'], 'color': ONYX_PALETTE['platinum']}),
					dbc.CardBody([
						dash_table.DataTable(
							id='novelty-incidents-table',
							columns=[
								{'name': 'Date', 'id': 'timestamp', 'type': 'text'},
								{'name': 'Title', 'id': 'title', 'type': 'text'},
								{'name': 'Novelty Score', 'id': 'novelty_score', 'type': 'numeric'},
								{'name': 'Risk Level', 'id': 'risk_level', 'type': 'text'},
								{'name': 'Category', 'id': 'primary_classification', 'type': 'text'},
								{'name': 'Novel Terms', 'id': 'novel_terms', 'type': 'text'}
							],
							data=table_data,
							page_size=10,
							style_table={
								'overflowX': 'auto',
								'borderRadius': '12px',
								'border': f'1px solid {ONYX_PALETTE["graphite"]}',
								'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.15)'
							},
							style_header={
								'backgroundColor': ONYX_PALETTE['jet_black'],
								'color': ONYX_PALETTE['platinum'],
								'fontWeight': '600'
							},
							style_cell={
								'backgroundColor': ONYX_PALETTE['graphite'],
								'color': ONYX_PALETTE['platinum'],
								'padding': '12px'
							},
							style_data_conditional=[
								{
									'if': {'column_id': 'novelty_score', 'filter_query': '{novelty_score} >= 8.5'},
									'backgroundColor': 'rgba(255, 59, 48, 0.3)',
									'color': '#ff3b30'
								},
								{
									'if': {'column_id': 'novelty_score', 'filter_query': '{novelty_score} >= 7 && {novelty_score} < 8.5'},
									'backgroundColor': 'rgba(255, 149, 0, 0.3)',
									'color': '#ff9500'
								},
								{
									'if': {'filter_query': '{risk_level} contains "Critical"'},
									'backgroundColor': 'rgba(255, 59, 48, 0.15)'
								},
								{
									'if': {'filter_query': '{risk_level} contains "High"'},
									'backgroundColor': 'rgba(255, 149, 0, 0.15)'
								}
							]
						)
					], style={'backgroundColor': ONYX_PALETTE['graphite'], 'color': ONYX_PALETTE['platinum']})
				])
			], width=12)
		], style={'padding': '0 32px', 'marginBottom': '24px'}),
		
		# Footer
		dbc.Row([
			dbc.Col([
				html.Div([
					html.P("AI Fraud Intelligence Dashboard • " + datetime.now().strftime("%B %Y"),
						style={'color': ONYX_PALETTE['ash'], 'fontSize': '14px'})
				], style={"textAlign": "center", "padding": "20px 0"})
			], width=12)
		], style={'backgroundColor': ONYX_PALETTE['jet_black'], 'marginTop': '24px'})
	], style={'backgroundColor': ONYX_PALETTE['onyx'], 'minHeight': '100vh'})
def create_novelty_page_layout(data_manager):
	"""
	Create the layout for the novelty detection page
	
	Args:
		data_manager: RealTimeDataIngestionManager instance
	
	Returns:
		Dash layout for the novelty detection page
	"""
	# Onyx palette for consistent styling
	ONYX_PALETTE = {
		'onyx': '#222526',
		'graphite': '#353A3E',
		'platinum': '#E0E0E0',
		'jet_black': '#1A1A1A',
		'ash': '#BFBFBF',
		'white': '#FFFFFF',
		'accent_blue': '#0071e3',
	}
	
	# Get novelty detector
	novelty_detector = data_manager.novelty_detector
	
	# Get novel incidents
	try:
		novel_incidents = novelty_detector.get_novel_incidents(timeframe_days=30, min_score=6.0)
		
		# Calculate some statistics
		total_incidents = len(novel_incidents)
		avg_score = 0
		highest_category = "None"
		
		if total_incidents > 0:
			avg_score = sum(item['novelty_data']['novelty_score'] for item in novel_incidents) / total_incidents
			
			# Get the most common category
			category_counts = {}
			for item in novel_incidents:
				category = item['report'].category
				category_counts[category] = category_counts.get(category, 0) + 1
			
			if category_counts:
				highest_category = max(category_counts.items(), key=lambda x: x[1])[0]
		
		# Format table data
		table_data = []
		for item in novel_incidents:
			report = item['report']
			novelty_data = item['novelty_data']
			
			# Format timestamp
			formatted_time = report.timestamp.strftime("%Y-%m-%d %H:%M")
			
			table_data.append({
				'timestamp': formatted_time,
				'title': report.title,
				'novelty_score': novelty_data['novelty_score'],
				'risk_level': novelty_data['risk_level'],
				'primary_classification': novelty_data['primary_classification'],
				'novel_terms': ", ".join(novelty_data['novel_terms'])
			})
	except Exception as e:
		print(f"Error getting novel incidents: {e}")
		total_incidents = 0
		avg_score = 0
		highest_category = "Error"
		table_data = []
	
	# Create layout
	return html.Div([
		# Header
		dbc.Row([
			dbc.Col([
				html.H1("Novel Fraud Pattern Detection", 
						style={'color': ONYX_PALETTE['platinum'], 'fontSize': '32px', 'fontWeight': '600'}),
				html.P("Identify emerging and unusual fraud patterns with advanced AI analysis",
					  style={'color': ONYX_PALETTE['ash'], 'fontSize': '17px'})
			])
		], style={'backgroundColor': ONYX_PALETTE['jet_black'], 'padding': '24px 32px', 'marginBottom': '24px'}),
		
		# Statistics Cards
		dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H2(total_incidents, className="card-title text-center", 
								style={'fontSize': '36px', 'fontWeight': '700'}),
						html.P("Novel Incidents Detected", className="card-text text-center text-muted")
					])
				], color="primary", outline=True)
			], width=4),
			
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H2(f"{avg_score:.1f}", className="card-title text-center", 
								style={'fontSize': '36px', 'fontWeight': '700', 'color': '#ff9500'}),
						html.P("Average Novelty Score", className="card-text text-center text-muted")
					])
				], color="warning", outline=True)
			], width=4),
			
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H2(highest_category, className="card-title text-center", 
								style={'fontSize': '28px', 'fontWeight': '600', 'color': '#ff3b30'}),
						html.P("Highest Risk Category", className="card-text text-center text-muted")
					])
				], color="danger", outline=True)
			], width=4)
		], style={'padding': '0 32px', 'marginBottom': '24px'}),
		
		# Methodology Explanation
		dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardHeader("Understanding Novelty Detection", 
								  style={'backgroundColor': ONYX_PALETTE['graphite'], 'color': ONYX_PALETTE['platinum']}),
					dbc.CardBody([
						html.P("The Novelty Detection system identifies potentially new fraud patterns that may not fit established categories, helping in early detection of emerging threats. Analysis includes:"),
						html.Ul([
							html.Li(html.Strong("Semantic Novelty: "),"How different the incident is semantically from known patterns (60% of score)"),
							html.Li(html.Strong("Vocabulary Novelty: "), "Presence of unusual or new terminology compared to existing incidents (40% of score)"),
							html.Li(html.Strong("Classification Confidence: "), "Low confidence in classification can indicate novel patterns"),
						]),
						html.P("Novelty scores range from 1 (similar to known patterns) to 10 (highly novel). Scores above 7 indicate potentially new fraud types that warrant investigation.")
					], style={'backgroundColor': ONYX_PALETTE['graphite'], 'color': ONYX_PALETTE['platinum']})
				])
			], width=12)
		], style={'padding': '0 32px', 'marginBottom': '24px'}),
		
		# Novel Incidents Table
		dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardHeader("Novel Fraud Incidents", 
								  style={'backgroundColor': ONYX_PALETTE['graphite'], 'color': ONYX_PALETTE['platinum']}),
					dbc.CardBody([
						dash_table.DataTable(
							id='novelty-incidents-table',
							columns=[
								{'name': 'Date', 'id': 'timestamp', 'type': 'text'},
								{'name': 'Title', 'id': 'title', 'type': 'text'},
								{'name': 'Novelty Score', 'id': 'novelty_score', 'type': 'numeric'},
								{'name': 'Risk Level', 'id': 'risk_level', 'type': 'text'},
								{'name': 'Category', 'id': 'primary_classification', 'type': 'text'},
								{'name': 'Novel Terms', 'id': 'novel_terms', 'type': 'text'}
							],
							data=table_data,
							page_size=10,
							style_table={
								'overflowX': 'auto',
								'borderRadius': '12px',
								'border': f'1px solid {ONYX_PALETTE["graphite"]}',
								'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.15)'
							},
							style_header={
								'backgroundColor': ONYX_PALETTE['jet_black'],
								'color': ONYX_PALETTE['platinum'],
								'fontWeight': '600'
							},
							style_cell={
								'backgroundColor': ONYX_PALETTE['graphite'],
								'color': ONYX_PALETTE['platinum'],
								'padding': '12px'
							},
							style_data_conditional=[
								{
									'if': {'column_id': 'novelty_score', 'filter_query': '{novelty_score} >= 8.5'},
									'backgroundColor': 'rgba(255, 59, 48, 0.3)',
									'color': '#ff3b30'
								},
								{
									'if': {'column_id': 'novelty_score', 'filter_query': '{novelty_score} >= 7 && {novelty_score} < 8.5'},
									'backgroundColor': 'rgba(255, 149, 0, 0.3)',
									'color': '#ff9500'
								},
								{
									'if': {'filter_query': '{risk_level} contains "Critical"'},
									'backgroundColor': 'rgba(255, 59, 48, 0.15)'
								},
								{
									'if': {'filter_query': '{risk_level} contains "High"'},
									'backgroundColor': 'rgba(255, 149, 0, 0.15)'
								}
							]
						)
					], style={'backgroundColor': ONYX_PALETTE['graphite'], 'color': ONYX_PALETTE['platinum']})
				])
			], width=12)
		], style={'padding': '0 32px', 'marginBottom': '24px'}),
		
		# Footer
		dbc.Row([
			dbc.Col([
				html.Div([
					html.P("AI Fraud Intelligence Dashboard • " + datetime.now().strftime("%B %Y"),
						style={'color': ONYX_PALETTE['ash'], 'fontSize': '14px'})
				], style={"textAlign": "center", "padding": "20px 0"})
			], width=12)
		], style={'backgroundColor': ONYX_PALETTE['jet_black'], 'marginTop': '24px'})
	], style={'backgroundColor': ONYX_PALETTE['onyx'], 'minHeight': '100vh'})

def seed_sample_data(data_manager):
	"""
	Seed database with sample data if it's empty
	
	Args:
		data_manager: RealTimeDataIngestionManager instance
	"""
	# Check if database is empty
	report_count = data_manager.db_session.query(FraudReport).count()
	
	if report_count > 0:
		# Database already has data
		return
	
	logger = logging.getLogger(__name__)
	logger.info("Seeding database with sample data...")
	
	# Sample data from January 2025 to now
	start_date = datetime(2025, 1, 1)
	end_date = datetime.now()
	
	# Categories with relative frequencies
	categories = {
		'Deepfake Fraud': 0.20,
		'Voice Cloning Scam': 0.15,
		'AI Phishing': 0.18,
		'Synthetic Identity Theft': 0.12,
		'Generative AI Misinformation': 0.10,
		'Financial Fraud': 0.15,
		'Social Media Manipulation': 0.07,
		'Automated Impersonation': 0.03
	}
	
	# Sources
	sources = [
		'Krebs on Security', 'The Hacker News', 'Dark Reading', 
		'Bleeping Computer', 'US-CERT Alerts', 'SANS Internet Storm Center',
		'Reddit Cybersecurity', 'Stack Exchange Security', 'AI Incident Database',
		'FTC Scam Alerts'
	]
	
	# Risk levels with relative frequencies
	risk_levels = {
		'Critical': 0.10,
		'High': 0.25,
		'Medium': 0.35,
		'Low': 0.20,
		'Minimal': 0.10
	}
	
	# Create sample data for ~100 incidents
	total_days = (end_date - start_date).days
	
	# This will create around 5-15 incidents per month
	daily_avg = 0.3  # incidents per day
	
	# Add an import that's needed
	from sqlalchemy import func
	
	# Generate reports
	for day in range(total_days):
		current_date = start_date + timedelta(days=day)
		
		# Determine number of incidents for this day (with some randomness)
		num_incidents = np.random.poisson(daily_avg)
		
		for _ in range(num_incidents):
			# Choose random category, source and risk level
			category = np.random.choice(list(categories.keys()), p=list(categories.values()))
			source = np.random.choice(sources)
			risk_level = np.random.choice(list(risk_levels.keys()), p=list(risk_levels.values()))
			
			# Create a title
			title = f"Sample {category} incident detected from {source}"
			
			# Create report
			report = FraudReport(
				source=source,
				title=title,
				description=f"This is a sample description for {category} incident",
				url="https://example.com/sample",
				category=category,
				timestamp=current_date + timedelta(hours=np.random.randint(0, 24)),
				risk_level=risk_level,
				impact_score=np.random.uniform(1, 10)
			)
			
			data_manager.db_session.add(report)
			data_manager.db_session.flush()  # To get ID
			
			# Add 1-3 random victim types
			num_victims = np.random.randint(1, 4)
			victim_types = np.random.choice(VICTIM_TYPES, size=num_victims, replace=False)
			
			for vtype in victim_types:
				# Get or create victim
				victim = data_manager.db_session.query(Victim).filter(Victim.type == vtype).first()
				
				if not victim:
					continue  # Should already be created in initialize_database
				
				# Create association
				victim_assoc = VictimAssociation(
					report_id=report.id,
					victim_id=victim.id,
					confidence=np.random.uniform(0.7, 1.0)
				)
				
				data_manager.db_session.add(victim_assoc)
			
			# Add 1-2 random tools
			num_tools = np.random.randint(1, 3)
			tool_types = np.random.choice(
				list(data_manager.source_scraper.fraud_filter.tool_indicators.keys()),
				size=min(num_tools, len(data_manager.source_scraper.fraud_filter.tool_indicators)),
				replace=False
			)
			
			for ttype in tool_types:
				# Get or create tool
				tool = data_manager.db_session.query(Tool).filter(Tool.name == ttype).first()
				
				if not tool:
					continue  # Should already be created in initialize_database
				
				# Create association
				tool_assoc = ToolAssociation(
					report_id=report.id,
					tool_id=tool.id,
					confidence=np.random.uniform(0.7, 1.0)
				)
				
				data_manager.db_session.add(tool_assoc)
	
	# Commit all changes
	data_manager.db_session.commit()



class NoveltyDetector:
	"""
	Advanced system to detect novel fraud patterns that don't match established categories
	"""
	def __init__(self, db_session, fraud_classifier):
		"""
		Initialize novelty detector
		
		Args:
			db_session: Database session
			fraud_classifier: Classification system to use
		"""
		self.db_session = db_session
		self.fraud_classifier = fraud_classifier
		self.logger = logging.getLogger(__name__)
		
		# Initialize embedding model for semantic similarity if available
		try:
			self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
		except Exception as e:
			self.logger.error(f"Error loading embedding model: {e}")
			self.embedding_model = None
		
		# Thresholds for novelty detection
		self.similarity_threshold = 0.75  # Below this is considered potentially novel
		self.minimum_confidence_threshold = 0.5  # Below this classification confidence is suspicious
		
		# Core fraud vocabulary to compare against
		self.core_fraud_vocabulary = self._build_core_vocabulary()
	
	def _build_core_vocabulary(self):
		"""
		Build vocabulary of common terms from existing fraud reports
		
		Returns:
			Dictionary of term frequencies
		"""
		# Get all existing reports
		reports = self.db_session.query(FraudReport).all()
		
		if not reports:
			# Default vocabulary if no reports exist
			return {
				"deepfake": 1.0, "voice": 1.0, "clone": 1.0, "synthetic": 1.0, 
				"ai": 1.0, "artificial": 1.0, "intelligence": 1.0, "machine": 1.0, 
				"learning": 1.0, "phishing": 1.0, "identity": 1.0, "theft": 1.0,
				"fraud": 1.0, "scam": 1.0, "impersonation": 1.0
			}
		
		# Build vocabulary from existing reports
		vocabulary = {}
		for report in reports:
			text = f"{report.title} {report.description}"
			words = text.lower().split()
			
			for word in words:
				if len(word) >= 4:  # Skip very short words
					vocabulary[word] = vocabulary.get(word, 0) + 1
		
		# Normalize frequencies
		max_freq = max(vocabulary.values()) if vocabulary else 1.0
		for word, freq in vocabulary.items():
			vocabulary[word] = freq / max_freq
		
		return vocabulary
	
	def analyze_text_novelty(self, text):
		"""
		Analyze how novel the text is compared to existing fraud reports
		
		Args:
			text: Text to analyze
		
		Returns:
			Tuple of (novelty_score, novel_terms)
		"""
		# Check word-level novelty
		words = text.lower().split()
		word_scores = []
		novel_terms = []
		
		for word in words:
			if len(word) >= 4:  # Skip very short words
				# If word is rare or unseen in our vocabulary, it's novel
				if word in self.core_fraud_vocabulary:
					word_scores.append(1.0 - self.core_fraud_vocabulary[word])
				else:
					word_scores.append(1.0)  # Completely novel word
					novel_terms.append(word)
		
		# Calculate overall novelty score from word scores
		word_novelty = sum(word_scores) / max(len(word_scores), 1)
		
		# Limit to top novel terms
		novel_terms = novel_terms[:5]
		
		return word_novelty, novel_terms
	
	def calculate_semantic_novelty(self, text):
		"""
		Calculate semantic novelty using embedding similarity
		
		Args:
			text: Text to analyze
		
		Returns:
			Semantic novelty score (0-1)
		"""
		if not self.embedding_model:
			return 0.5  # Default middle value if model not available
		
		try:
			# Get random sample of existing reports (max 50 for efficiency)
			reports = self.db_session.query(FraudReport).order_by(func.random()).limit(50).all()
			
			if not reports:
				return 0.8  # High novelty if no existing reports
			
			# Get embedding for new text
			new_embedding = self.embedding_model.encode(text)
			
			# Get embeddings for existing reports
			similarities = []
			for report in reports:
				report_text = f"{report.title} {report.description}"
				report_embedding = self.embedding_model.encode(report_text)
				
				# Calculate cosine similarity
				similarity = np.dot(new_embedding, report_embedding) / (
					np.linalg.norm(new_embedding) * np.linalg.norm(report_embedding)
				)
				similarities.append(similarity)
			
			# Find maximum similarity (closest match)
			max_similarity = max(similarities) if similarities else 0
			
			# Convert to novelty score (1 - similarity)
			semantic_novelty = 1.0 - max_similarity
			
			return semantic_novelty
			
		except Exception as e:
			self.logger.error(f"Error calculating semantic novelty: {e}")
			return 0.5  # Default to middle value
	
	def suggest_classifications(self, text, primary_classification, confidence):
		"""
		Suggest alternative classifications for potentially novel fraud
		
		Args:
			text: Text to analyze
			primary_classification: Main classification from the classifier
			confidence: Confidence in primary classification
		
		Returns:
			List of tuples (classification, confidence)
		"""
		# If confidence is high, we trust the primary classification
		if confidence >= 0.8:
			return [(primary_classification, confidence)]
		
		suggestions = [(primary_classification, confidence)]
		
		# Try zero-shot classification if available
		if hasattr(self.fraud_classifier, 'zero_shot_classification'):
			try:
				result, conf = self.fraud_classifier.zero_shot_classification(text)
				if result and result != primary_classification:
					suggestions.append((result, conf))
			except Exception:
				pass
		
		# Try ML classification
		if hasattr(self.fraud_classifier, 'ml_classification'):
			try:
				result, conf = self.fraud_classifier.ml_classification(text)
				if result and result != primary_classification and result != suggestions[-1][0]:
					suggestions.append((result, conf))
			except Exception:
				pass
		
		# Add "Novel AI Fraud" if confidence is very low
		if confidence < 0.4:
			suggestions.append(("Novel AI Fraud", 1.0 - confidence))
		
		return suggestions
	
	def analyze_novelty(self, report):
		"""
		Comprehensive novelty analysis for a fraud report
		
		Args:
			report: FraudReport object or dictionary with report data
		
		Returns:
			Dictionary with novelty analysis
		"""
		# Extract text from report
		if isinstance(report, FraudReport):
			text = f"{report.title} {report.description}"
			category = report.category
			confidence = 0.7  # Default confidence if not stored
		else:
			text = f"{report.get('title', '')} {report.get('description', '')}"
			category = report.get('category', 'Unknown')
			confidence = report.get('confidence', 0.7)
		
		# Get word-level novelty
		word_novelty, novel_terms = self.analyze_text_novelty(text)
		
		# Get semantic novelty
		semantic_novelty = self.calculate_semantic_novelty(text)
		
		# Calculate combined novelty score (weighted average)
		combined_novelty = (word_novelty * 0.4) + (semantic_novelty * 0.6)
		
		# Get classification confidence penalty
		confidence_penalty = max(0, (1.0 - confidence) * 0.5)
		
		# Final novelty score with confidence adjustment
		novelty_score = min(1.0, combined_novelty + confidence_penalty)
		
		# Determine if this is a novel pattern
		is_novel = novelty_score > 0.7
		
		# Suggest possible classifications
		suggested_classes = self.suggest_classifications(text, category, confidence)
		
		# Calculate risk level for novel pattern
		if is_novel:
			# Novel patterns start with higher risk
			if novelty_score > 0.9:
				risk_level = "Critical"
			elif novelty_score > 0.8:
				risk_level = "High"
			else:
				risk_level = "Medium"
		else:
			# Non-novel patterns keep their original risk level
			risk_level = report.risk_level if isinstance(report, FraudReport) else "Medium"
		
		return {
			'novelty_score': round(novelty_score * 10, 1),  # 0-10 scale
			'is_novel': is_novel,
			'novel_terms': novel_terms,
			'semantic_novelty': round(semantic_novelty * 10, 1),
			'word_novelty': round(word_novelty * 10, 1),
			'primary_classification': category,
			'suggested_classifications': suggested_classes,
			'risk_level': risk_level
		}
	
	def get_novel_incidents(self, timeframe_days=30, min_score=6.0, limit=20):
		"""
		Get recent incidents with high novelty scores
		
		Args:
			timeframe_days: Number of days to analyze
			min_score: Minimum novelty score (0-10)
			limit: Maximum number of reports to return
		
		Returns:
			List of reports with novelty scores
		"""
		# Calculate start date
		start_date = datetime.now() - timedelta(days=timeframe_days)
		
		# Get recent reports
		reports = self.db_session.query(FraudReport).filter(
			FraudReport.timestamp >= start_date
		).order_by(FraudReport.timestamp.desc()).limit(100).all()
		
		# Analyze novelty for each report
		scored_reports = []
		for report in reports:
			novelty_data = self.analyze_novelty(report)
			
			# Only include reports above minimum score
			if novelty_data['novelty_score'] >= min_score:
				scored_reports.append({
					'report': report,
					'novelty_data': novelty_data
				})
		
		# Sort by novelty score (descending) and limit results
		scored_reports.sort(key=lambda x: x['novelty_data']['novelty_score'], reverse=True)
		return scored_reports[:limit]
	


def add_novelty_detection_page(dashboard, data_manager):
	"""
	Add a novelty detection page to the dashboard
	
	Args:
		dashboard: EnhancedAIFraudDashboard instance
		data_manager: RealTimeDataIngestionManager instance
	"""
	# Onyx palette colors
	ONYX_PALETTE = {
		'onyx': '#222526',
		'graphite': '#353A3E',
		'platinum': '#E0E0E0',
		'jet_black': '#1A1A1A',
		'ash': '#BFBFBF',
		'white': '#FFFFFF',
		'accent_blue': '#0071e3',
	}
	
	# Custom styles (same as main dashboard)
	custom_styles = {
		'container': {
			'backgroundColor': ONYX_PALETTE['onyx'],
			'fontFamily': 'SF Pro Display, SF Pro Icons, Helvetica Neue, Helvetica, Arial, sans-serif',
			'color': ONYX_PALETTE['platinum'],
			'padding': '0',
			'maxWidth': '100%'
		},
		'header': {
			'backgroundColor': ONYX_PALETTE['jet_black'],
			'padding': '24px 32px',
			'borderBottom': f'1px solid {ONYX_PALETTE["graphite"]}',
			'marginBottom': '24px'
		},
		'title': {
			'fontSize': '32px',
			'fontWeight': '600',
			'marginBottom': '4px',
			'color': ONYX_PALETTE['platinum']
		},
		'subtitle': {
			'fontSize': '17px',
			'fontWeight': '400',
			'color': ONYX_PALETTE['ash']
		},
		'card': {
			'borderRadius': '16px',
			'border': f'1px solid {ONYX_PALETTE["graphite"]}',
			'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.15)',
			'backgroundColor': ONYX_PALETTE['graphite'],
			'marginBottom': '24px',
			'overflow': 'hidden'
		},
		'card_header': {
			'borderBottom': f'1px solid {ONYX_PALETTE["onyx"]}',
			'padding': '16px 20px',
			'backgroundColor': ONYX_PALETTE['graphite']
		},
		'card_title': {
			'fontSize': '18px',
			'fontWeight': '600',
			'color': ONYX_PALETTE['platinum'],
			'margin': '0'
		},
		'card_body': {
			'padding': '20px'
		},
		'button': {
			'backgroundColor': ONYX_PALETTE['accent_blue'],
			'color': ONYX_PALETTE['white'],
			'borderRadius': '20px',
			'border': 'none',
			'padding': '10px 20px',
			'fontWeight': '500',
			'fontSize': '15px',
			'cursor': 'pointer'
		}
	}
	
	# Create a NoveltyDetector instance
	novelty_detector = NoveltyDetector(data_manager.db_session, data_manager.fraud_classifier)
	
	# Add tab to the dashboard
	novelty_page = html.Div([
		# Header Section
		dbc.Row([
			dbc.Col([
				html.Div([
					html.H1("Novel Fraud Detection", style=custom_styles['title']),
					html.P("Identify and analyze emerging fraud patterns and techniques", 
						style=custom_styles['subtitle'])
				])
			], width=12)
		], style=custom_styles['header']),
		
		# Filter Controls
		dbc.Row([
			dbc.Col([
				html.Div([
					html.Div([
						html.Label("Time Range", style={'fontWeight': '600', 'marginBottom': '8px', 'color': ONYX_PALETTE['platinum']}),
						dcc.Dropdown(
							id='novelty-time-range',
							options=[
								{'label': '7 Days', 'value': 7},
								{'label': '30 Days', 'value': 30},
								{'label': '90 Days', 'value': 90}
							],
							value=30,
							clearable=False,
							style={
								'borderRadius': '8px', 
								'backgroundColor': ONYX_PALETTE['graphite'],
								'color': ONYX_PALETTE['onyx']
							}
						)
					], style={'flex': '1', 'marginRight': '16px'}),
					html.Div([
						html.Label("Minimum Novelty Score", style={'fontWeight': '600', 'marginBottom': '8px', 'color': ONYX_PALETTE['platinum']}),
						dcc.Slider(
							id='novelty-threshold-slider',
							min=1,
							max=10,
							step=0.5,
							value=6.0,
							marks={i: str(i) for i in range(1, 11)},
							tooltip={"placement": "bottom", "always_visible": True}
						)
					], style={'flex': '2'})
				], style={'display': 'flex', 'marginBottom': '24px'})
			], width=12)
		], style={'padding': '0 32px'}),
		
		# Novelty Score Overview Card
		dbc.Row([
			dbc.Col([
				html.Div([
					html.Div([
						html.H5("Novelty Detection Overview", style=custom_styles['card_title'])
					], style=custom_styles['card_header']),
					html.Div([
						dbc.Row([
							# Total Novel Incidents
							dbc.Col([
								html.Div([
									html.Div(id="total-novel-incidents", children="--", style={
										'fontSize': '36px',
										'fontWeight': '700',
										'color': ONYX_PALETTE['white'],
										'textAlign': 'center',
										'marginBottom': '8px'
									}),
									html.Div("Novel Incidents Detected", style={
										'fontSize': '14px',
										'color': ONYX_PALETTE['ash'],
										'textAlign': 'center'
									})
								], style={
									'backgroundColor': ONYX_PALETTE['onyx'],
									'padding': '20px',
									'borderRadius': '12px'
								})
							], width=4),
							
							# Average Novelty Score
							dbc.Col([
								html.Div([
									html.Div(id="avg-novelty-score", children="--", style={
										'fontSize': '36px',
										'fontWeight': '700',
										'color': '#ff9500',  # Orange
										'textAlign': 'center',
										'marginBottom': '8px'
									}),
									html.Div("Average Novelty Score", style={
										'fontSize': '14px',
										'color': ONYX_PALETTE['ash'],
										'textAlign': 'center'
									})
								], style={
									'backgroundColor': ONYX_PALETTE['onyx'],
									'padding': '20px',
									'borderRadius': '12px'
								})
							], width=4),
							
							# Highest Risk Category
							dbc.Col([
								html.Div([
									html.Div(id="highest-novelty-category", children="--", style={
										'fontSize': '24px',
										'fontWeight': '600',
										'color': '#ff3b30',  # Red
										'textAlign': 'center',
										'marginBottom': '8px'
									}),
									html.Div("Highest Risk Novel Category", style={
										'fontSize': '14px',
										'color': ONYX_PALETTE['ash'],
										'textAlign': 'center'
									})
								], style={
									'backgroundColor': ONYX_PALETTE['onyx'],
									'padding': '20px',
									'borderRadius': '12px'
								})
							], width=4)
						])
					], style=custom_styles['card_body'])
				], style=custom_styles['card'])
			], width=12)
		], style={'padding': '0 32px'}),
		
		# Novelty Score Distribution & Top Novel Terms
		dbc.Row([
			# Novelty Score Distribution
			dbc.Col([
				html.Div([
					html.Div([
						html.H5("Novelty Score Distribution", style=custom_styles['card_title'])
					], style=custom_styles['card_header']),
					html.Div([
						dcc.Graph(
							id="novelty-score-distribution",
							config={'displayModeBar': False}
						)
					], style=custom_styles['card_body'])
				], style=custom_styles['card'])
			], width=7),
			
			# Top Novel Terms
			dbc.Col([
				html.Div([
					html.Div([
						html.H5("Emerging Terms & Concepts", style=custom_styles['card_title'])
					], style=custom_styles['card_header']),
					html.Div([
						html.Div(id="novel-terms-container", style={
							'height': '100%',
							'overflow': 'auto'
						})
					], style={
						'padding': '12px', 
						'height': 'calc(100% - 53px)'
					})
				], style={
					**custom_styles['card'],
					'height': '100%'
				})
			], width=5)
		], style={'padding': '0 32px'}),
		
		# Novel Incidents Table with Risk Levels
		dbc.Row([
			dbc.Col([
				html.Div([
					html.Div([
						html.H5("Novel Fraud Incidents", style=custom_styles['card_title'])
					], style=custom_styles['card_header']),
					html.Div([
						# Data table with Onyx theme
						dash_table.DataTable(
							id='novelty-incidents-table',
							columns=[
								{'name': 'Date', 'id': 'timestamp', 'type': 'text'},
								{'name': 'Title', 'id': 'title', 'type': 'text'},
								{'name': 'Novelty Score', 'id': 'novelty_score', 'type': 'numeric'},
								{'name': 'Risk Level', 'id': 'risk_level', 'type': 'text'},
								{'name': 'Primary Classification', 'id': 'primary_classification', 'type': 'text'},
								{'name': 'Alt. Classifications', 'id': 'alt_classifications', 'type': 'text'}
							],
							data=[],
							sort_action='native',
							filter_action='native',
							fixed_rows={'headers': True},
							style_as_list_view=False,
							style_table={
								'overflowX': 'auto',
								'borderRadius': '12px',
								'border': f'1px solid {ONYX_PALETTE["graphite"]}',
								'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.15)'
							},
							style_header={
								'backgroundColor': ONYX_PALETTE['jet_black'],
								'fontWeight': '600',
								'textAlign': 'left',
								'padding': '14px 16px',
								'borderBottom': f'1px solid {ONYX_PALETTE["graphite"]}',
								'fontFamily': 'SF Pro Display, SF Pro Icons, Helvetica Neue, Helvetica, Arial, sans-serif',
								'fontSize': '14px',
								'color': ONYX_PALETTE['platinum'],
								'height': '60px'
							},
							style_cell={
								'textAlign': 'left',
								'padding': '14px 16px',
								'minWidth': '140px',
								'width': 'auto',
								'maxWidth': '300px',
								'whiteSpace': 'normal',
								'overflow': 'hidden',
								'textOverflow': 'ellipsis',
								'height': 'auto',
								'backgroundColor': ONYX_PALETTE['graphite'],
								'color': ONYX_PALETTE['platinum'],
								'borderBottom': f'1px solid {ONYX_PALETTE["onyx"]}',
								'fontFamily': 'SF Pro Display, SF Pro Icons, Helvetica Neue, Helvetica, Arial, sans-serif',
								'fontSize': '14px'
							},
							style_data={
								'backgroundColor': ONYX_PALETTE['graphite'],
								'color': ONYX_PALETTE['platinum'],
								'height': 'auto',
								'minHeight': '60px'
							},
							style_data_conditional=[
								# Critical risk
								{
									'if': {'filter_query': '{risk_level} contains "Critical"'},
									'backgroundColor': 'rgba(255, 59, 48, 0.15)',
									'color': '#ff3b30',
									'fontWeight': '500'
								},
								# High risk
								{
									'if': {'filter_query': '{risk_level} contains "High"'},
									'backgroundColor': 'rgba(255, 149, 0, 0.15)',
									'color': '#ff9500',
									'fontWeight': '500'
								},
								# Medium risk
								{
									'if': {'filter_query': '{risk_level} contains "Medium"'},
									'backgroundColor': 'rgba(255, 204, 0, 0.15)',
									'color': '#ffc300',
									'fontWeight': '500'
								},
								# Low risk
								{
									'if': {'filter_query': '{risk_level} contains "Low"'},
									'backgroundColor': 'rgba(52, 199, 89, 0.15)',
									'color': '#34c759',
									'fontWeight': '500'
								},
								# Novelty score highlighting
								{
									'if': {'filter_query': '{novelty_score} >= 9'},
									'backgroundColor': 'rgba(255, 59, 48, 0.15)'
								},
								{
									'if': {'filter_query': '{novelty_score} >= 7 && {novelty_score} < 9'},
									'backgroundColor': 'rgba(255, 149, 0, 0.15)'
								},
								# Alternating row colors
								{
									'if': {'row_index': 'odd'},
									'backgroundColor': ONYX_PALETTE['onyx']
								}
							]
						)
					], style=custom_styles['card_body'])
				], style=custom_styles['card'])
			], width=12)
		], style={'padding': '0 32px', 'marginBottom': '32px'}),
		
		# Explanation Section
		dbc.Row([
			dbc.Col([
				html.Div([
					html.Div([
						html.H5("Understanding Novelty Detection", style=custom_styles['card_title'])
					], style=custom_styles['card_header']),
					html.Div([
						html.P([
							"The Novelty Detection system identifies potentially new fraud patterns that may not fit established categories. This helps in early detection of emerging threats by analyzing:",
							html.Ul([
								html.Li([html.Strong("Semantic Novelty: "), "How different the incident is semantically from known patterns (60% of score)"]),
								html.Li([html.Strong("Vocabulary Novelty: "), "Presence of unusual or new terminology compared to existing incidents (40% of score)"]),
								html.Li([html.Strong("Classification Confidence: "), "Low confidence in classification can indicate novel patterns"]),
								html.Li([html.Strong("Novel Risk Assessment: "), "Risk levels for novel patterns are calculated based on novelty score and underlying threat factors"])
							]),
							"Novelty scores range from 1 (similar to known patterns) to 10 (highly novel). Scores above 7 indicate potentially new fraud types that warrant investigation."
						])
					], style=custom_styles['card_body'])
				], style={**custom_styles['card'], 'marginBottom': '24px'})
			], width=12)
		], style={'padding': '0 32px'})
	])
	
	# Register callbacks for the novelty detection page
	@dashboard.app.callback(
		[
			Output("total-novel-incidents", "children"),
			Output("avg-novelty-score", "children"),
			Output("highest-novelty-category", "children"),
			Output("novelty-score-distribution", "figure"),
			Output("novel-terms-container", "children"),
			Output("novelty-incidents-table", "data")
		],
		[
			Input("novelty-time-range", "value"),
			Input("novelty-threshold-slider", "value")
		]
	)
	def update_novelty_dashboard(time_range, min_score):
		"""
		Update novelty dashboard components based on selected filters
		
		Args:
			time_range: Number of days to analyze
			min_score: Minimum novelty score threshold
		
		Returns:
			Updated dashboard components
		"""
		# Get novel incidents
		novel_incidents = novelty_detector.get_novel_incidents(
			timeframe_days=time_range,
			min_score=min_score,
			limit=50
		)
		
		# Handle empty results
		if not novel_incidents:
			empty_fig = go.Figure()
			empty_fig.update_layout(
				title="No novel incidents found with current filters",
				xaxis_title="Novelty Score",
				yaxis_title="Count"
			)
			
			return (
				"0",  # total novel incidents
				"0.0",  # avg novelty score
				"None",  # highest novelty category
				empty_fig,  # distribution figure
				"No novel terms identified",  # novel terms
				[]  # table data
			)
		
		# 1. Total novel incidents
		total_incidents = len(novel_incidents)
		
		# 2. Average novelty score
		avg_score = sum(item['novelty_data']['novelty_score'] for item in novel_incidents) / total_incidents
		
		# 3. Highest novelty category
		category_scores = {}
		for item in novel_incidents:
			category = item['report'].category
			score = item['novelty_data']['novelty_score']
			
			if category in category_scores:
				category_scores[category].append(score)
			else:
				category_scores[category] = [score]
		
		# Calculate average score for each category
		avg_category_scores = {
			cat: sum(scores)/len(scores) for cat, scores in category_scores.items()
		}
		
		# Get category with highest average novelty score
		highest_category = max(avg_category_scores.items(), key=lambda x: x[1])[0] if avg_category_scores else "None"
		
		# 4. Novelty score distribution
		all_scores = [item['novelty_data']['novelty_score'] for item in novel_incidents]
		
		bins = list(range(int(min_score), 11, 1))
		if not bins:
			bins = [0, 10]
		
		fig = go.Figure()
		fig.add_trace(go.Histogram(
			x=all_scores,
			nbinsx=len(bins),
			marker_color='rgba(55, 128, 191, 0.7)',
			hovertemplate='Novelty Score: %{x}<br>Count: %{y}<extra></extra>'
		))
		
		fig.update_layout(
			title="Distribution of Novelty Scores",
			xaxis_title="Novelty Score",
			yaxis_title="Number of Incidents",
			bargap=0.1
		)
		
		# 5. Novel terms
		all_terms = []
		for item in novel_incidents:
			all_terms.extend(item['novelty_data']['novel_terms'])
		
		# Count term frequencies
		term_counts = {}
		for term in all_terms:
			term_counts[term] = term_counts.get(term, 0) + 1
		
		# Sort by frequency (descending)
		sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
		
		# Create term badges
		term_badges = []
		for term, count in sorted_terms[:20]:  # Limit to top 20
			term_badges.append(
				html.Span(f"{term} ({count})", style={
					'display': 'inline-block',
					'backgroundColor': 'rgba(0, 113, 227, 0.2)',
					'color': ONYX_PALETTE['white'],
					'padding': '8px 12px',
					'borderRadius': '16px',
					'margin': '4px',
					'fontSize': '14px',
					'fontWeight': '500'
				})
			)
		
		term_container = html.Div(term_badges, style={
			'display': 'flex',
			'flexWrap': 'wrap',
			'gap': '8px'
		}) if term_badges else "No distinct novel terms identified"
		
		# 6. Table data
		table_data = []
		for item in novel_incidents:
			report = item['report']
			novelty_data = item['novelty_data']
			
			# Get alternative classifications
			alt_classes = []
			for cls, conf in novelty_data['suggested_classifications'][1:]:
				alt_classes.append(f"{cls} ({conf:.2f})")
			
			alt_classifications = ", ".join(alt_classes) if alt_classes else "None"
			
			# Format timestamp
			formatted_time = report.timestamp.strftime("%Y-%m-%d %H:%M")
			
			table_data.append({
				'timestamp': formatted_time,
				'title': report.title,
				'novelty_score': novelty_data['novelty_score'],
				'risk_level': novelty_data['risk_level'],
				'primary_classification': novelty_data['primary_classification'],
				'alt_classifications': alt_classifications
			})
		
		return (
			str(total_incidents),
			f"{avg_score:.1f}",
			highest_category,
			fig,
			term_container,
			table_data
		)
	
	# Store the novelty page
	dashboard.novelty_page = novelty_page
	
	# Create tab-based layout
	tab_layout = html.Div([
		dcc.Tabs(id="dashboard-tabs", value="main-dashboard", children=[
			dcc.Tab(label="Main Dashboard", value="main-dashboard", style={
				'backgroundColor': ONYX_PALETTE['jet_black'],
				'color': ONYX_PALETTE['ash'],
				'border': f'1px solid {ONYX_PALETTE["graphite"]}',
				'borderRadius': '8px 8px 0 0',
				'padding': '12px 24px',
				'fontWeight': '500'
			}, selected_style={
				'backgroundColor': ONYX_PALETTE['onyx'],
				'color': ONYX_PALETTE['white'],
				'border': f'1px solid {ONYX_PALETTE["graphite"]}',
				'borderBottom': f'2px solid {ONYX_PALETTE["accent_blue"]}',
				'borderRadius': '8px 8px 0 0',
				'padding': '12px 24px',
				'fontWeight': '600'
			}),
			dcc.Tab(label="Novelty Detection", value="novelty-detection", style={
				'backgroundColor': ONYX_PALETTE['jet_black'],
				'color': ONYX_PALETTE['ash'],
				'border': f'1px solid {ONYX_PALETTE["graphite"]}',
				'borderRadius': '8px 8px 0 0',
				'padding': '12px 24px',
				'fontWeight': '500'
			}, selected_style={
				'backgroundColor': ONYX_PALETTE['onyx'],
				'color': ONYX_PALETTE['white'],
				'border': f'1px solid {ONYX_PALETTE["graphite"]}',
				'borderBottom': f'2px solid {ONYX_PALETTE["accent_blue"]}',
				'borderRadius': '8px 8px 0 0',
				'padding': '12px 24px',
				'fontWeight': '600'
			})
		], style={
			'borderBottom': f'1px solid {ONYX_PALETTE["graphite"]}',
			'marginBottom': '0px',
			'backgroundColor': ONYX_PALETTE['jet_black'],
			'padding': '0 32px'
		}),
		html.Div(id="dashboard-content")
	], style=custom_styles['container'])
	
	# Save reference to original layout
	original_layout = dashboard.app.layout
	dashboard.main_dashboard = original_layout
	#Create tab-based layout
	tab_layout = html.Div([
		dcc.Tabs(id="dashboard-tabs", value="main-dashboard", children=[
			dcc.Tab(label="Main Dashboard", value="main-dashboard"),
			dcc.Tab(label="Novelty Detection", value="novelty-detection")
	]),
	html.Div(id="dashboard-content")
	], style=custom_styles['container'])

	# Update the dashboard layout
	dashboard.app.layout = tab_layout
	
	# Register callback to switch between tabs
	@dashboard.app.callback(
		Output("dashboard-content", "children"),
		Input("dashboard-tabs", "value")
	)
	def render_dashboard_content(tab):
		if tab == "novelty-detection":
			return novelty_page
		else:
			return original_layout
	
	# Update auto-refresh to avoid conflicts
	@dashboard.app.callback(
		Output("dashboard-content", "style"),
		Input("auto-refresh", "n_intervals"),
		prevent_initial_call=True
	)
	def refresh_all_dashboards(n):
		return {"display": "block"}  # No actual style change
	
	
