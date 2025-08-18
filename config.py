#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生産版設定ファイル - 環境変数対応
Environment-based configuration for production use
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Streamlit import for secrets management
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

class Config:
    """Production configuration class with environment variable support"""
    
    # ==================================================
    # Base Directory Configuration
    # ==================================================
    BASE_DIR = Path(os.getenv('BASE_DIR', Path(__file__).parent))
    
    # ==================================================
    # Directory Path Settings
    # ==================================================
    INPUT_DIR = Path(os.getenv('INPUT_DIR', BASE_DIR / "input_all"))
    OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', BASE_DIR / "output"))
    TEMP_DIR = Path(os.getenv('TEMP_DIR', BASE_DIR / "temp"))
    LOG_DIR = Path(os.getenv('LOG_DIR', BASE_DIR / "logs"))
    
    # Create directories if they don't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True) 
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # ==================================================
    # File Name Settings
    # ==================================================
    OUTPUT_CSV_NAME = os.getenv('OUTPUT_CSV_NAME', "extracted_journal_entries.csv")
    OUTPUT_CSV_PATH = OUTPUT_DIR / OUTPUT_CSV_NAME
    LOG_FILE_NAME = os.getenv('LOG_FILE_NAME', "pdf_extraction.log")
    LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME
    
    # ==================================================
    # API Configuration
    # ==================================================
    
    # Anthropic Claude API (Claude Sonnet 4.0) - メインAPI
    @classmethod
    def _get_secret(cls, key: str, default: str = None) -> str:
        """Get secret from Streamlit secrets or environment variables"""
        if STREAMLIT_AVAILABLE:
            try:
                return st.secrets.get(key, os.getenv(key, default))
            except:
                return os.getenv(key, default)
        return os.getenv(key, default)
    
    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return self._get_secret('ANTHROPIC_API_KEY')
    
    ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-0')
    ANTHROPIC_MAX_TOKENS = int(os.getenv('ANTHROPIC_MAX_TOKENS', '64000'))
    ANTHROPIC_BETA_HEADERS = os.getenv('ANTHROPIC_BETA_HEADERS', 'context-1m-2025-08-07')
    
    # OCR.space API
    OCR_SPACE_API_KEY = os.getenv('OCR_SPACE_API_KEY')
    OCR_SPACE_ENDPOINT = os.getenv('OCR_SPACE_ENDPOINT', 'https://api.ocr.space/parse/image')
    
    # Azure Document Intelligence
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT')
    AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_KEY')
    
    # Google Document AI
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    GOOGLE_PROJECT_ID = os.getenv('GOOGLE_PROJECT_ID')
    
    # ==================================================
    # Processing Settings
    # ==================================================
    USE_MOCK_DATA = os.getenv('USE_MOCK_DATA', 'false').lower() == 'true'  # 本番環境はデフォルトで実API使用
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    # PDF Processing
    PAGES_PER_SPLIT = int(os.getenv('PAGES_PER_SPLIT', '5'))
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '50'))
    PDF_DPI = int(os.getenv('PDF_DPI', '300'))
    
    # API Rate Limiting
    API_REQUEST_INTERVAL = float(os.getenv('API_REQUEST_INTERVAL', '2'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '0'))  # No retries to save API costs
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '60'))
    
    # OCR Settings
    OCR_LANGUAGES = os.getenv('OCR_LANGUAGES', 'jpn+eng')
    OCR_ENGINE = os.getenv('OCR_ENGINE', 'auto')
    
    # ==================================================
    # Logging Configuration
    # ==================================================
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', '10485760'))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    MASK_API_KEYS_IN_LOGS = os.getenv('MASK_API_KEYS_IN_LOGS', 'true').lower() == 'true'
    
    # ==================================================
    # Production Safety Settings
    # ==================================================
    VALIDATE_INPUT_FILES = os.getenv('VALIDATE_INPUT_FILES', 'true').lower() == 'true'
    BACKUP_ORIGINAL_FILES = os.getenv('BACKUP_ORIGINAL_FILES', 'true').lower() == 'true'
    CLEANUP_TEMP_FILES = os.getenv('CLEANUP_TEMP_FILES', 'true').lower() == 'true'
    ENCRYPT_TEMP_FILES = os.getenv('ENCRYPT_TEMP_FILES', 'false').lower() == 'true'
    
    # ==================================================
    # Performance Settings
    # ==================================================
    MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '3'))
    WORKER_POOL_SIZE = int(os.getenv('WORKER_POOL_SIZE', '4'))
    MAX_MEMORY_MB = int(os.getenv('MAX_MEMORY_MB', '2048'))
    GARBAGE_COLLECTION_INTERVAL = int(os.getenv('GARBAGE_COLLECTION_INTERVAL', '100'))
    
    # ==================================================
    # Business Logic Settings
    # ==================================================
    VALIDATE_AMOUNTS = os.getenv('VALIDATE_AMOUNTS', 'true').lower() == 'true'
    MIN_AMOUNT = float(os.getenv('MIN_AMOUNT', '1'))
    MAX_AMOUNT = float(os.getenv('MAX_AMOUNT', '10000000'))
    
    REQUIRED_DATE_FORMAT = os.getenv('REQUIRED_DATE_FORMAT', 'R%y/%m/%d')
    VALIDATE_ACCOUNT_NAMES = os.getenv('VALIDATE_ACCOUNT_NAMES', 'true').lower() == 'true'
    
    # ==================================================
    # Error Handling
    # ==================================================
    ENABLE_RETRY = os.getenv('ENABLE_RETRY', 'false').lower() == 'true'  # Disabled to save API costs
    RETRY_DELAY = float(os.getenv('RETRY_DELAY', '5'))
    EXPONENTIAL_BACKOFF = os.getenv('EXPONENTIAL_BACKOFF', 'false').lower() == 'true'
    
    SEND_ERROR_REPORTS = os.getenv('SEND_ERROR_REPORTS', 'false').lower() == 'true'
    ERROR_REPORT_EMAIL = os.getenv('ERROR_REPORT_EMAIL')
    
    # ==================================================
    # Development Settings
    # ==================================================
    DEVELOPMENT_MODE = os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true'
    RUN_UNIT_TESTS = os.getenv('RUN_UNIT_TESTS', 'false').lower() == 'true'
    GENERATE_TEST_REPORTS = os.getenv('GENERATE_TEST_REPORTS', 'false').lower() == 'true'
    
    # ==================================================
    # CSV Field Definitions
    # ==================================================
    CSV_COLUMNS = [
        "契約日",
        "借方科目", 
        "貸方科目",
        "摘要",
        "金額",
        "備考",
        "参照元ファイル"
    ]
    
    REQUIRED_FIELDS = ["契約日", "借方科目", "貸方科目", "摘要", "金額"]
    
    # ==================================================
    # Target PDF File Patterns
    # ==================================================
    PDF_FILE_PATTERNS = [
        "西京（2月）.pdf",
        "退去（2月）.pdf", 
        "家賃（2月）.pdf",
        "更新（2月）.pdf",
        "三菱（2月）.pdf",
        "振替（2月）.pdf", 
        "新規（2月）.pdf",
        "*.pdf"  # Accept all PDF files
    ]
    
    # ==================================================
    # Methods for Configuration Validation
    # ==================================================
    
    @classmethod
    def validate_configuration(cls) -> List[str]:
        """
        Validate the configuration and return list of issues
        
        Returns:
            List of configuration issues/warnings
        """
        issues = []
        
        # API Key validation
        if not cls.USE_MOCK_DATA:
            # Check if at least one API key is configured
            config_instance = cls()
            anthropic_key = config_instance.ANTHROPIC_API_KEY
            has_anthropic = anthropic_key and anthropic_key != 'DUMMY_API_KEY'
            
            if not has_anthropic:
                issues.append("ANTHROPIC_API_KEY is required for production use")
            
            # Validate key formats
            if anthropic_key and anthropic_key != 'DUMMY_API_KEY' and not anthropic_key.startswith('sk-ant-'):
                issues.append("ANTHROPIC_API_KEY format appears invalid (should start with 'sk-ant-')")
        
        # Directory validation
        if not cls.INPUT_DIR.exists():
            issues.append(f"Input directory does not exist: {cls.INPUT_DIR}")
        
        # File size validation
        if cls.MAX_FILE_SIZE_MB <= 0:
            issues.append("MAX_FILE_SIZE_MB must be greater than 0")
        
        # Processing validation
        if cls.PAGES_PER_SPLIT <= 0:
            issues.append("PAGES_PER_SPLIT must be greater than 0")
        
        if cls.API_REQUEST_INTERVAL < 0:
            issues.append("API_REQUEST_INTERVAL cannot be negative")
        
        # Amount validation
        if cls.MIN_AMOUNT < 0:
            issues.append("MIN_AMOUNT cannot be negative")
        
        if cls.MAX_AMOUNT <= cls.MIN_AMOUNT:
            issues.append("MAX_AMOUNT must be greater than MIN_AMOUNT")
        
        return issues
    
    @classmethod
    def get_api_key_summary(cls) -> Dict[str, str]:
        """
        Get summary of API key configuration (masked for security)
        
        Returns:
            Dictionary with API key status
        """
        def mask_key(key: Optional[str]) -> str:
            if not key or key == 'DUMMY_API_KEY':
                return "Not configured"
            return f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "Configured"
        
        config_instance = cls()
        return {
            'anthropic': mask_key(config_instance.ANTHROPIC_API_KEY),
            'ocr_space': mask_key(cls.OCR_SPACE_API_KEY),
            'azure': mask_key(cls.AZURE_DOCUMENT_INTELLIGENCE_KEY),
        }
    
    @classmethod
    def is_production_ready(cls) -> bool:
        """
        Check if configuration is ready for production use
        
        Returns:
            True if ready for production
        """
        if cls.USE_MOCK_DATA:
            return False
            
        # Check if at least one API key is properly configured
        config_instance = cls()
        anthropic_key = config_instance.ANTHROPIC_API_KEY
        has_anthropic = anthropic_key and anthropic_key != 'DUMMY_API_KEY'
        
        if not has_anthropic:
            return False
            
        if not cls.INPUT_DIR.exists():
            return False
            
        return len(cls.validate_configuration()) == 0
    
    @classmethod
    def setup_logging(cls) -> logging.Logger:
        """
        Setup production logging configuration
        
        Returns:
            Configured logger
        """
        # Create logger
        logger = logging.getLogger('pdf_extractor')
        logger.setLevel(getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            cls.LOG_FILE_PATH,
            maxBytes=cls.LOG_MAX_BYTES,
            backupCount=cls.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(cls.LOG_FORMAT))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(cls.LOG_FORMAT))
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

# Create global config instance
config = Config()

# Legacy compatibility - keep existing variable names
BASE_DIR = config.BASE_DIR
INPUT_DIR = config.INPUT_DIR
OUTPUT_DIR = config.OUTPUT_DIR
TEMP_DIR = config.TEMP_DIR
OUTPUT_CSV_NAME = config.OUTPUT_CSV_NAME
OUTPUT_CSV_PATH = config.OUTPUT_CSV_PATH
LOG_FILE_NAME = config.LOG_FILE_NAME
ANTHROPIC_API_KEY = lambda: config.ANTHROPIC_API_KEY
USE_MOCK_DATA = config.USE_MOCK_DATA
PAGES_PER_SPLIT = config.PAGES_PER_SPLIT
API_REQUEST_INTERVAL = config.API_REQUEST_INTERVAL
MAX_TOKENS = config.ANTHROPIC_MAX_TOKENS
MODEL_NAME = config.ANTHROPIC_MODEL
LOG_LEVEL = config.LOG_LEVEL
LOG_FORMAT = config.LOG_FORMAT
CSV_COLUMNS = config.CSV_COLUMNS
PDF_FILE_PATTERNS = config.PDF_FILE_PATTERNS

def validate_config() -> bool:
    """
    Validate configuration and print issues
    
    Returns:
        True if configuration is valid
    """
    issues = config.validate_configuration()
    
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✅ Configuration validation passed")
    return True

def print_config_summary():
    """Print configuration summary"""
    print("=" * 60)
    print("📋 Production OCR System Configuration")
    print("=" * 60)
    
    print(f"🏠 Base Directory: {config.BASE_DIR}")
    print(f"📁 Input Directory: {config.INPUT_DIR}")
    print(f"📁 Output Directory: {config.OUTPUT_DIR}")
    print(f"📄 Output File: {config.OUTPUT_CSV_NAME}")
    
    print(f"\n🔧 Processing Settings:")
    print(f"  - Mock Data: {'ON' if config.USE_MOCK_DATA else 'OFF'}")
    print(f"  - Debug Mode: {'ON' if config.DEBUG_MODE else 'OFF'}")
    print(f"  - Pages per Split: {config.PAGES_PER_SPLIT}")
    print(f"  - API Interval: {config.API_REQUEST_INTERVAL}s")
    print(f"  - Max File Size: {config.MAX_FILE_SIZE_MB}MB")
    
    print(f"\n🔐 API Configuration:")
    api_summary = config.get_api_key_summary()
    for api, status in api_summary.items():
        print(f"  - {api.title()}: {status}")
    
    print(f"\n📊 Logging:")
    print(f"  - Log Level: {config.LOG_LEVEL}")
    print(f"  - Log File: {config.LOG_FILE_PATH}")
    print(f"  - Mask API Keys: {'ON' if config.MASK_API_KEYS_IN_LOGS else 'OFF'}")
    
    print(f"\n🚀 Production Ready: {'YES' if config.is_production_ready() else 'NO'}")
    print("=" * 60)

if __name__ == "__main__":
    print_config_summary()
    validate_config()