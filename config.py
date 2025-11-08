#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生産版設定ファイル - 環境変数対応
Environment-based configuration for production use
"""

import os
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load YAML configuration
def load_yaml_config():
    """Load configuration from config.yaml if it exists"""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load config.yaml: {e}")
    return {}

_yaml_config = load_yaml_config()

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
    OUTPUT_CSV_NAME = os.getenv('OUTPUT_CSV_NAME', "extracted_journal_entries_mjs45.csv")
    OUTPUT_CSV_PATH = OUTPUT_DIR / OUTPUT_CSV_NAME
    LOG_FILE_NAME = os.getenv('LOG_FILE_NAME', "pdf_extraction.log")
    LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME
    
    # ==================================================
    # Account Code CSV Path
    # ==================================================
    ACCOUNT_CODE_CSV_PATH = os.getenv('ACCOUNT_CODE_CSV_PATH', BASE_DIR / "勘定科目コード一覧.csv")
    
    # ==================================================
    # API Configuration
    # ==================================================
    
    # Anthropic Claude API (Claude Sonnet 4.0) - メインAPI
    @classmethod
    def _get_secret(cls, key: str, default: str = None) -> str:
        """Get secret from Streamlit secrets, TOML file, or environment variables"""
        # First try Streamlit secrets if available
        if STREAMLIT_AVAILABLE:
            try:
                return st.secrets.get(key, os.getenv(key, default))
            except:
                pass
        
        # Try loading from .streamlit/secrets.toml directly
        try:
            import toml
            secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
            if secrets_path.exists():
                secrets_data = toml.load(secrets_path)
                if key in secrets_data:
                    return secrets_data[key]
        except Exception:
            pass
        
        # Fallback to environment variables
        return os.getenv(key, default)
    
    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return self._get_secret('ANTHROPIC_API_KEY')
    
    @property 
    def GOOGLE_API_KEY(self) -> str:
        """Get Google API key from Streamlit secrets or environment variables"""
        return self._get_secret('GOOGLE_API_KEY')
    
    ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-20250514')
    ANTHROPIC_MAX_TOKENS = int(os.getenv('ANTHROPIC_MAX_TOKENS', '64000'))  # Claude最適化: 大量データ処理対応
    ANTHROPIC_BETA_HEADERS = os.getenv('ANTHROPIC_BETA_HEADERS', 'context-1m-2025-08-07')  # 修正: 不正な日付形式を修正
    
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
    USE_MOCK_DATA = False  # モックモード完全撤廃 - 常に本番API使用
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    # PDF Processing
    # 環境変数より優先: config.yamlまたはコード内のデフォルト値を使用
    PAGES_PER_SPLIT = _yaml_config.get('pdf', {}).get('pages_per_split', 7)  # Claude最適化: バランス型（API呼び出し回数と画像サイズのトレードオフ）
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '50'))
    PDF_DPI = _yaml_config.get('pdf', {}).get('dpi', 150)  # Claude最適化: 150DPI（品質とコストのバランス）
    
    # Right column zoom for credit side improvement
    RIGHT_COL_ZOOM = _yaml_config.get('pdf', {}).get('right_col_zoom', False)
    RIGHT_COL_DPI = int(_yaml_config.get('pdf', {}).get('dpi', PDF_DPI))  # Use same DPI or config override

    # Voucher numbering settings
    USE_LLM_VOUCHER_NO = _yaml_config.get('processing', {}).get('use_llm_voucher_no', False)  # 貸借一致ブロック方式を優先
    VOUCHER_NO_WIDTH = int(_yaml_config.get('processing', {}).get('voucher_no_width', 4))  # 4桁0001-9999形式
    
    # API Rate Limiting
    API_REQUEST_INTERVAL = 0.5  # Claude最適化: 並列処理前提で短縮（レート制限対策）
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '1'))  # 1回リトライのみ
    REQUEST_TIMEOUT = 180  # Claude最適化: 180秒（7ページ処理に十分）
    
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
    MAX_CONCURRENT_REQUESTS = 1  # シーケンシャル処理（Streamlit環境でのメモリ制約考慮）
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
    # Post-Processing Settings
    # ==================================================
    class POSTPROCESS:
        SUM_TOLERANCE = int(os.getenv('POSTPROCESS_SUM_TOLERANCE', '0'))
        DROP_BOTH_CODE_EMPTY = os.getenv('POSTPROCESS_DROP_BOTH_CODE_EMPTY', 'true').lower() == 'true'
    
    # ==================================================
    # Error Handling
    # ==================================================
    ENABLE_RETRY = os.getenv('ENABLE_RETRY', 'true').lower() == 'true'  # 1回リトライ有効
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
    # API Pricing Configuration (USD per 1K tokens)
    # ==================================================
    API_PRICES = {
        # OpenAI GPT-5 (Latest)
        "gpt-5": {"input": 0.005, "output": 0.015},
        "gpt-5-mini": {"input": 0.001, "output": 0.003},
        # Anthropic Claude (Latest)
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},  # Legacy
        # Google Gemini (Latest)
        "gemini-2.5-flash": {"input": 0.0003, "output": 0.0006},
        "gemini-2.5-pro": {"input": 0.0015, "output": 0.005},
        "gemini-1.5-pro": {"input": 0.0015, "output": 0.005},  # Legacy
        "gemini-1.5-flash": {"input": 0.0003, "output": 0.0006},  # Legacy
    }
    
    # USD to JPY conversion rate (approximate)
    USD_TO_JPY_RATE = float(os.getenv('USD_TO_JPY_RATE', '150'))
    
    @classmethod
    def get_current_usd_to_jpy_rate(cls) -> float:
        """
        リアルタイムUSD-JPY為替レートを取得
        複数のAPIを試行してフォールバックを提供
        
        Returns:
            float: USD to JPY conversion rate
        """
        import requests
        logger = logging.getLogger(__name__)
        
        # API候補リスト（優先順位順）
        apis = [
            {
                "url": "https://api.fxratesapi.com/latest?base=USD&currencies=JPY",
                "path": ["rates", "JPY"]
            },
            {
                "url": "https://api.exchangerate-api.com/v4/latest/USD",
                "path": ["rates", "JPY"]
            },
            {
                "url": "https://api.exchangerate.host/latest?base=USD&symbols=JPY",
                "path": ["rates", "JPY"]
            }
        ]
        
        for i, api in enumerate(apis):
            try:
                response = requests.get(api["url"], timeout=3)
                response.raise_for_status()
                data = response.json()
                
                # パスに従ってデータを取得
                current_data = data
                for key in api["path"]:
                    if key in current_data:
                        current_data = current_data[key]
                    else:
                        raise ValueError(f"Key '{key}' not found in response")
                
                rate = float(current_data)
                if 100 <= rate <= 200:  # 妥当な範囲チェック
                    logger.info(f"Current USD-JPY rate: {rate} (from API #{i+1})")
                    return rate
                else:
                    raise ValueError(f"Rate {rate} is outside reasonable range")
                    
            except Exception as e:
                logger.debug(f"API #{i+1} failed: {e}")
                continue
        
        # すべてのAPIが失敗した場合
        logger.warning(f"All exchange rate APIs failed. Using fallback rate: {cls.USD_TO_JPY_RATE}")
        return cls.USD_TO_JPY_RATE

    # ==================================================
    # MJS 45-Column CSV Headers (完全一致)
    # ==================================================
    MJS_45_COLUMNS = [
        "伝票日付", "内部月", "伝票NO", "証憑NO", "データ種別", "仕訳入力形式",
        "（借）科目ｺｰﾄﾞ", "（借）補助ｺｰﾄﾞ", "（借）部門ｺｰﾄﾞ", "（借）セグメントｺｰﾄﾞ",
        "（借）消費税区分", "（借）業種", "（借）税込区分", "（借）補助区分1",
        "（借）補助ｺｰﾄﾞ1", "（借）補助区分2", "（借）補助ｺｰﾄﾞ2",
        "（貸）科目ｺｰﾄﾞ", "（貸）補助ｺｰﾄﾞ", "（貸）部門ｺｰﾄﾞ", "（貸）セグメントｺｰﾄﾞ",
        "（貸）消費税区分", "（貸）業種", "（貸）税込区分", "（貸）補助区分1",
        "（貸）補助ｺｰﾄﾞ1", "（貸）補助区分2", "（貸）補助ｺｰﾄﾞ2",
        "金額", "消費税額", "消費税ｺｰﾄﾞ", "消費税率", "外税同時入力区分",
        "資金繰入力区分", "資金繰ｺｰﾄﾞ", "摘要", "摘要コード1", "摘要コード2",
        "摘要コード3", "摘要コード4", "摘要コード5", "期日", "付箋", "付箋コメント",
        "事業者取引区分"
    ]
    
    # 5カラムJSONの必須フィールド
    REQUIRED_JSON_FIELDS = ["伝票日付", "借貸区分", "科目名", "金額", "摘要"]
    
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
MJS_45_COLUMNS = config.MJS_45_COLUMNS
ACCOUNT_CODE_CSV_PATH = config.ACCOUNT_CODE_CSV_PATH
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
    
    print(f"\n🚀 Production Ready: {'YES' if Config.is_production_ready() else 'NO'}")
    print("=" * 60)

if __name__ == "__main__":
    print_config_summary()
    validate_config()