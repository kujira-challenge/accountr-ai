#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生産版PDF仕訳抽出システム - Production Ready
PDFファイルを5ページずつ分割してClaude APIで仕訳情報を抽出し、統合CSVを出力する
Enhanced with production features: retry logic, error handling, logging, configuration management
"""

# Initialize logging first
from logging_config import setup_logging
setup_logging()

import os
import json
import csv
import base64
import logging
import time
import gc
import hashlib
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass

# Core libraries
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
import fitz  # PyMuPDF
import pandas as pd

# API clients
try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from anthropic import Anthropic, APIError, RateLimitError
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# LLM Provider abstraction
from llm_providers.factory import build as build_llm

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Local configuration
from config import config, Config

# Set up logger using config
logger = Config.setup_logging()

@dataclass
class ProcessingResult:
    """Result of PDF processing operation"""
    success: bool
    data: List[Dict] = None
    error_message: str = None
    processing_time: float = 0.0
    file_path: str = None
    pages_processed: int = 0
    total_cost_usd: float = 0.0
    total_cost_jpy: float = 0.0

@dataclass 
class APIRetryConfig:
    """Configuration for API retry logic"""
    max_retries: int = 3
    base_delay: float = 1.0
    exponential_backoff: bool = True
    timeout: float = 60.0

def calculate_api_cost(model_name: str, usage) -> float:
    """
    Calculate API cost based on usage and model pricing (flexible for all providers)
    
    Args:
        model_name: Name of the model used
        usage: API usage object with token counts (provider-agnostic)
        
    Returns:
        Cost in USD
    """
    from config import config
    
    model = model_name.lower()
    
    # Extract token counts (handle different provider formats)
    input_tokens = 0
    output_tokens = 0
    
    if hasattr(usage, 'input_tokens') and hasattr(usage, 'output_tokens'):
        # Anthropic format
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
    elif hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
        # OpenAI format
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
    elif hasattr(usage, 'prompt_token_count') and hasattr(usage, 'candidates_token_count'):
        # Gemini format
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
    else:
        # Fallback: try common attribute names
        for attr in ['input_tokens', 'prompt_tokens', 'tokens_in']:
            if hasattr(usage, attr):
                input_tokens = getattr(usage, attr, 0)
                break
        for attr in ['output_tokens', 'completion_tokens', 'candidates_token_count', 'tokens_out']:
            if hasattr(usage, attr):
                output_tokens = getattr(usage, attr, 0)
                break
    
    # Flexible model pricing lookup
    pricing = None
    
    # Try exact match first
    if model in config.API_PRICES:
        pricing = config.API_PRICES[model]
    else:
        # Try partial matches for various model families
        for price_model, price_data in config.API_PRICES.items():
            price_model_lower = price_model.lower()
            
            # Anthropic models
            if ("claude" in model and "claude" in price_model_lower and
                any(x in model for x in ["sonnet-4", "3-5-sonnet", "haiku"])):
                if ("sonnet-4" in model and "sonnet-4" in price_model_lower) or \
                   ("3-5-sonnet" in model and "3-5-sonnet" in price_model_lower) or \
                   ("haiku" in model and "haiku" in price_model_lower):
                    pricing = price_data
                    break
            
            # Gemini models
            elif ("gemini" in model and "gemini" in price_model_lower):
                if ("2.5" in model and "2.5" in price_model_lower) or \
                   ("1.5" in model and "1.5" in price_model_lower):
                    if ("flash" in model and "flash" in price_model_lower) or \
                       ("pro" in model and "pro" in price_model_lower):
                        pricing = price_data
                        break
            
            # OpenAI models
            elif ("gpt" in model and "gpt" in price_model_lower):
                if ("gpt-5" in model and "gpt-5" in price_model_lower) or \
                   ("gpt-4" in model and "gpt-4" in price_model_lower):
                    pricing = price_data
                    break
    
    # Calculate cost if pricing found
    if pricing and input_tokens > 0 and output_tokens > 0:
        price_in = pricing.get("input", 0.003)  # Default fallback
        price_out = pricing.get("output", 0.015)  # Default fallback
        cost = (input_tokens / 1000 * price_in + output_tokens / 1000 * price_out)
        
        # Log cost calculation for debugging
        from logging import getLogger
        logger = getLogger(__name__)
        logger.debug(f"Cost calculation: model={model_name}, in_tokens={input_tokens}, "
                    f"out_tokens={output_tokens}, cost=${cost:.4f} USD")
        
        return cost
    
    # If no pricing found, log warning and return 0
    from logging import getLogger
    logger = getLogger(__name__)
    logger.warning(f"No pricing found for model: {model_name}, returning $0.00 cost")
    
    return 0.0

class ProductionPDFExtractor:
    """Production-ready PDF extractor with enterprise features"""
    
    def __init__(self, api_key: Optional[str] = None, use_mock: Optional[bool] = None):
        """
        Initialize production PDF extractor with multi-provider support
        
        Args:
            api_key: API key (overrides config if provided)
            use_mock: Deprecated parameter (ignored, always uses production API)
        """
        # Load configuration
        self.config = self._load_config()
        
        # LLM Provider configuration
        self.provider_name = self.config["llm"]["provider"]
        self.model_name = self.config["llm"]["model"]
        self.temperature = self.config["llm"].get("temperature", 0.0)
        self.pricing = self.config.get("pricing", {})
        
        # Legacy compatibility
        self.api_provider = self.provider_name
        self.api_key = api_key or (
            config.ANTHROPIC_API_KEY if self.provider_name == "anthropic" 
            else os.environ.get("GOOGLE_API_KEY")
        )
        self.max_tokens = 4096
            
        self.use_mock = False  # モックモード完全撤廃
        
        # Initialize LLM provider
        self.llm_provider = build_llm(self.provider_name, self.model_name, self.pricing)
        
        # Initialize legacy API client for backward compatibility
        self.client = None
        if self.provider_name == "anthropic":
            self._initialize_api_client()
        
        # Processing settings
        self.max_retries = config.MAX_RETRIES
        self.request_timeout = config.REQUEST_TIMEOUT
        self.api_interval = config.API_REQUEST_INTERVAL
        
        # Performance settings
        self.max_concurrent_requests = config.MAX_CONCURRENT_REQUESTS
        self.worker_pool_size = config.WORKER_POOL_SIZE
        
        # File processing settings - 5ページ単位で安定化
        self.pages_per_split = 5
        self.max_file_size = config.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        
        # Security settings
        self.mask_api_keys = config.MASK_API_KEYS_IN_LOGS
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'pages_processed': 0,
            'api_calls_made': 0,
            'errors_encountered': 0,
            'total_processing_time': 0.0,
            'total_cost_usd': 0.0,
            'total_cost_jpy': 0.0
        }
        
        logger.info(f"ProductionPDFExtractor initialized - Provider: {self.api_provider}, Model: {self.model_name}")
    
    def _load_config(self) -> dict:
        """Load configuration from config.yaml"""
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}, using defaults")
            return {
                "llm": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20240620", 
                    "temperature": 0.0
                },
                "pricing": {
                    "anthropic": {
                        "claude-3-5-sonnet-20240620": {"in": 0.000003, "out": 0.000015}
                    }
                }
            }
    
    def _initialize_api_client(self) -> None:
        """Initialize Anthropic API client with validation"""
        if not self.api_key or self.api_key == 'DUMMY_API_KEY':
            raise ValueError("Valid Anthropic API key required for production use")
        
        try:
            if not HAS_ANTHROPIC:
                raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
            
            # Claude Sonnet 4.0用のクライアント初期化
            client_kwargs = {"api_key": self.api_key}
            
            # ベータヘッダーの設定
            if hasattr(config, 'ANTHROPIC_BETA_HEADERS') and config.ANTHROPIC_BETA_HEADERS:
                client_kwargs["default_headers"] = {"anthropic-beta": config.ANTHROPIC_BETA_HEADERS}
            
            self.client = Anthropic(**client_kwargs)
            logger.info(f"Claude Sonnet 4.0 API client initialized successfully - Model: {self.model_name}")
            
            # Test API connectivity
            if config.DEBUG_MODE:
                self._test_api_connectivity()
                
        except Exception as e:
            logger.error(f"Failed to initialize Claude API client: {e}")
            raise
    
    def _test_api_connectivity(self) -> bool:
        """Test Claude API connectivity with a simple request"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            logger.info(f"Claude Sonnet 4.0 API connectivity test successful - Model: {self.model_name}")
            return True
        except Exception as e:
            logger.warning(f"Claude API connectivity test failed: {e}")
            return False
    
    def validate_file(self, pdf_path: Path) -> Tuple[bool, str]:
        """
        Validate PDF file before processing
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not pdf_path.exists():
                return False, f"File does not exist: {pdf_path}"
            
            if not pdf_path.is_file():
                return False, f"Path is not a file: {pdf_path}"
            
            # Check file extension
            if pdf_path.suffix.lower() != '.pdf':
                return False, f"File is not a PDF: {pdf_path}"
            
            # Check file size
            file_size = pdf_path.stat().st_size
            if file_size > self.max_file_size:
                return False, f"File too large: {file_size} bytes (max: {self.max_file_size})"
            
            # Try to open and read PDF
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                page_count = len(reader.pages)
                
                if page_count == 0:
                    return False, "PDF contains no pages"
                
                # Check if PDF is encrypted
                if reader.is_encrypted:
                    return False, "PDF is password protected"
            
            logger.info(f"File validation passed: {pdf_path.name} ({page_count} pages, {file_size} bytes)")
            return True, ""
            
        except Exception as e:
            return False, f"File validation error: {str(e)}"
    
    def backup_file(self, pdf_path: Path) -> Optional[Path]:
        """Create backup of original file if enabled"""
        if not config.BACKUP_ORIGINAL_FILES:
            return None
        
        try:
            backup_dir = config.OUTPUT_DIR / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{pdf_path.stem}_{timestamp}{pdf_path.suffix}"
            backup_path = backup_dir / backup_name
            
            shutil.copy2(pdf_path, backup_path)
            logger.info(f"File backed up: {backup_path}")
            
            return backup_path
        except Exception as e:
            logger.warning(f"Failed to backup file: {e}")
            return None
    
    def split_pdf(self, pdf_path: Path, output_dir: Path, pages_per_split: int = 5) -> List[Path]:
        """
        Split PDF with enhanced error handling and validation
        
        Args:
            pdf_path: Source PDF file path
            output_dir: Output directory for split files
            pages_per_split: Number of pages per split
            
        Returns:
            List of split file paths
        """
        logger.info(f"Starting PDF split: {pdf_path.name}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        split_files = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                total_pages = len(reader.pages)
                
                logger.info(f"Total pages: {total_pages}")
                
                if total_pages == 0:
                    raise ValueError("PDF contains no pages")
                
                for start_page in range(0, total_pages, pages_per_split):
                    end_page = min(start_page + pages_per_split, total_pages)
                    
                    try:
                        writer = PdfWriter()
                        
                        # Add pages to writer
                        for page_num in range(start_page, end_page):
                            writer.add_page(reader.pages[page_num])
                        
                        # Generate output filename
                        output_filename = f"{pdf_path.stem}_pages_{start_page+1}-{end_page}.pdf"
                        output_path = output_dir / output_filename
                        
                        # Write split file
                        with open(output_path, 'wb') as output_file:
                            writer.write(output_file)
                        
                        split_files.append(output_path)
                        logger.info(f"Split created: {output_filename} (pages {start_page+1}-{end_page})")
                        
                    except Exception as e:
                        logger.error(f"Failed to create split {start_page+1}-{end_page}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"PDF split failed: {pdf_path} - {e}")
            # Clean up any partial split files
            for split_file in split_files:
                try:
                    if split_file.exists():
                        split_file.unlink()
                except:
                    pass
            raise
            
        return split_files
    
    def pdf_to_images_base64(self, pdf_path: Path) -> List[str]:
        """
        Convert PDF to base64 images with enhanced error handling
        
        Args:
            pdf_path: PDF file path
            
        Returns:
            List of base64 encoded images
        """
        images = []
        doc = None
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    
                    # High resolution conversion
                    mat = fitz.Matrix(config.PDF_DPI / 72.0, config.PDF_DPI / 72.0)
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PNG bytes
                    img_data = pix.tobytes("png")
                    
                    # Base64 encode
                    img_b64 = base64.b64encode(img_data).decode('utf-8')
                    images.append(img_b64)
                    
                    logger.debug(f"Converted page {page_num + 1} to image ({len(img_data)} bytes)")
                    
                    # Memory cleanup
                    pix = None
                    
                except Exception as e:
                    logger.error(f"Failed to convert page {page_num + 1}: {e}")
                    continue
                    
            return images
            
        except Exception as e:
            logger.error(f"PDF to images conversion failed: {pdf_path} - {e}")
            raise
        finally:
            if doc:
                doc.close()
    
    def extract_with_retry(self, pdf_path: Path, page_start: int, page_end: int, 
                          retry_config: Optional[APIRetryConfig] = None) -> List[Dict]:
        """
        Extract data with single attempt (no retry to save API costs)
        
        Args:
            pdf_path: PDF file path
            page_start: Start page number
            page_end: End page number  
            retry_config: Retry configuration (ignored - kept for compatibility)
            
        Returns:
            Extracted data list
        """
        try:
            # Perform single extraction attempt
            result = self._extract_from_pdf_chunk_internal(pdf_path, page_start, page_end)
            
            if result is not None:
                self.stats['api_calls_made'] += 1
                return result
            else:
                logger.error(f"Extraction returned None for {pdf_path.name} pages {page_start}-{page_end}")
                self.stats['errors_encountered'] += 1
                return []
                    
        except Exception as e:
            # Single attempt failed - return empty result immediately
            logger.error(f"API extraction failed for {pdf_path.name} pages {page_start}-{page_end}: {e}")
            self.stats['errors_encountered'] += 1
            return []
    
    def _extract_from_pdf_chunk_internal(self, pdf_path: Path, page_start: int, page_end: int) -> List[Dict]:
        """Internal extraction method with timeout handling"""
        filename = pdf_path.name
        page_range = f"{page_start}-{page_end}"
        
        logger.info(f"Extracting: {filename} (pages {page_range}) [本番API使用]")
        
        # Create extraction task with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._perform_api_extraction, pdf_path, filename, page_range)
            
            try:
                result = future.result(timeout=self.request_timeout)
                return result
            except TimeoutError:
                logger.error(f"API request timed out after {self.request_timeout}s")
                raise
    
    def _perform_api_extraction(self, pdf_path: Path, filename: str, page_range: str) -> List[Dict]:
        """Perform the actual API extraction"""
        try:
            # Convert PDF to images
            images_b64 = self.pdf_to_images_base64(pdf_path)
            if not images_b64:
                logger.warning(f"No images extracted from {filename}")
                return []
            
            # Systemプロンプト使用（固定化）
            system_prompt = self._get_system_prompt()
            
            # ユーザープロンプトに列ヒント文を追加
            user_prompt = f"ファイル名: {filename}, ページ: {page_range}\n"
            for page_idx in range(1, len(images_b64) + 1):
                user_prompt += f"ページ{page_idx}：左=借方｜中央=摘要｜右=貸方（列を取り違えないでください）\n"
            
            # Claude Sonnet 4.0 API呼び出し（JSON抽出ガード付き）
            response_text, cost_usd = self._call_claude_api(system_prompt, user_prompt, images_b64)
            
            # JSON抽出ガードを使用してパース
            from utils.json_guard import parse_5cols_json, get_fallback_entry
            try:
                extracted_data = parse_5cols_json(response_text)
            except Exception as e:
                logger.error(f"JSON抽出ガードでエラー: {e}")
                logger.info(f"JSON抽出失敗 - 原因: {type(e).__name__}: {str(e)[:100]}")
                # 1回のみリトライ（短縮プロンプト）
                logger.info("短縮プロンプトで1回リトライ")
                try:
                    retry_system = "出力はJSON配列のみ。各要素は5カラム（伝票日付/借貸区分/科目名/金額/摘要）。余計な文字・コードフェンス禁止。"
                    retry_response, retry_cost = self._call_claude_api(retry_system, user_prompt, images_b64)
                    cost_usd += retry_cost
                    extracted_data = parse_5cols_json(retry_response)
                    logger.info(f"リトライ成功: {len(extracted_data)}エントリ")
                except Exception as retry_e:
                    logger.error(f"リトライも失敗: {retry_e}")
                    logger.info(f"JSON抽出失敗 - 原因: リトライ後もパース失敗 ({type(retry_e).__name__})")
                    extracted_data = get_fallback_entry("抽出不能")
            
            # 5カラムの前段整形（コード列を見ない段階での重複・形状整理）
            from utils.reconcile_entries import reconcile_entries
            # ========== DIAG: stage2_reconcile ==========
            stage2_before = len(extracted_data)
            extracted_data, reconcile_metrics = reconcile_entries(extracted_data, sum_tolerance=0, return_metrics=True)
            stage2_count = len(extracted_data)
            logger.info(f"DIAG stage2_reconcile: count={stage2_count} (before={stage2_before}, diff={stage2_count-stage2_before})")
            logger.info(f"DIAG stage2_reconcile metrics: splits={reconcile_metrics['split_count']}, swaps={reconcile_metrics['swap_count']}, drops={reconcile_metrics['drop_count']}")
            if stage2_count == 0:
                logger.error("DIAG stage2_reconcile=0: 前段整形で全削除")
                raise RuntimeError("DIAG stage2_reconcile=0: 前段整形処理でエントリが全て削除されました。")
            
            # 後処理: 貸借ペア保証と金額バリデーション
            from utils.postprocess import enforce_debit_credit_pairs, validate_amounts
            from config import config
            paired_entries = enforce_debit_credit_pairs(extracted_data)
            # ========== DIAG: stage3_pair_validate ==========
            stage3_before = len(paired_entries)
            extracted_data, error_entries = validate_amounts(paired_entries)
            stage3_count = len(extracted_data)
            logger.info(f"DIAG stage3_pair_validate: count={stage3_count} (before={stage3_before}, errors={len(error_entries)})")
            if stage3_count == 0:
                logger.error("DIAG stage3_pair_validate=0: 金額バリデーションで全削除")
                raise RuntimeError("DIAG stage3_pair_validate=0: 金額バリデーションでエントリが全て削除されました。")
            
            if error_entries:
                logger.warning(f"Post-processing found {len(error_entries)} error entries (zero amounts, etc.)")
                # エラーエントリの情報をログに記録（UI表示用）
                for err in error_entries[:3]:  # 最初の3件のみログ出力
                    logger.warning(f"Error entry: 日付={err.get('伝票日付', '')}, 金額={err.get('金額', 0)}, 摘要={err.get('摘要', '')[:50]}...")
                if len(error_entries) > 3:
                    logger.warning(f"... and {len(error_entries) - 3} more error entries")
                
                # エラー情報をstatsに保存（UI表示用）
                if not hasattr(self, 'error_entries'):
                    self.error_entries = []
                self.error_entries.extend(error_entries)
            
            # 統計更新
            self.stats['total_cost_usd'] += cost_usd
            current_usd_rate = config.get_current_usd_to_jpy_rate()
            cost_jpy = cost_usd * current_usd_rate
            self.stats['total_cost_jpy'] += cost_jpy
            
            # 成功ログ（詳細情報付き）
            logger.info(f"抽出成功: {len(extracted_data)}エントリ, 費用: ${cost_usd:.4f} USD (¥{cost_jpy:.2f} JPY)")
            logger.debug(f"抽出データサンプル: {self._sanitize_extracted_data_for_logging(extracted_data[:2])}")
            return extracted_data
            
        except Exception as e:
            logger.error(f"API extraction failed: {filename} - {e}")
            raise
    
    def _sanitize_for_logging(self, text: str) -> str:
        """
        ログ出力用にPII（個人識別情報）をマスキング
        
        Args:
            text: 元のテキスト
            
        Returns:
            str: マスキング後のテキスト
        """
        from utils.masking import mask_personal_info
        return mask_personal_info(text)
        
    def _sanitize_extracted_data_for_logging(self, data: List[Dict]) -> List[Dict]:
        """
        抽出データのログ出力用サニタイズ
        
        Args:
            data: 抽出データ
            
        Returns:
            List[Dict]: マスキング後のデータ
        """
        from utils.masking import mask_list_for_logging
        return mask_list_for_logging(data, sample_size=2)
    
    def _call_claude_api(self, system_prompt: str, user_prompt: str, images_b64: List[str]) -> Tuple[str, float]:
        """Call LLM API with system prompt and images using provider abstraction"""
        try:
            logger.info(f"Making {self.provider_name} API call with model: {self.model_name}")
            
            # Convert PNG base64 images to JPEG base64 for provider consistency
            images_jpeg_b64 = []
            for img_b64 in images_b64:
                # Convert PNG to JPEG for consistency across providers
                from PIL import Image
                import io
                
                # Decode base64 PNG
                img_data = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_data))
                
                # Convert to JPEG
                jpeg_buffer = io.BytesIO()
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")
                img.save(jpeg_buffer, format="JPEG", quality=90)
                jpeg_data = jpeg_buffer.getvalue()
                
                # Re-encode to base64
                jpeg_b64 = base64.b64encode(jpeg_data)
                images_jpeg_b64.append(jpeg_b64)
            
            # Use the LLM provider abstraction
            result = self.llm_provider.generate(
                system=system_prompt,
                user=user_prompt,
                images=images_jpeg_b64,
                model=self.model_name,
                temperature=self.temperature
            )
            
            # Log usage information
            logger.info(f"使用トークン数: input={result.tokens_in}, "
                      f"output={result.tokens_out}, "
                      f"total={result.tokens_in + result.tokens_out}")
            
            # Log cost information
            current_usd_rate = config.get_current_usd_to_jpy_rate()
            cost_jpy = result.cost_usd * current_usd_rate
            logger.info(f"推定費用: ${result.cost_usd:.4f} USD (¥{cost_jpy:.2f} JPY) [レート: {current_usd_rate:.2f}]")
            logger.info(f"Provider={self.provider_name}, Model={self.model_name}, Tokens_in={result.tokens_in}, Tokens_out={result.tokens_out}, Cost_USD={result.cost_usd:.4f}")
            
            response_text = result.text.strip()
            
            # レスポンス内容をログに記録（デバッグ用・PII配慮）
            sanitized_preview = self._sanitize_for_logging(response_text[:8000])  # 最大8KB
            logger.debug(f"LLM API Response length: {len(response_text)} characters")
            logger.debug(f"LLM API Response preview: {sanitized_preview[:500]}...")
            
            # 推定結果件数をログに記録
            estimated_entries = response_text.count('"{') if response_text else 0
            logger.info(f"{self.provider_name} API推定結果: {estimated_entries}件のエントリ候補")
            
            # デバッグ用: レスポンスをファイルに保存
            if config.DEBUG_MODE:
                debug_dir = config.LOG_DIR / "debug_responses"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_file = debug_dir / f"{self.provider_name}_response_{int(time.time())}.txt"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(f"Provider: {self.provider_name}\n")
                    f.write(f"Model: {self.model_name}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Cost: ${result.cost_usd:.4f} USD\n")
                    f.write(f"Tokens: {result.tokens_in} in, {result.tokens_out} out\n")
                    f.write("="*50 + "\n")
                    f.write(response_text)
                logger.info(f"{self.provider_name} API response saved to: {debug_file}")
            
            return response_text, result.cost_usd
            
        except Exception as e:
            logger.error(f"{self.provider_name} API call failed: {e}")
            raise
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean API response to extract valid JSON"""
        # Remove code block markers if present
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.rfind("```")
            if end != -1 and end > start:
                response_text = response_text[start:end].strip()
        
        # Remove any leading/trailing text that's not JSON
        response_text = response_text.strip()
        start_bracket = response_text.find('[')
        end_bracket = response_text.rfind(']')
        
        if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
            response_text = response_text[start_bracket:end_bracket + 1]
        
        return response_text
    
    def _attempt_json_recovery(self, response_text: str, filename: str, page_range: str) -> List[Dict]:
        """Attempt to recover data from malformed JSON response"""
        logger.info("Attempting JSON recovery...")
        
        try:
            # Try to find individual JSON objects
            import re
            
            # Look for objects enclosed in curly braces
            object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(object_pattern, response_text, re.DOTALL)
            
            recovered_objects = []
            for match in matches:
                try:
                    obj = json.loads(match)
                    if isinstance(obj, dict) and any(key in obj for key in config.REQUIRED_FIELDS):
                        recovered_objects.append(obj)
                except:
                    continue
            
            if recovered_objects:
                logger.info(f"Recovered {len(recovered_objects)} objects from malformed JSON")
                return self._validate_extracted_data(recovered_objects, filename, page_range)
            
        except Exception as e:
            logger.error(f"JSON recovery failed: {e}")
        
        # Return empty list if recovery fails
        return []
    
    def _validate_extracted_data(self, data: List[Dict], filename: str, page_range: str) -> List[Dict]:
        """Validate and clean extracted data"""
        if not data:
            return []
        
        validated_data = []
        
        for i, entry in enumerate(data):
            try:
                # Ensure entry is a dictionary
                if not isinstance(entry, dict):
                    logger.warning(f"Entry {i} is not a dictionary, skipping")
                    continue
                
                # Validate required fields exist (5カラムJSON)
                missing_fields = [field for field in config.REQUIRED_JSON_FIELDS if field not in entry]
                for field in missing_fields:
                    entry[field] = "" if field != "金額" else 0
                
                # Validate amount field
                if config.VALIDATE_AMOUNTS and entry.get("金額"):
                    try:
                        amount = float(str(entry["金額"]).replace(",", ""))
                        if not (config.MIN_AMOUNT <= amount <= config.MAX_AMOUNT):
                            logger.warning(f"Amount {amount} out of valid range, entry {i}")
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid amount format in entry {i}: {entry.get('金額')}")
                
                # 5カラムJSONには参照元ファイル/ページは含まない（MJS変換で付与）
                
                validated_data.append(entry)
                
            except Exception as e:
                logger.error(f"Error validating entry {i}: {e}")
                continue
        
        logger.info(f"Validated {len(validated_data)} entries out of {len(data)}")
        return validated_data
    
    def get_enhanced_mock_data(self, filename: str, page_range: str) -> List[Dict]:
        """Enhanced mock data for testing"""
        mock_entries = [
            {
                "契約日": "R07/02/04",
                "借方科目": "水道光熱費",
                "貸方科目": "三菱UFJ普通預金",
                "摘要": "クリスクレメント共用 水道料",
                "金額": 3014,
                "備考": "契約者名: ; 号室: ; 物件名: ; オーナー: ",
                "参照元ファイル": filename,
                "ページ": page_range
            },
            {
                "契約日": "R07/02/05",
                "借方科目": "預り金",
                "貸方科目": "三菱普通",
                "摘要": "ジェイリース (株) 保証料",
                "金額": 12000,
                "備考": "契約者名: 山下良三; 号室: No.13; 物件名: フィールドパーキング; オーナー: 林篤史",
                "参照元ファイル": filename,
                "ページ": page_range
            }
        ]
        
        # Add randomization for testing
        if config.DEBUG_MODE:
            import random
            base_amount = random.randint(1000, 50000)
            mock_entries.append({
                "契約日": f"R07/02/{random.randint(1, 28):02d}",
                "借方科目": random.choice(["現金", "普通預金", "売掛金"]),
                "貸方科目": random.choice(["売上高", "受取手数料", "預り金"]),
                "摘要": f"テスト取引 #{random.randint(1000, 9999)}",
                "金額": base_amount,
                "備考": f"テストデータ - {filename}",
                "参照元ファイル": filename,
                "ページ": page_range
            })
        
        return mock_entries
    
    def _get_system_prompt(self) -> str:
        """強化システムプロンプト（オブジェクト配列強制・貸借ペア保証・摘要統一・右列強化）"""
        return """あなたは「帳票OCR変換器（監査人視点）」です。A4の不動産伝票PDF（退去・振替・更新・新規など）から【オブジェクトの配列】をJSON形式で出力してください。

★★重要★★ 貸方科目名の不明返却は禁止。読めない場合は候補集合から最尤を必ず選択してください。右列=貸方です。左右を取り違えないでください。

【出力形式（厳守）】
- 出力は JSON で、**オブジェクト（辞書）の配列**のみ。
- 各オブジェクトは **必須5キー**：["伝票日付","借貸区分","科目名","金額","摘要"]
- ★重要★ 5要素の配列 ["2024/1/1", "借方", "現金", 1000, "摘要"] やタプルで返してはならない。必ず { "伝票日付": "...", "借貸区分": "...", "科目名": "...", "金額": ..., "摘要": "..." } の辞書オブジェクト形式。
- 余計な説明文・コードフェンス・markdown記法は一切禁止。

【出力カラム（5キー必須）】
1) "伝票日付"（YYYY/M/D、西暦文字列）
2) "借貸区分"（"借方" または "貸方"）
3) "科目名"（不明なら空文字でよい）
4) "金額"（半角数字、正整数）
5) "摘要"（共通摘要＋行固有＋「; 借方:XXX / 貸方:YYY」）

【レイアウトの絶対則】
- 伝票は「左=借方／右=貸方」。中央に共通摘要。
- 毎「枠」（行帯）について、必ず借方レッグと貸方レッグのペアを出力する。
  - 片側が読めない場合：借貸区分は正しく、金額は枠の金額、不明側の科目名は空文字でよい。
  - いずれの場合も摘要末尾に「; 借方:XXX / 貸方:YYY」を必ず付加。不明側は「不明【OCR注意】」と明記。
- 枠ごとの借方金額合計と貸方金額合計は一致させる。合わないJSONを出力してはならない。
  不足分は"不明"レッグを追加して合わせる（科目名は空文字で可）。

【共通摘要ルール】
- 物件名・号室・契約者名・オーナー名・○月分賃料など、中央の枠単位の情報は、同枠の全レッグに継承する。
- 枠内で不一致がある場合は情報量の多いものを採用。不確実なら摘要末尾に【OCR注意:目視確認推奨】。

【フォールバック】
- 抽出不能でも配列1要素は必ず返す（科目名空、摘要末尾に【OCR注意】）。
- 金額に負数や記号は使わない。
- 出力例: [{"伝票日付":"","借貸区分":"借方","科目名":"","金額":0,"摘要":"抽出不能【OCR注意】"}]"""
    
    def process_single_pdf(self, pdf_path: Path, temp_dir: Path, pages_per_split: Optional[int] = None) -> ProcessingResult:
        """
        Process single PDF with comprehensive error handling
        
        Args:
            pdf_path: PDF file path
            temp_dir: Temporary directory
            pages_per_split: Pages per split (uses config if None)
            
        Returns:
            ProcessingResult with detailed information
        """
        start_time = time.time()
        pages_per_split = pages_per_split or self.pages_per_split
        
        logger.info(f"Starting PDF processing: {pdf_path.name}")
        
        # Validate file
        is_valid, error_msg = self.validate_file(pdf_path)
        if not is_valid:
            return ProcessingResult(
                success=False,
                error_message=error_msg,
                file_path=str(pdf_path),
                processing_time=time.time() - start_time
            )
        
        # Backup file if enabled
        backup_path = self.backup_file(pdf_path)
        
        all_extracted_data = []
        split_files = []
        
        try:
            # Create PDF-specific temp directory
            pdf_temp_dir = temp_dir / pdf_path.stem
            pdf_temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Split PDF
            split_files = self.split_pdf(pdf_path, pdf_temp_dir, pages_per_split)
            
            if not split_files:
                raise ValueError("No split files were created")
            
            # Process each split with threading if enabled
            if self.max_concurrent_requests > 1:
                all_extracted_data = self._process_splits_concurrent(split_files, pages_per_split)
            else:
                all_extracted_data = self._process_splits_sequential(split_files, pages_per_split)
            
            # Update statistics
            self.stats['files_processed'] += 1
            self.stats['pages_processed'] += self.get_pdf_page_count(pdf_path)
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            
            logger.info(f"PDF processing completed: {pdf_path.name} - {len(all_extracted_data)} entries in {processing_time:.2f}s")
            
            # 現在のファイル処理の費用計算（最新レート使用）
            current_usd_rate = config.get_current_usd_to_jpy_rate()
            # 統計の費用を最新レートで再計算
            recalculated_cost_jpy = self.stats['total_cost_usd'] * current_usd_rate
            
            return ProcessingResult(
                success=True,
                data=all_extracted_data,
                file_path=str(pdf_path),
                processing_time=processing_time,
                pages_processed=len(split_files) * pages_per_split,
                total_cost_usd=self.stats['total_cost_usd'],
                total_cost_jpy=recalculated_cost_jpy
            )
            
        except Exception as e:
            error_msg = f"PDF processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.stats['errors_encountered'] += 1
            
            return ProcessingResult(
                success=False,
                error_message=error_msg,
                file_path=str(pdf_path),
                processing_time=time.time() - start_time
            )
            
        finally:
            # Cleanup split files
            if config.CLEANUP_TEMP_FILES:
                for split_file in split_files:
                    try:
                        if split_file.exists():
                            split_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {split_file}: {e}")
            
            # Memory cleanup
            if self.stats['files_processed'] % config.GARBAGE_COLLECTION_INTERVAL == 0:
                gc.collect()
    
    def _process_splits_sequential(self, split_files: List[Path], pages_per_split: int) -> List[Dict]:
        """Process split files sequentially"""
        all_data = []
        
        for i, split_file in enumerate(split_files):
            # Extract page range from filename (e.g., "file_pages_1-5.pdf" -> start=1, end=5)
            filename = split_file.stem
            if '_pages_' in filename:
                try:
                    page_range_str = filename.split('_pages_')[1]
                    start_str, end_str = page_range_str.split('-')
                    page_start = int(start_str)
                    page_end = int(end_str)
                except (ValueError, IndexError):
                    # Fallback to calculation method
                    page_start = i * pages_per_split + 1
                    page_end = (i + 1) * pages_per_split
            else:
                page_start = i * pages_per_split + 1
                page_end = (i + 1) * pages_per_split
            
            # Rate limiting (本番API使用時)
            if i > 0:
                time.sleep(self.api_interval)
            
            extracted_data = self.extract_with_retry(split_file, page_start, page_end)
            all_data.extend(extracted_data)
            
            logger.info(f"Split processed: {split_file.name} - {len(extracted_data)} entries")
        
        return all_data
    
    def _process_splits_concurrent(self, split_files: List[Path], pages_per_split: int) -> List[Dict]:
        """Process split files concurrently with rate limiting"""
        all_data = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            # Submit all tasks
            futures = []
            for i, split_file in enumerate(split_files):
                # Extract page range from filename (e.g., "file_pages_1-5.pdf" -> start=1, end=5)
                filename = split_file.stem
                if '_pages_' in filename:
                    try:
                        page_range_str = filename.split('_pages_')[1]
                        start_str, end_str = page_range_str.split('-')
                        page_start = int(start_str)
                        page_end = int(end_str)
                    except (ValueError, IndexError):
                        # Fallback to calculation method
                        page_start = i * pages_per_split + 1
                        page_end = (i + 1) * pages_per_split
                else:
                    page_start = i * pages_per_split + 1
                    page_end = (i + 1) * pages_per_split
                
                future = executor.submit(self.extract_with_retry, split_file, page_start, page_end)
                futures.append((future, split_file.name))
                
                # Rate limiting for concurrent requests (本番API使用時)
                time.sleep(self.api_interval / self.max_concurrent_requests)
            
            # Collect results
            for future, filename in futures:
                try:
                    extracted_data = future.result(timeout=self.request_timeout * 2)
                    all_data.extend(extracted_data)
                    logger.info(f"Split processed: {filename} - {len(extracted_data)} entries")
                except Exception as e:
                    logger.error(f"Split processing failed: {filename} - {e}")
                    continue
        
        return all_data
    
    def get_pdf_page_count(self, pdf_path: Path) -> int:
        """Get PDF page count with error handling"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                return len(reader.pages)
        except Exception as e:
            logger.error(f"Failed to get page count for {pdf_path}: {e}")
            return 0
    
    def process_directory(self, input_dir: Path, output_csv: Path, temp_dir: Path) -> bool:
        """
        Process directory with enhanced error handling and reporting
        
        Args:
            input_dir: Input directory
            output_csv: Output CSV file
            temp_dir: Temporary directory
            
        Returns:
            Success status
        """
        start_time = time.time()
        logger.info(f"Starting directory processing: {input_dir}")
        
        # Validate input directory
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return False
        
        # Create output directories
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Get PDF files
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in: {input_dir}")
            return True  # Not an error, just empty directory
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process files
        all_data = []
        successful_files = 0
        failed_files = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            
            try:
                result = self.process_single_pdf(pdf_file, temp_dir)
                
                if result.success:
                    all_data.extend(result.data)
                    successful_files += 1
                    logger.info(f"Completed: {pdf_file.name} - {len(result.data)} entries")
                else:
                    failed_files.append((pdf_file.name, result.error_message))
                    logger.error(f"Failed: {pdf_file.name} - {result.error_message}")
                    
            except Exception as e:
                error_msg = f"Unexpected error processing {pdf_file.name}: {e}"
                logger.error(error_msg, exc_info=True)
                failed_files.append((pdf_file.name, error_msg))
        
        # Save results
        try:
            self.save_to_csv(all_data, output_csv)
            
            # Generate processing report
            self._generate_processing_report(
                start_time, len(pdf_files), successful_files, 
                failed_files, len(all_data), output_csv
            )
            
            logger.info(f"Directory processing completed: {successful_files}/{len(pdf_files)} files successful")
            return len(failed_files) == 0
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
        
        finally:
            # Cleanup temp directory
            if config.CLEANUP_TEMP_FILES:
                self.cleanup_temp_dir(temp_dir)
    
    def _generate_processing_report(self, start_time: float, total_files: int, successful_files: int,
                                  failed_files: List[Tuple[str, str]], total_entries: int, output_csv: Path):
        """Generate detailed processing report"""
        processing_time = time.time() - start_time
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': len(failed_files),
            'total_entries_extracted': total_entries,
            'output_file': str(output_csv),
            'statistics': self.stats.copy(),
            'failed_file_details': [{'filename': name, 'error': error} for name, error in failed_files]
        }
        
        # Save report
        report_file = config.LOG_DIR / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processing report saved: {report_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"🕐 Processing Time: {processing_time:.2f} seconds")
        print(f"📁 Total Files: {total_files}")
        print(f"✅ Successful: {successful_files}")
        print(f"❌ Failed: {len(failed_files)}")
        print(f"📄 Total Entries: {total_entries}")
        print(f"📊 API Calls Made: {self.stats['api_calls_made']}")
        if failed_files:
            print(f"\n❌ Failed Files:")
            for filename, error in failed_files:
                print(f"   - {filename}: {error[:100]}...")
        print(f"{'='*60}")
    
    def save_to_csv(self, data: List[Dict], output_path: Path) -> None:
        """Save data to CSV with enhanced error handling"""
        if not data:
            logger.warning("No data to save")
            # Create empty CSV with 5-column headers (for internal use only)
            df = pd.DataFrame(columns=config.REQUIRED_JSON_FIELDS)
        else:
            df = pd.DataFrame(data)
            
            # Ensure all required 5-column fields exist
            for col in config.REQUIRED_JSON_FIELDS:
                if col not in df.columns:
                    df[col] = "" if col != "金額" else 0
        
        try:
            # Save with BOM for Excel compatibility
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"CSV saved: {output_path} ({len(data)} entries)")
            
        except Exception as e:
            logger.error(f"CSV save failed: {e}")
            raise
    
    def cleanup_temp_dir(self, temp_dir: Path) -> None:
        """Cleanup temporary directory with enhanced error handling"""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"Temp directory cleaned: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        stats['average_processing_time'] = (
            stats['total_processing_time'] / stats['files_processed'] 
            if stats['files_processed'] > 0 else 0
        )
        return stats

# Legacy compatibility wrapper
class PDFExtractor(ProductionPDFExtractor):
    """Legacy compatibility wrapper"""
    
    def __init__(self, api_key: str = "DUMMY_API_KEY", use_mock: bool = True):
        super().__init__(api_key=api_key, use_mock=use_mock)
    
    def extract_from_pdf_chunk(self, pdf_path: Path, page_start: int, page_end: int) -> List[Dict]:
        """Legacy compatibility method"""
        return self.extract_with_retry(pdf_path, page_start, page_end)

def main():
    """Main function for direct execution"""
    from config import print_config_summary, validate_config
    
    print_config_summary()
    
    if not validate_config():
        logger.error("Configuration validation failed")
        return False
    
    # Use configuration values
    extractor = ProductionPDFExtractor(
        api_key=config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY,
        use_mock=config.USE_MOCK_DATA,
        api_provider='openai' if config.OPENAI_API_KEY else 'anthropic'
    )
    
    # Process directory
    success = extractor.process_directory(
        config.INPUT_DIR,
        config.OUTPUT_CSV_PATH,
        config.TEMP_DIR
    )
    
    # Print statistics
    stats = extractor.get_statistics()
    print(f"\n📊 Final Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)