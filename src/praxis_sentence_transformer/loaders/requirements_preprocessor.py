"""
Module for preprocessing requirements text
"""

import spacy
import re
from typing import Optional, Dict
from ..logger import setup_logging, handle_exception
from langdetect import detect, LangDetectException

class RequirementsPreprocessor:
    """Handles preprocessing of requirements text"""
    
    def __init__(self):
        """Initialize RequirementsPreprocessor with multiple language models"""
        self.logger = setup_logging("requirements-preprocessor")
        self.logger.debug("Initializing RequirementsPreprocessor")
        
        # Initialize language models
        self.nlp_models: Dict[str, spacy.language.Language] = {}
        self._initialize_models()
        
        # Compile regex patterns for cleaning
        self.rtf_pattern = re.compile(r'\\[a-z]+[0-9]* |{\\[^}]*}|\\\w+')
        self.escape_chars = {
            '\\\'e0': 'à', '\\\'e8': 'è', '\\\'e9': 'é',
            '\\\'ec': 'ì', '\\\'f2': 'ò', '\\\'f9': 'ù'
        }
        
        self.logger.debug("RequirementsPreprocessor initialization complete")
    
    def _initialize_models(self):
        """Initialize required spaCy models"""
        models = {
            'it': 'it_core_news_sm',  # Italian
            'en': 'en_core_web_sm',   # English
            'multi': 'xx_ent_wiki_sm' # Multilingual
        }
        
        for lang, model in models.items():
            try:
                self.logger.debug(f"Loading {lang} model: {model}")
                self.nlp_models[lang] = spacy.load(model)
            except OSError:
                self.logger.warning(f"Downloading {lang} model: {model}")
                spacy.cli.download(model)
                self.nlp_models[lang] = spacy.load(model)
    
    def _detect_language(self, text: str) -> str:
        """Detect text language"""
        try:
            # Clean text before detection
            clean_text = self._clean_rtf(text)
            clean_text = self._replace_escape_chars(clean_text)
            lang = detect(clean_text)
            return 'it' if lang == 'it' else 'en'
        except LangDetectException:
            self.logger.warning("Language detection failed, using multilingual model")
            return 'multi'
    
    @handle_exception
    def preprocess_text(self, text: str, preserve_case: bool = False) -> str:
        """
        Preprocess text using appropriate language model
        
        Args:
            text: Input text to preprocess
            preserve_case: If True, preserves original case (for CC documents)
            
        Returns:
            str: Preprocessed text
        """
        # Clean RTF formatting and escape sequences
        text = self._clean_rtf(text)
        text = self._replace_escape_chars(text)
        
        # Detect language and get appropriate model
        lang = self._detect_language(text)
        nlp = self.nlp_models.get(lang, self.nlp_models['multi'])
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Basic preprocessing:
        # 1. Remove punctuation and extra whitespace
        # 2. Optionally convert to lowercase
        # 3. Keep accented characters
        processed_text = ' '.join([
            token.text if preserve_case else token.text.lower()
            for token in doc 
            if not token.is_punct and not token.is_space
        ])
        
        return processed_text.strip()
    
    def _clean_rtf(self, text: str) -> str:
        """Remove RTF formatting"""
        text = self.rtf_pattern.sub(' ', text)
        return ' '.join(text.split())
    
    def _replace_escape_chars(self, text: str) -> str:
        """Replace escaped Italian characters with proper accented chars"""
        for escape_seq, char in self.escape_chars.items():
            text = text.replace(escape_seq, char)
        return text