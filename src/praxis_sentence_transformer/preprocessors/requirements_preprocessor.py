from typing import List, Dict

class RequirementsPreprocessor:
    def __init__(self, nlp):
        self.nlp = nlp

    def preprocess_text(self, text: str, preserve_case: bool = True) -> str:
        """
        Preprocess text while preserving case sensitivity
        
        Args:
            text (str): Input text to preprocess
            preserve_case (bool): Whether to preserve original case. Defaults to True.
        
        Returns:
            str: Preprocessed text with original case preserved
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Process with spaCy while preserving case
        doc = self.nlp(text)
        
        # Filter tokens but maintain original case
        tokens = []
        for token in doc:
            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                continue
                
            # Keep original token text instead of lowercase
            token_text = token.text if preserve_case else token.text.lower()
            tokens.append(token_text)
        
        # Rejoin tokens with spaces
        processed_text = ' '.join(tokens)
        
        return processed_text

    def preprocess_requirements(self, requirements: List[Dict]) -> List[Dict]:
        """
        Preprocess a list of requirements while preserving case
        
        Args:
            requirements (List[Dict]): List of requirement dictionaries
            
        Returns:
            List[Dict]: Preprocessed requirements with case preserved
        """
        processed = []
        for req in requirements:
            processed_req = req.copy()
            processed_req['content'] = self.preprocess_text(req['content'], preserve_case=True)
            processed.append(processed_req)
        return processed 