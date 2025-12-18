"""
api_clients.py - API Clients for External Data Sources

This module provides clients for:
- DeepSeek API (V3.2 for generation, R1 for validation)
- Wikidata SPARQL queries (for factual data)
- ConceptNet REST API (for common sense relations)

All clients include retry logic with exponential backoff.
"""

import os
import time
import json
import logging
import requests
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# DEEPSEEK API CLIENT
# ============================================================================

@dataclass
class DeepSeekClient:
    """
    Client for DeepSeek API (V3.2 and R1 models).
    
    Handles:
    - Chat completions with retry logic
    - JSON parsing with fallback
    - Rate limiting
    """
    
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com/v1"
    retry_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    request_log: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Load API key from .env file or environment."""
        if self.api_key is None:
            # Try loading from .env file first
            try:
                from dotenv import load_dotenv
                from pathlib import Path
                
                # Look for .env in project root
                env_path = Path(__file__).parent.parent / '.env'
                if env_path.exists():
                    load_dotenv(env_path)
            except ImportError:
                pass  # dotenv not installed, use environment directly
            
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            logger.warning("DEEPSEEK_API_KEY not set. API calls will fail.")
    
    def _make_request(
        self,
        endpoint: str,
        payload: Dict,
        timeout: int = 60,
    ) -> Dict:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        last_error = None
        delay = self.retry_base_delay
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                
                # Log request
                self.request_log.append({
                    "timestamp": time.time(),
                    "endpoint": endpoint,
                    "status": response.status_code,
                    "attempt": attempt + 1,
                })
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    logger.warning(f"Rate limited, waiting {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, self.retry_max_delay)
                elif response.status_code >= 500:  # Server error
                    logger.warning(f"Server error {response.status_code}, retrying...")
                    time.sleep(delay)
                    delay = min(delay * 2, self.retry_max_delay)
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout, attempt {attempt + 1}")
                last_error = "timeout"
                time.sleep(delay)
                delay = min(delay * 2, self.retry_max_delay)
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                last_error = str(e)
                time.sleep(delay)
                delay = min(delay * 2, self.retry_max_delay)
        
        raise RuntimeError(f"DeepSeek API failed after {self.retry_attempts} attempts: {last_error}")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "deepseek-chat",  # V3.2
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate text using DeepSeek chat API.
        
        Args:
            system_prompt: System instruction
            user_prompt: User message
            model: Model to use (deepseek-chat for V3.2, deepseek-reasoner for R1)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Generated text content
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = self._make_request("chat/completions", payload)
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format: {response}")
            raise RuntimeError(f"Failed to parse DeepSeek response: {e}")
    
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        retry_on_invalid: bool = True,
    ) -> Dict:
        """
        Generate JSON response with validation.
        
        If response is not valid JSON, retries with explicit JSON instruction.
        
        Args:
            system_prompt: System instruction (should mention JSON format)
            user_prompt: User message
            model: Model to use
            temperature: Lower for more deterministic output
            max_tokens: Max tokens
            retry_on_invalid: Retry with explicit instruction if invalid
            
        Returns:
            Parsed JSON dict
        """
        content = self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Try to parse JSON
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except json.JSONDecodeError:
            if retry_on_invalid:
                logger.warning("Invalid JSON, retrying with explicit instruction...")
                retry_prompt = user_prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no other text."
                content = self.generate(
                    system_prompt=system_prompt,
                    user_prompt=retry_prompt,
                    model=model,
                    temperature=0.1,  # Lower temp for retry
                    max_tokens=max_tokens,
                )
                try:
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    return json.loads(content.strip())
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse failed on retry: {content[:200]}")
                    raise RuntimeError(f"Failed to get valid JSON from DeepSeek: {e}")
            else:
                raise


# ============================================================================
# WIKIDATA SPARQL CLIENT
# ============================================================================

@dataclass  
class WikidataClient:
    """
    Client for Wikidata SPARQL queries.
    
    Provides pre-built queries for:
    - Country capitals
    - Country currencies
    """
    
    endpoint: str = "https://query.wikidata.org/sparql"
    timeout: int = 30
    
    # SPARQL queries per specification
    QUERY_CAPITALS = """
    SELECT ?country ?countryLabel ?capital ?capitalLabel WHERE {
      ?country wdt:P31 wd:Q6256.  # instance of country
      ?country wdt:P36 ?capital.   # has capital
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 500
    """
    
    QUERY_CURRENCIES = """
    SELECT ?country ?countryLabel ?currency ?currencyLabel WHERE {
      ?country wdt:P31 wd:Q6256.    # instance of country
      ?country wdt:P38 ?currency.   # has currency
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 500
    """
    
    def _execute_query(self, query: str) -> List[Dict]:
        """Execute SPARQL query and return results."""
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "SemanticGravityExperiment/1.0",
        }
        
        try:
            response = requests.get(
                self.endpoint,
                params={"query": query},
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", {}).get("bindings", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Wikidata query failed: {e}")
            raise
    
    def get_capitals(self) -> List[Tuple[str, str]]:
        """
        Get list of (country, capital) pairs.
        
        Filters to capitals with single-word alphabetic labels.
        
        Returns:
            List of (country_name, capital_name) tuples
        """
        results = self._execute_query(self.QUERY_CAPITALS)
        
        pairs = []
        for row in results:
            country = row.get("countryLabel", {}).get("value", "")
            capital = row.get("capitalLabel", {}).get("value", "")
            
            # Filter: capital must be single word, alphabetic only
            if capital and country:
                capital_clean = capital.strip()
                if capital_clean.isalpha() and " " not in capital_clean:
                    pairs.append((country, capital_clean))
        
        logger.info(f"Retrieved {len(pairs)} valid capital pairs from Wikidata")
        return pairs
    
    def get_currencies(self) -> List[Tuple[str, str]]:
        """
        Get list of (country, currency) pairs.
        
        Filters to currencies with single-word alphabetic labels.
        
        Returns:
            List of (country_name, currency_name) tuples
        """
        results = self._execute_query(self.QUERY_CURRENCIES)
        
        pairs = []
        for row in results:
            country = row.get("countryLabel", {}).get("value", "")
            currency = row.get("currencyLabel", {}).get("value", "")
            
            # Filter: currency must be single word, alphabetic only
            if currency and country:
                currency_clean = currency.strip()
                if currency_clean.isalpha() and " " not in currency_clean:
                    pairs.append((country, currency_clean))
        
        logger.info(f"Retrieved {len(pairs)} valid currency pairs from Wikidata")
        return pairs


# ============================================================================
# CONCEPTNET API CLIENT
# ============================================================================

@dataclass
class ConceptNetClient:
    """
    Client for ConceptNet REST API.
    
    Fetches relations:
    - UsedFor: what something is used for
    - MadeOf: material composition
    - HasProperty: properties/attributes
    """
    
    base_url: str = "https://api.conceptnet.io"
    timeout: int = 15
    rate_limit_delay: float = 0.1  # Seconds between requests
    _last_request_time: float = field(default=0.0, repr=False)
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _get_edges(self, concept: str, relation: str, limit: int = 20) -> List[Dict]:
        """
        Get edges for a concept and relation.
        
        Args:
            concept: The concept to query (e.g., "scissors")
            relation: Relation type (UsedFor, MadeOf, HasProperty)
            limit: Max edges to return
            
        Returns:
            List of edge dictionaries
        """
        self._rate_limit()
        
        # Build URL
        url = f"{self.base_url}/query"
        params = {
            "start": f"/c/en/{concept.lower().replace(' ', '_')}",
            "rel": f"/r/{relation}",
            "limit": limit,
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("edges", [])
        except requests.exceptions.RequestException as e:
            logger.warning(f"ConceptNet query failed for {concept}/{relation}: {e}")
            return []
    
    def _parse_concept(self, uri: str) -> Optional[str]:
        """
        Parse concept from ConceptNet URI.
        
        Example: /c/en/cut -> cut
                 /c/en/cut_paper -> cut paper
        """
        if not uri or not uri.startswith("/c/en/"):
            return None
        
        concept = uri.replace("/c/en/", "").replace("_", " ")
        
        # Filter to single-word targets only
        if " " in concept:
            return None
        
        # Must be alphabetic
        if not concept.isalpha():
            return None
        
        return concept
    
    def get_used_for(self, concept: str) -> List[Tuple[str, str]]:
        """
        Get what a concept is used for.
        
        Args:
            concept: Subject concept (e.g., "scissors")
            
        Returns:
            List of (concept, target) tuples where target is what it's used for
        """
        edges = self._get_edges(concept, "UsedFor")
        
        results = []
        for edge in edges:
            end_uri = edge.get("end", {}).get("@id", "")
            target = self._parse_concept(end_uri)
            if target:
                results.append((concept, target))
        
        return results
    
    def get_made_of(self, concept: str) -> List[Tuple[str, str]]:
        """
        Get what a concept is made of.
        
        Args:
            concept: Subject concept (e.g., "table")
            
        Returns:
            List of (concept, material) tuples
        """
        edges = self._get_edges(concept, "MadeOf")
        
        results = []
        for edge in edges:
            end_uri = edge.get("end", {}).get("@id", "")
            target = self._parse_concept(end_uri)
            if target:
                results.append((concept, target))
        
        return results
    
    def get_has_property(self, concept: str) -> List[Tuple[str, str]]:
        """
        Get properties of a concept.
        
        Args:
            concept: Subject concept (e.g., "snow")
            
        Returns:
            List of (concept, property) tuples
        """
        edges = self._get_edges(concept, "HasProperty")
        
        results = []
        for edge in edges:
            end_uri = edge.get("end", {}).get("@id", "")
            target = self._parse_concept(end_uri)
            if target:
                results.append((concept, target))
        
        return results
    
    def get_all_relations(
        self,
        concepts: List[str],
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Get all relations for a list of concepts.
        
        Args:
            concepts: List of concepts to query
            
        Returns:
            Dict with keys 'UsedFor', 'MadeOf', 'HasProperty', 
            each containing (subject, relation, target) tuples
        """
        results = {
            "UsedFor": [],
            "MadeOf": [],
            "HasProperty": [],
        }
        
        for concept in concepts:
            for subj, target in self.get_used_for(concept):
                results["UsedFor"].append((subj, "UsedFor", target))
            
            for subj, target in self.get_made_of(concept):
                results["MadeOf"].append((subj, "MadeOf", target))
            
            for subj, target in self.get_has_property(concept):
                results["HasProperty"].append((subj, "HasProperty", target))
        
        return results


# ============================================================================
# IDIOMS DATA LOADER
# ============================================================================

def download_idioms_csv(save_path: str = None) -> str:
    """
    Download idioms.csv from baiango/english_idioms repo.
    
    Args:
        save_path: Path to save file. If None, returns content as string.
        
    Returns:
        CSV content as string
    """
    url = "https://raw.githubusercontent.com/baiango/english_idioms/main/idioms.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved idioms to {save_path}")
        
        return content
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download idioms: {e}")
        raise


# ============================================================================
# UNIT TESTS
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("API CLIENTS TESTS")
    print("=" * 60)
    
    # Test 1: Wikidata (no API key needed)
    print("\n1. Testing WikidataClient:")
    try:
        wikidata = WikidataClient()
        capitals = wikidata.get_capitals()
        print(f"   ✅ Retrieved {len(capitals)} capitals")
        if capitals:
            print(f"   Sample: {capitals[:3]}")
    except Exception as e:
        print(f"   ❌ Wikidata failed: {e}")
    
    # Test 2: ConceptNet (no API key needed)
    print("\n2. Testing ConceptNetClient:")
    try:
        conceptnet = ConceptNetClient()
        scissors_uses = conceptnet.get_used_for("scissors")
        print(f"   ✅ Retrieved {len(scissors_uses)} uses for 'scissors'")
        if scissors_uses:
            print(f"   Sample: {scissors_uses[:3]}")
    except Exception as e:
        print(f"   ❌ ConceptNet failed: {e}")
    
    # Test 3: Idioms download
    print("\n3. Testing idioms download:")
    try:
        content = download_idioms_csv()
        lines = content.strip().split('\n')
        print(f"   ✅ Downloaded {len(lines)} lines")
        print(f"   Header: {lines[0][:60]}...")
    except Exception as e:
        print(f"   ❌ Idioms download failed: {e}")
    
    # Test 4: DeepSeek (only if API key set)
    print("\n4. Testing DeepSeekClient:")
    if os.environ.get("DEEPSEEK_API_KEY"):
        try:
            deepseek = DeepSeekClient()
            result = deepseek.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say 'Hello' and nothing else.",
                max_tokens=10,
            )
            print(f"   ✅ DeepSeek response: {result}")
        except Exception as e:
            print(f"   ❌ DeepSeek failed: {e}")
    else:
        print("   ⚠️ DEEPSEEK_API_KEY not set, skipping")
    
    print("\n" + "=" * 60)
    print("API clients tests complete!")
    print("=" * 60)
