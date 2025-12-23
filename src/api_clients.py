"""
api_clients.py - API Clients for External Data Sources

This module provides clients for:
- DeepSeek API (reasoner model for generation and validation)
- Wikidata SPARQL queries (for factual data)
- ConceptNet REST API (for common sense relations)

All clients include retry logic with exponential backoff.
"""

import os
import re
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

try:
    import requests
except ImportError:  # pragma: no cover - handled explicitly at runtime
    requests = None


def _require_requests():
    if requests is None:
        raise RuntimeError(
            "The 'requests' package is required for API calls. "
            "Install it with `pip install requests`."
        )
    return requests


def _load_env_file(path: "Path") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ if missing."""
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and not os.environ.get(key):
                    os.environ[key] = value
    except OSError as e:
        logger.warning("Failed to read .env file at %s: %s", path, e)


# ============================================================================
# DEEPSEEK API CLIENT
# ============================================================================

@dataclass
class DeepSeekClient:
    """
    Client for DeepSeek API (reasoner model).
    
    Handles:
    - Chat completions with retry logic
    - JSON parsing with fallback
    - Rate limiting
    - Full request/response logging to disk
    """
    
    api_key: Optional[str] = None
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    retry_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    json_retry_attempts: int = 4
    request_log: List[Dict] = field(default_factory=list)
    request_log_path: Optional[str] = None
    min_request_interval: float = 0.0
    _last_request_time: float = field(default=0.0, repr=False)

    def _get_deepseek_config(self) -> Dict[str, Any]:
        try:
            from .config import CONFIG
        except ImportError:
            try:
                from config import CONFIG
            except ImportError:
                return {}
        if isinstance(CONFIG, dict):
            return CONFIG.get("deepseek", {}) or {}
        return {}

    def _get_request_timeout(self, deepseek_config: Optional[Dict[str, Any]] = None) -> int:
        config = deepseek_config if deepseek_config is not None else self._get_deepseek_config()
        timeout = config.get("request_timeout_seconds")
        if isinstance(timeout, int) and timeout > 0:
            return timeout
        return 180

    def _example_value_for_type(self, expected_type: type) -> object:
        if isinstance(expected_type, tuple):
            expected_type = expected_type[0]
        if expected_type is bool:
            return True
        if expected_type is int:
            return 0
        if expected_type is float:
            return 0.0
        if expected_type is list:
            return []
        if expected_type is dict:
            return {}
        return "value"

    def _build_json_example(
        self,
        required_keys: Optional[List[str]] = None,
        required_schema: Optional[Dict[str, type]] = None,
    ) -> str:
        example: Dict[str, object] = {}
        if required_keys:
            for key in required_keys:
                example[key] = "value"
        if required_schema:
            for key, expected_type in required_schema.items():
                example[key] = self._example_value_for_type(expected_type)
        if not example:
            return "{}"
        return json.dumps(example, ensure_ascii=True)

    def _prepare_json_mode_prompts(
        self,
        system_prompt: str,
        user_prompt: str,
        required_keys: Optional[List[str]] = None,
        required_schema: Optional[Dict[str, type]] = None,
    ) -> Tuple[str, str]:
        combined = f"{system_prompt}\n{user_prompt}"
        has_json_word = "json" in combined.lower()
        has_example = bool(re.search(r"\{\s*\"", combined))
        additions: List[str] = []
        if not has_json_word:
            additions.append("Return a json object only.")
        if not has_example:
            example = self._build_json_example(required_keys, required_schema)
            additions.append(f"Example JSON:\n{example}")
        if additions:
            system_prompt = system_prompt.rstrip() + "\n\n" + "\n".join(additions)
        return system_prompt, user_prompt
    
    def __post_init__(self):
        """Load API key from .env file or environment."""
        if self.api_key is None:
            # Try loading from .env file first
            try:
                from dotenv import load_dotenv
                
                # Look for .env in project root
                env_path = Path(__file__).parent.parent / '.env'
                if env_path.exists():
                    load_dotenv(env_path)
            except ImportError:
                env_path = Path(__file__).parent.parent / '.env'
                if env_path.exists():
                    _load_env_file(env_path)
            
            if not os.environ.get("DEEPSEEK_API_KEY"):
                try:
                    env_path = Path(__file__).parent.parent / '.env'
                    if env_path.exists():
                        _load_env_file(env_path)
                except OSError:
                    pass

            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            logger.warning("DEEPSEEK_API_KEY not set. API calls will fail.")

        deepseek_config = self._get_deepseek_config()
        json_retry = deepseek_config.get("json_retry_attempts")
        if isinstance(json_retry, int) and json_retry > 0:
            self.json_retry_attempts = json_retry

        cfg_base_url = deepseek_config.get("base_url")
        if cfg_base_url and self.base_url == DEFAULT_DEEPSEEK_BASE_URL:
            self.base_url = cfg_base_url
        if self.base_url:
            self.base_url = self.base_url.rstrip("/")

        if self.request_log_path is None:
            env_log_path = os.environ.get("DEEPSEEK_LOG_PATH")
            if env_log_path:
                self.request_log_path = env_log_path
                self._validate_log_path()
                return
            try:
                from .config import get_base_paths
            except ImportError:
                try:
                    from config import get_base_paths
                except ImportError:
                    get_base_paths = None

            if get_base_paths is not None:
                try:
                    paths = get_base_paths()
                    self.request_log_path = str(paths["data_root"] / "deepseek_requests.jsonl")
                except Exception as e:
                    logger.warning("Could not resolve data_root for DeepSeek logging: %s", e)

            # Per spec: DeepSeek logging must not be silently disabled
            # If we still don't have a log path, use a fallback and warn
            if self.request_log_path is None:
                fallback_path = os.path.join(os.getcwd(), "deepseek_requests.jsonl")
                logger.warning(
                    "DeepSeek request_log_path not configured. "
                    "Using fallback: %s. Set DEEPSEEK_LOG_PATH env var or "
                    "configure get_base_paths() to specify log location.",
                    fallback_path
                )
                self.request_log_path = fallback_path
        self._validate_log_path()

    def _validate_log_path(self) -> None:
        """Ensure the DeepSeek request log path is writable."""
        if not self.request_log_path:
            raise RuntimeError("DeepSeek request_log_path is not configured.")
        try:
            from pathlib import Path
            path = Path(self.request_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not path.is_file():
                raise RuntimeError(f"{path} exists and is not a file.")
            with path.open("a", encoding="utf-8"):
                pass
        except Exception as e:
            raise RuntimeError(
                f"DeepSeek request_log_path is not writable: {self.request_log_path}"
            ) from e
    
    def _append_log(self, record: Dict) -> None:
        """Append a log record to the request log file."""
        if not self.request_log_path:
            raise RuntimeError("DeepSeek request_log_path is not configured.")
        try:
            from pathlib import Path
            path = Path(self.request_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            raise RuntimeError(
                f"Failed to write DeepSeek request log to {self.request_log_path}"
            ) from e
    
    def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        if self.min_request_interval > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(
        self,
        endpoint: str,
        payload: Dict,
        timeout: int = 180,
    ) -> Dict:
        """Make HTTP request with retry logic and full logging."""
        req = _require_requests()
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        last_error = None
        delay = self.retry_base_delay
        
        for attempt in range(self.retry_attempts):
            # Rate limit before each attempt
            self._rate_limit()
            
            log_record = {
                "timestamp": time.time(),
                "endpoint": endpoint,
                "payload": payload,
                "attempt": attempt + 1,
            }
            
            try:
                response = req.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                
                log_record["status_code"] = response.status_code
                
                # Try to parse response JSON
                resp_json = None
                try:
                    resp_json = response.json()
                    log_record["response_json"] = resp_json
                except json.JSONDecodeError:
                    log_record["response_text"] = response.text[:2000]
                
                if response.status_code == 200:
                    if resp_json is not None:
                        # Success path: log once with response_json
                        self.request_log.append(log_record)
                        self._append_log(log_record)
                        return resp_json
                    else:
                        # JSON parse failed on 200 - treat as transient error
                        # Set error before logging so disk log includes it
                        log_record["error"] = "invalid_json"
                        self.request_log.append(log_record)
                        self._append_log(log_record)
                        logger.warning(f"Invalid JSON on 200 response, retrying...")
                        last_error = "invalid_json"
                        time.sleep(delay)
                        delay = min(delay * 2, self.retry_max_delay)
                        continue
                
                # Non-200 status codes: log before handling
                self.request_log.append(log_record)
                self._append_log(log_record)
                
                if response.status_code == 429:  # Rate limited
                    logger.warning(f"Rate limited, waiting {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, self.retry_max_delay)
                elif response.status_code >= 500:  # Server error
                    logger.warning(f"Server error {response.status_code}, retrying...")
                    time.sleep(delay)
                    delay = min(delay * 2, self.retry_max_delay)
                else:
                    # Non-200, non-retryable error, log response text if not already
                    if "response_text" not in log_record:
                        log_record["response_text"] = response.text[:2000]
                    response.raise_for_status()
                    
            except req.exceptions.Timeout:
                logger.warning(f"Request timeout, attempt {attempt + 1}")
                last_error = "timeout"
                log_record["error"] = "timeout"
                self.request_log.append(log_record)
                self._append_log(log_record)
                time.sleep(delay)
                delay = min(delay * 2, self.retry_max_delay)
            except req.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                last_error = str(e)
                log_record["error"] = str(e)
                self.request_log.append(log_record)
                self._append_log(log_record)
                time.sleep(delay)
                delay = min(delay * 2, self.retry_max_delay)
        
        raise RuntimeError(f"DeepSeek API failed after {self.retry_attempts} attempts: {last_error}")
    
    def _extract_message_text(self, response: Dict[str, Any]) -> str:
        """Extract text content from DeepSeek response."""
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Unexpected response format: {response}")
            raise RuntimeError(f"Failed to parse DeepSeek response: {e}") from e
        return message.get("content") or ""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "deepseek-reasoner",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        Generate text using DeepSeek chat API.
        
        Args:
            system_prompt: System instruction
            user_prompt: User message
            model: Model to use (deepseek-reasoner)
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
        deepseek_config = self._get_deepseek_config()
        thinking = deepseek_config.get("thinking")
        if thinking:
            payload["thinking"] = thinking

        response = self._make_request(
            "chat/completions",
            payload,
            timeout=self._get_request_timeout(deepseek_config),
        )
        return self._extract_message_text(response)

    def _extract_json_candidates(self, content: str) -> List[str]:
        candidates: List[str] = []
        raw = (content or "").strip()
        if not raw:
            return candidates

        if "```" in raw:
            parts = raw.split("```")
            for idx in range(1, len(parts), 2):
                block = parts[idx].strip()
                if not block:
                    continue
                if block.lower().startswith("json"):
                    lines = block.splitlines()
                    block = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
                if block:
                    candidates.append(block)

        candidates.append(raw)

        if "{" in raw and "}" in raw:
            start = raw.find("{")
            end = raw.rfind("}")
            if start < end:
                candidates.append(raw[start:end + 1])

        if "[" in raw and "]" in raw:
            start = raw.find("[")
            end = raw.rfind("]")
            if start < end:
                candidates.append(raw[start:end + 1])

        seen = set()
        unique: List[str] = []
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    def _parse_json_content(self, content: str) -> Optional[object]:
        for candidate in self._extract_json_candidates(content):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None
    
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "deepseek-reasoner",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        retry_on_invalid: bool = True,
        required_keys: Optional[List[str]] = None,
        max_retries: Optional[int] = None,
        required_schema: Optional[Dict[str, type]] = None,
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
        max_retries = max_retries or self.json_retry_attempts
        last_error: Optional[str] = None

        deepseek_config = self._get_deepseek_config()
        response_format = deepseek_config.get("response_format")
        thinking = deepseek_config.get("thinking")
        timeout = self._get_request_timeout(deepseek_config)

        system_prompt, user_prompt = self._prepare_json_mode_prompts(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            required_keys=required_keys,
            required_schema=required_schema,
        )
        retry_prompt = user_prompt

        for attempt in range(max_retries):
            temp = temperature if attempt == 0 else 0.1
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": retry_prompt},
                ],
                "temperature": temp,
                "max_tokens": max_tokens,
            }
            if response_format:
                payload["response_format"] = response_format
            if thinking:
                payload["thinking"] = thinking

            response = self._make_request("chat/completions", payload, timeout=timeout)
            content = self._extract_message_text(response)
            if not content.strip():
                last_error = "empty_content"
                if not retry_on_invalid:
                    break
                retry_prompt = (
                    user_prompt
                    + "\n\nIMPORTANT: Return a non-empty json object only. "
                    "Do not include markdown or code fences. "
                    "Your response must start with '{' and end with '}'."
                )
                continue

            parsed = self._parse_json_content(content)
            if isinstance(parsed, list):
                parsed = {"prompts": parsed}

            if isinstance(parsed, dict):
                if required_keys and not all(k in parsed for k in required_keys):
                    last_error = "missing_required_keys"
                elif required_schema:
                    schema_ok = True
                    for key, expected_type in required_schema.items():
                        if key not in parsed or not isinstance(parsed.get(key), expected_type):
                            schema_ok = False
                            break
                        value = parsed.get(key)
                        if expected_type in (int, float) and isinstance(value, bool):
                            schema_ok = False
                            break
                    if not schema_ok:
                        last_error = "invalid_schema"
                    else:
                        return parsed
                else:
                    return parsed
            else:
                last_error = "invalid_json"

            if not retry_on_invalid:
                break

            required_hint = ""
            if required_keys:
                required_hint = " Required keys: " + ", ".join(required_keys) + "."
            if required_schema:
                type_map = {
                    list: "array",
                    dict: "object",
                    bool: "boolean",
                    int: "integer",
                    float: "number",
                    str: "string",
                }
                schema_bits = []
                for key, expected_type in required_schema.items():
                    if isinstance(expected_type, tuple):
                        type_names = [type_map.get(t, getattr(t, "__name__", "value")) for t in expected_type]
                        schema_bits.append(f"{key}=" + "|".join(type_names))
                    else:
                        schema_bits.append(f"{key}={type_map.get(expected_type, getattr(expected_type, '__name__', 'value'))}")
                if schema_bits:
                    required_hint += " Schema: " + ", ".join(schema_bits) + "."
            retry_prompt = (
                user_prompt
                + "\n\nIMPORTANT: Return ONLY valid json with double quotes. "
                "Do not include markdown or code fences. "
                "Your response must start with '{' and end with '}'."
                + required_hint
            )

        logger.error("Failed to get valid JSON after %d attempts (last_error=%s).", max_retries, last_error)
        raise RuntimeError("Failed to get valid JSON from DeepSeek.")
    
    def generate_json_r1(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> Dict:
        """
        Generate JSON using DeepSeek reasoner (thinking mode).
        
        Thin wrapper around generate_json with model set to deepseek-reasoner.
        
        Args:
            system_prompt: System instruction
            user_prompt: User message
            temperature: Sampling temperature (default 0.1 for consistency)
            max_tokens: Max tokens to generate
            
        Returns:
            Parsed JSON dict
        """
        return self.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="deepseek-reasoner",
            temperature=temperature,
            max_tokens=max_tokens,
            retry_on_invalid=True,
        )


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
        req = _require_requests()
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "SemanticGravityExperiment/1.0",
        }
        
        try:
            response = req.get(
                self.endpoint,
                params={"query": query},
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", {}).get("bindings", [])
            
        except req.exceptions.RequestException as e:
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
        
        req = _require_requests()
        try:
            response = req.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("edges", [])
        except req.exceptions.RequestException as e:
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
    
    req = _require_requests()
    try:
        response = req.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved idioms to {save_path}")
        
        return content
        
    except req.exceptions.RequestException as e:
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

    if requests is None:
        print("SKIP: 'requests' package not installed; API client tests skipped.")
        sys.exit(0)
    
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
    try:
        deepseek = DeepSeekClient()
        if deepseek.api_key:
            result = deepseek.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say 'Hello' and nothing else.",
                max_tokens=2000,
            )
            print(f"   ✅ DeepSeek response: {result}")
        else:
            print("   ⚠️ DEEPSEEK_API_KEY not set, skipping")
    except Exception as e:
        print(f"   ❌ DeepSeek failed: {e}")
    
    print("\n" + "=" * 60)
    print("API clients tests complete!")
    print("=" * 60)
