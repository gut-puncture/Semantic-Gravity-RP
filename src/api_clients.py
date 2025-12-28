"""
api_clients.py - API Clients for External Data Sources

This module provides clients for:
- OpenAI Responses API (GPT-5.2 for generation and validation, batch-only workflow)
- Wikidata SPARQL queries (for factual data)
- Rest Countries REST API (for country metadata)
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

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

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
# OPENAI GPT-5.2 API CLIENT (BATCH WORKFLOW)
# ============================================================================

@dataclass
class OpenAIClient:
    """
    Client for OpenAI Responses API (GPT-5.2).

    Handles:
    - Responses requests with retry logic
    - JSON parsing with fallback
    - Rate limiting
    - Batch file submission + retrieval
    - Full request/response logging to disk
    """

    api_key: Optional[str] = None
    base_url: str = DEFAULT_OPENAI_BASE_URL
    retry_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    json_retry_attempts: int = 4
    request_log: List[Dict] = field(default_factory=list)
    request_log_path: Optional[str] = None
    min_request_interval: float = 0.0
    _last_request_time: float = field(default=0.0, repr=False)

    def _get_openai_config(self) -> Dict[str, Any]:
        try:
            from .config import CONFIG
        except ImportError:
            try:
                from config import CONFIG
            except ImportError:
                return {}
        if isinstance(CONFIG, dict):
            return CONFIG.get("openai", {}) or {}
        return {}

    def _get_request_timeout(self, openai_config: Optional[Dict[str, Any]] = None) -> int:
        config = openai_config if openai_config is not None else self._get_openai_config()
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

            if not os.environ.get("OPENAI_API_KEY"):
                try:
                    env_path = Path(__file__).parent.parent / '.env'
                    if env_path.exists():
                        _load_env_file(env_path)
                except OSError:
                    pass

            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set. API calls will fail.")

        openai_config = self._get_openai_config()
        json_retry = openai_config.get("json_retry_attempts")
        if isinstance(json_retry, int) and json_retry > 0:
            self.json_retry_attempts = json_retry

        cfg_base_url = openai_config.get("base_url")
        if cfg_base_url and self.base_url == DEFAULT_OPENAI_BASE_URL:
            self.base_url = cfg_base_url
        if self.base_url:
            self.base_url = self.base_url.rstrip("/")

        if self.request_log_path is None:
            env_log_path = os.environ.get("OPENAI_LOG_PATH")
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
                    self.request_log_path = str(paths["data_root"] / "gpt5_requests.jsonl")
                except Exception as e:
                    logger.warning("Could not resolve data_root for GPT-5.2 logging: %s", e)

            # Per spec: logging must not be silently disabled
            if self.request_log_path is None:
                fallback_path = os.path.join(os.getcwd(), "gpt5_requests.jsonl")
                logger.warning(
                    "GPT-5.2 request_log_path not configured. "
                    "Using fallback: %s. Set OPENAI_LOG_PATH env var or "
                    "configure get_base_paths() to specify log location.",
                    fallback_path
                )
                self.request_log_path = fallback_path
        self._validate_log_path()

    def _validate_log_path(self) -> None:
        """Ensure the request log path is writable."""
        if not self.request_log_path:
            raise RuntimeError("OpenAI request_log_path is not configured.")
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
                f"OpenAI request_log_path is not writable: {self.request_log_path}"
            ) from e
    
    def _append_log(self, record: Dict) -> None:
        """Append a log record to the request log file."""
        if not self.request_log_path:
            raise RuntimeError("OpenAI request_log_path is not configured.")
        try:
            from pathlib import Path
            path = Path(self.request_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            raise RuntimeError(
                f"Failed to write OpenAI request log to {self.request_log_path}"
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
        
        raise RuntimeError(f"OpenAI API failed after {self.retry_attempts} attempts: {last_error}")
    
    def _extract_message_text(self, response: Dict[str, Any]) -> str:
        """Extract text content from OpenAI response."""
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected response format: {response}")

        body = response.get("body")
        if isinstance(body, dict):
            return self._extract_message_text(body)

        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        output = response.get("output")
        if isinstance(output, list):
            chunks: List[str] = []
            for item in output:
                content = item.get("content") if isinstance(item, dict) else None
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    text = part.get("text")
                    if part_type in ("output_text", "text") and text:
                        chunks.append(text)
            if chunks:
                return "\n".join(chunks)

        try:
            message = response["choices"][0]["message"]
            return message.get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Unexpected response format: %s", response)
            raise RuntimeError(f"Failed to parse OpenAI response: {e}") from e

    def _build_responses_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2000,
        text_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config = self._get_openai_config()
        payload = {
            "model": model or config.get("model", "gpt-5.2-2025-12-11"),
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "store": config.get("store", True),
            "reasoning": config.get("reasoning", {"effort": "none"}),
        }
        if text_format is None:
            text_format = config.get("text_format") or config.get("response_format")
        text_block: Dict[str, Any] = {}
        if text_format:
            text_block["format"] = text_format
        text_verbosity = config.get("verbosity")
        if text_verbosity:
            text_block["verbosity"] = text_verbosity
        if text_block:
            payload["text"] = text_block
        return payload

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        Generate text using OpenAI Responses API.

        Args:
            system_prompt: System instruction
            user_prompt: User message
            model: Model to use (gpt-5.2-2025-12-11)
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            Generated text content
        """
        payload = self._build_responses_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = self._make_request(
            "responses",
            payload,
            timeout=self._get_request_timeout(self._get_openai_config()),
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
        model: Optional[str] = None,
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

        openai_config = self._get_openai_config()
        text_format = openai_config.get("text_format") or openai_config.get("response_format")
        timeout = self._get_request_timeout(openai_config)

        system_prompt, user_prompt = self._prepare_json_mode_prompts(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            required_keys=required_keys,
            required_schema=required_schema,
        )
        retry_prompt = user_prompt

        for attempt in range(max_retries):
            temp = temperature if attempt == 0 else 0.1
            payload = self._build_responses_payload(
                system_prompt=system_prompt,
                user_prompt=retry_prompt,
                model=model,
                temperature=temp,
                max_output_tokens=max_tokens,
                text_format=text_format,
            )

            response = self._make_request("responses", payload, timeout=timeout)
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
        raise RuntimeError("Failed to get valid JSON from OpenAI.")
    
    def generate_json_r1(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> Dict:
        """
        Generate JSON using GPT-5.2 (reasoning effort none).

        Thin wrapper around generate_json with model set to gpt-5.2-2025-12-11.
        """
        return self.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="gpt-5.2-2025-12-11",
            temperature=temperature,
            max_tokens=max_tokens,
            retry_on_invalid=True,
        )

    def build_batch_request(
        self,
        custom_id: str,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2000,
        text_format: Optional[Dict[str, Any]] = None,
        endpoint: str = "/v1/responses",
    ) -> Dict[str, Any]:
        payload = self._build_responses_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            text_format=text_format,
        )
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": endpoint,
            "body": payload,
        }

    def write_batch_requests(self, requests: List[Dict[str, Any]], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for req in requests:
                f.write(json.dumps(req, ensure_ascii=True) + "\n")
        return output_path

    def upload_batch_file(self, input_path: Path) -> str:
        req = _require_requests()
        url = f"{self.base_url}/files"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"file": (input_path.name, input_path.read_bytes())}
        data = {"purpose": "batch"}
        response = req.post(url, headers=headers, files=files, data=data, timeout=self._get_request_timeout())
        response.raise_for_status()
        payload = response.json()
        return payload.get("id", "")

    def create_batch(self, input_file_id: str, completion_window: Optional[str] = None) -> Dict[str, Any]:
        config = self._get_openai_config()
        window = completion_window or config.get("batch_completion_window", "24h")
        payload = {
            "input_file_id": input_file_id,
            "endpoint": "/v1/responses",
            "completion_window": window,
        }
        return self._make_request("batches", payload, timeout=self._get_request_timeout(config))

    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        req = _require_requests()
        url = f"{self.base_url}/batches/{batch_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = req.get(url, headers=headers, timeout=self._get_request_timeout())
        response.raise_for_status()
        return response.json()

    def download_file(self, file_id: str, output_path: Path) -> Path:
        req = _require_requests()
        url = f"{self.base_url}/files/{file_id}/content"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = req.get(url, headers=headers, timeout=self._get_request_timeout())
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return output_path

    def parse_batch_output(self, output_path: Path) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                custom_id = payload.get("custom_id")
                if not custom_id:
                    continue
                results[custom_id] = payload
        return results

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: float = 5.0,
        timeout_seconds: float = 6 * 60 * 60,
    ) -> Dict[str, Any]:
        start_time = time.time()
        while True:
            status_payload = self.get_batch(batch_id)
            status = status_payload.get("status")
            if status in ("completed", "failed", "canceled", "expired"):
                return status_payload
            if time.time() - start_time > timeout_seconds:
                raise RuntimeError(f"Batch {batch_id} timed out after {timeout_seconds} seconds.")
            time.sleep(poll_interval)

    def run_batch_requests(
        self,
        requests: List[Dict[str, Any]],
        batch_dir: Path,
        batch_name: str,
        poll_interval: float = 5.0,
        timeout_seconds: float = 6 * 60 * 60,
    ) -> Dict[str, Dict[str, Any]]:
        batch_dir.mkdir(parents=True, exist_ok=True)
        input_path = batch_dir / f"{batch_name}_input.jsonl"
        output_path = batch_dir / f"{batch_name}_output.jsonl"

        self.write_batch_requests(requests, input_path)
        input_file_id = self.upload_batch_file(input_path)
        if not input_file_id:
            raise RuntimeError("Failed to upload batch input file.")

        batch = self.create_batch(input_file_id)
        batch_id = batch.get("id")
        if not batch_id:
            raise RuntimeError(f"Batch creation failed: {batch}")

        status_payload = self.wait_for_batch(
            batch_id,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
        )
        status = status_payload.get("status")
        if status != "completed":
            raise RuntimeError(f"Batch {batch_id} ended with status {status}.")

        output_file_id = status_payload.get("output_file_id")
        if not output_file_id:
            raise RuntimeError(f"Batch {batch_id} completed without output_file_id.")

        self.download_file(output_file_id, output_path)
        return self.parse_batch_output(output_path)

    def extract_json_from_batch_payload(self, payload: Dict[str, Any]) -> Optional[object]:
        response = payload.get("response")
        if response is None:
            return None
        content = self._extract_message_text(response)
        return self._parse_json_content(content)


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
    - Person occupations
    - Person demonyms (via citizenship)
    - Person birth months
    - Country national sports
    """
    
    endpoint: str = "https://query.wikidata.org/sparql"
    timeout: int = 120
    
    # SPARQL queries per specification
    QUERY_CAPITALS = """
    SELECT ?country ?countryLabel ?capital ?capitalLabel WHERE {
      ?country wdt:P31 wd:Q6256.  # instance of country
      ?country wdt:P36 ?capital.   # has capital
      ?country rdfs:label ?countryLabel.
      ?capital rdfs:label ?capitalLabel.
      FILTER(LANG(?countryLabel) = "en")
      FILTER(LANG(?capitalLabel) = "en")
    }
    LIMIT 500
    """
    
    QUERY_CURRENCIES = """
    SELECT ?country ?countryLabel ?currency ?currencyLabel WHERE {
      ?country wdt:P31 wd:Q6256.    # instance of country
      ?country wdt:P38 ?currency.   # has currency
      ?country rdfs:label ?countryLabel.
      ?currency rdfs:label ?currencyLabel.
      FILTER(LANG(?countryLabel) = "en")
      FILTER(LANG(?currencyLabel) = "en")
    }
    LIMIT 500
    """

    QUERY_OCCUPATIONS = """
    SELECT ?person ?personLabel ?occupation ?occupationLabel WHERE {
      ?person wdt:P31 wd:Q5.         # instance of human
      ?person wdt:P106 ?occupation.  # occupation
      ?person rdfs:label ?personLabel.
      ?occupation rdfs:label ?occupationLabel.
      FILTER(LANG(?personLabel) = "en")
      FILTER(LANG(?occupationLabel) = "en")
    }
    LIMIT 200
    """

    QUERY_DEMONYMS = """
    SELECT ?person ?personLabel ?demonym WHERE {
      ?person wdt:P31 wd:Q5.
      ?person wdt:P27 ?country.   # citizenship
      ?country wdt:P1549 ?demonym.
      ?person rdfs:label ?personLabel.
      FILTER(LANG(?personLabel) = "en")
      FILTER(LANG(?demonym) = "en")
    }
    LIMIT 200
    """

    QUERY_BIRTH_MONTHS = """
    SELECT ?person ?personLabel ?birthdate WHERE {
      ?person wdt:P31 wd:Q5.
      ?person p:P569 ?birthStatement.
      ?birthStatement ps:P569 ?birthdate.
      ?birthStatement psv:P569 ?valueNode.
      ?valueNode wikibase:timePrecision ?precision.
      FILTER(?precision >= 10)
      ?person rdfs:label ?personLabel.
      FILTER(LANG(?personLabel) = "en")
    }
    LIMIT 200
    """

    QUERY_NATIONAL_SPORTS = """
    SELECT ?country ?countryLabel ?sport ?sportLabel WHERE {
      ?country wdt:P31 wd:Q6256.     # instance of country
      ?country wdt:P241 ?sport.      # national sport
      ?country rdfs:label ?countryLabel.
      ?sport rdfs:label ?sportLabel.
      FILTER(LANG(?countryLabel) = "en")
      FILTER(LANG(?sportLabel) = "en")
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

        retry_attempts = 3
        for attempt in range(retry_attempts):
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
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < retry_attempts - 1:
                    delay = 2 ** attempt
                    logger.warning(
                        "Wikidata query failed with status %s (attempt %d/%d); retrying in %ds.",
                        status,
                        attempt + 1,
                        retry_attempts,
                        delay,
                    )
                    time.sleep(delay)
                    continue
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

    def get_occupations(self) -> List[Tuple[str, str]]:
        """
        Get list of (person, occupation) pairs.

        Filters to occupations with single-word alphabetic labels.
        """
        results = self._execute_query(self.QUERY_OCCUPATIONS)
        pairs = []
        for row in results:
            person = row.get("personLabel", {}).get("value", "")
            occupation = row.get("occupationLabel", {}).get("value", "")
            if person and occupation:
                occupation_clean = occupation.strip()
                if occupation_clean.isalpha() and " " not in occupation_clean:
                    pairs.append((person, occupation_clean))
        logger.info(f"Retrieved {len(pairs)} valid occupation pairs from Wikidata")
        return pairs

    def get_demonyms(self) -> List[Tuple[str, str]]:
        """
        Get list of (person, demonym) pairs.

        Filters to demonyms with single-word alphabetic labels.
        """
        results = self._execute_query(self.QUERY_DEMONYMS)
        pairs = []
        for row in results:
            person = row.get("personLabel", {}).get("value", "")
            demonym = row.get("demonym", {}).get("value", "")
            if person and demonym:
                demonym_clean = demonym.strip()
                if demonym_clean.isalpha() and " " not in demonym_clean:
                    pairs.append((person, demonym_clean))
        logger.info(f"Retrieved {len(pairs)} valid demonym pairs from Wikidata")
        return pairs

    def get_birth_months(self) -> List[Tuple[str, str]]:
        """
        Get list of (person, birth_month_name) pairs.
        """
        results = self._execute_query(self.QUERY_BIRTH_MONTHS)
        month_map = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December",
        }
        pairs = []
        for row in results:
            person = row.get("personLabel", {}).get("value", "")
            birthdate = row.get("birthdate", {}).get("value", "")
            if not person or not birthdate:
                continue
            match = re.search(r"[+-]?\d{4,}-([0-1]\d)-", str(birthdate))
            if not match:
                continue
            try:
                month_num = int(match.group(1))
            except ValueError:
                continue
            month_name = month_map.get(month_num)
            if month_name:
                pairs.append((person, month_name))
        logger.info(f"Retrieved {len(pairs)} valid birth month pairs from Wikidata")
        return pairs

    def get_national_sports(self) -> List[Tuple[str, str]]:
        """
        Get list of (country, sport) pairs.

        Filters to sport names with single-word alphabetic labels.
        """
        results = self._execute_query(self.QUERY_NATIONAL_SPORTS)
        pairs = []
        for row in results:
            country = row.get("countryLabel", {}).get("value", "")
            sport = row.get("sportLabel", {}).get("value", "")
            if country and sport:
                sport_clean = sport.strip()
                if sport_clean.isalpha() and " " not in sport_clean:
                    pairs.append((country, sport_clean))
        logger.info(f"Retrieved {len(pairs)} valid national sport pairs from Wikidata")
        return pairs


# ============================================================================
# REST COUNTRIES API CLIENT
# ============================================================================

@dataclass
class RestCountriesClient:
    """
    Client for the Rest Countries API (v3.1).

    Fetches country metadata for:
    - Capitals
    - Currencies
    - Languages
    - Continents
    """

    endpoint: str = "https://restcountries.com/v3.1/all"
    timeout: int = 30
    fields: str = "name,capital,currencies,languages,continents"

    def fetch_all(self) -> List[Dict[str, Any]]:
        req = _require_requests()
        params = {"fields": self.fields}
        response = req.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("Unexpected Rest Countries response shape.")
        return payload

    def load_or_fetch(self, cache_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        if cache_path and cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Failed to read Rest Countries cache at %s; refetching.", cache_path)
        payload = self.fetch_all()
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        return payload

    def get_country_records(self, cache_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        payload = self.load_or_fetch(cache_path)
        records: List[Dict[str, Any]] = []
        for row in payload:
            name = (row.get("name") or {}).get("common")
            if not name:
                continue
            records.append({
                "name": name,
                "capitals": row.get("capital") or [],
                "currencies": row.get("currencies") or {},
                "languages": row.get("languages") or {},
                "continents": row.get("continents") or [],
            })
        return records


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
    
    # Test 4: OpenAI (only if API key set)
    print("\n4. Testing OpenAIClient:")
    try:
        client = OpenAIClient()
        if client.api_key:
            result = client.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say 'Hello' and nothing else.",
                max_tokens=2000,
            )
            print(f"   ✅ OpenAI response: {result}")
        else:
            print("   ⚠️ OPENAI_API_KEY not set, skipping")
    except Exception as e:
        print(f"   ❌ OpenAI failed: {e}")
    
    print("\n" + "=" * 60)
    print("API clients tests complete!")
    print("=" * 60)
