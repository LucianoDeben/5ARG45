from pathlib import Path
import joblib
import logging
from typing import Any, Optional

class CacheManager:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        if cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def get(self, key: str) -> Optional[Any]:
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{key}.joblib"
        if cache_file.exists():
            logging.debug(f"Loading cached data for key: {key}")
            return joblib.load(cache_file)
        return None
        
    def set(self, key: str, data: Any) -> None:
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.joblib"
            logging.debug(f"Caching data for key: {key}")
            joblib.dump(data, cache_file)