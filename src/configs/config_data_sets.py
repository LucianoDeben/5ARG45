from dataclasses import dataclass, field
from typing import Optional, Union, List, Type
from pathlib import Path
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

@dataclass
class LINCSDatasetConfig:
    gctx_path: Path
    in_memory: bool = False
    scaler: Optional[TransformerMixin] = field(
        default_factory=lambda: StandardScaler()
    )
    feature_space: Union[str, List[str]] = "landmark"
    chunk_size: int = 2000
    cache_dir: Optional[Path] = None

    @staticmethod
    def with_minmax_scaler() -> 'LINCSDatasetConfig':
        """Factory method for MinMaxScaler configuration"""
        return LINCSDatasetConfig(
            scaler=MinMaxScaler()
        )

    @staticmethod
    def with_standard_scaler() -> 'LINCSDatasetConfig':
        """Factory method for StandardScaler configuration"""
        return LINCSDatasetConfig(
            scaler=StandardScaler()
        )