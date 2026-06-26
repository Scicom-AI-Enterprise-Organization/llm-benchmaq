"""Configuration utilities for benchmaq."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import yaml


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    status: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, 
            "status": self.status, 
            "metrics": self.metrics, 
            "duration": self.duration, 
            "error": self.error
        }


def load_config(config_path: str, _seen: Optional[set] = None) -> dict:
    """Load a YAML config, resolving any ``extends:`` template references.

    A config may pull in shared provider templates via an ``extends:`` key
    holding a template path (string) or a list of paths, e.g.::

        extends: ../templates/runpod_h200.yaml
        benchmark:
          - name: qwen3.6-27b-voice
            model: { repo_id: Qwen/Qwen3.6-27B }
            bench: [ ... ]

    Templates are plain config files themselves (so a template can ``extends:``
    another). Paths are resolved relative to the file doing the extending.
    Merging is **shallow**: top-level keys in the child override the template
    wholesale. With multiple templates, later ones win over earlier ones, and
    the child config wins over all of them. The ``extends`` key is stripped
    from the returned config.
    """
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)

    # Guard against extends cycles (a -> b -> a).
    _seen = _seen or set()
    if config_path in _seen:
        chain = " -> ".join(list(_seen) + [config_path])
        raise ValueError(f"Circular 'extends' detected: {chain}")
    _seen = _seen | {config_path}

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    extends = config.pop("extends", None)
    if not extends:
        return config

    if isinstance(extends, str):
        extends = [extends]

    base_dir = os.path.dirname(config_path)
    merged: Dict[str, Any] = {}
    for template in extends:
        tpl_path = template if os.path.isabs(template) else os.path.join(base_dir, template)
        merged.update(load_config(tpl_path, _seen))  # later templates override earlier
    merged.update(config)  # child config overrides all templates
    return merged
