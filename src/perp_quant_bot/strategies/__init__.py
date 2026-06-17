"""Trading strategies that are not single-name directional prediction."""
from .cross_sectional import run_cross_sectional
from .funding_carry import run_funding_carry

__all__ = ["run_cross_sectional", "run_funding_carry"]
