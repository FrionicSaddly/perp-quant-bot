"""Trading strategies that are not single-name directional prediction."""
from .basis_carry import run_basis_carry
from .cross_exchange import run_cross_exchange
from .cross_sectional import run_cross_sectional
from .funding_carry import run_funding_carry
from .pairs import run_pairs

__all__ = [
    "run_cross_sectional", "run_funding_carry", "run_basis_carry",
    "run_cross_exchange", "run_pairs",
]
