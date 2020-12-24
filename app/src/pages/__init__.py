from .page1 import Page1
from .page2 import Page2
from ..utils import Page

from typing import Dict, Type


PAGE_MAP: Dict[str, Type[Page]] = {
    "Grade 3": Page1,
    "Grade 6": Page2,
}

__all__ = ["PAGE_MAP"]