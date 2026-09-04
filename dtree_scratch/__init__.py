"""decision-tree-from-scratch: dependency-free CART trees and random forests.

No numpy, no scikit-learn — every split criterion, tree build, and
ensemble aggregation is plain Python. See README.md for scope and usage.
"""

from dtree_scratch.tree import DecisionTreeClassifier, DecisionTreeRegressor
from dtree_scratch.forest import RandomForestClassifier, RandomForestRegressor

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
]

__version__ = "0.1.0"
