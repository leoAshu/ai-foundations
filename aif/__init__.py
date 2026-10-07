"""Reusable base classes for the ai_foundations notebooks."""

from aif.data import DataModule
from aif.module import Classifier, Module
from aif.optim import SGD, BaseOptimizer

__version__ = '0.1.0'

__all__ = ['DataModule', 'Module', 'Classifier', 'BaseOptimizer', 'SGD']
