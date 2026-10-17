"""Reusable base classes and models for the ai_foundations notebooks."""

from aif.data import DataModule
from aif.device import cpu, gpu, num_gpus, try_gpu
from aif.models import LinearRegression, SoftmaxRegression
from aif.module import Classifier, Module
from aif.optim import SGD, BaseOptimizer
from aif.plotting import plot, plot_history, show_images
from aif.text import BaseTimeMachine, TimeMachine, Vocab
from aif.trainer import Trainer

__version__ = '0.1.0'

__all__ = [
    'DataModule',
    'Module',
    'Classifier',
    'LinearRegression',
    'SoftmaxRegression',
    'BaseOptimizer',
    'SGD',
    'Trainer',
    'plot',
    'plot_history',
    'show_images',
    'cpu',
    'gpu',
    'num_gpus',
    'try_gpu',
    'BaseTimeMachine',
    'TimeMachine',
    'Vocab',
]
