"""
translator.config_presets.py - Quick presets for Translator configurations.
"""

from .config import TranslatorConfig

def DefaultTranslator():
    return TranslatorConfig()

def VerboseTranslator():
    return TranslatorConfig(verbose=True)

def FaultyTranslator():
    return TranslatorConfig(fault_tolerant=False)

def MinimalCodeTranslator():
    return TranslatorConfig(n_code=2)
