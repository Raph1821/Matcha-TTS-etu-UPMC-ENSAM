"""
Utilitaires de journalisation pour environnements multi-GPU
Version de reproduction: réimplémentation avec structure différente
"""
import logging
from typing import Callable

_rank_zero_decorator = None
try:
    from pytorch_lightning.utilities import rank_zero_only as _rank_zero_decorator
except ImportError:
    try:
        from lightning.pytorch.utilities import rank_zero_only as _rank_zero_decorator
    except ImportError:
        pass


def _create_noop_decorator() -> Callable:
    """Crée un décorateur no-op si rank_zero_only n'est pas disponible"""
    def noop_wrapper(func: Callable) -> Callable:
        return func
    return noop_wrapper


def _apply_rank_zero_filter(logger_instance: logging.Logger) -> None:
    """
    Applique le filtre rank-zero à toutes les méthodes de journalisation
    
    En environnement multi-GPU, seul le processus rank 0 doit enregistrer
    les logs pour éviter la duplication.
    """
    decorator = _rank_zero_decorator if _rank_zero_decorator is not None else _create_noop_decorator()
    
    log_methods = [
        "debug", "info", "warning", "error", 
        "exception", "fatal", "critical"
    ]
    
    for method_name in log_methods:
        original_method = getattr(logger_instance, method_name, None)
        if original_method is not None:
            wrapped_method = decorator(original_method)
            setattr(logger_instance, method_name, wrapped_method)


def get_pylogger(logger_name: str = __name__) -> logging.Logger:
    """
    Crée et configure un logger compatible multi-GPU
    
    Args:
        logger_name: Nom du logger (par défaut: nom du module appelant)
        
    Returns:
        Logger configuré avec filtrage rank-zero pour multi-GPU
    """
    logger = logging.getLogger(logger_name)
    
    _apply_rank_zero_filter(logger)
    
    return logger

