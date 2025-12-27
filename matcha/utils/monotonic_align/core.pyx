"""
Monotonic Alignment Search (MAS) - Implémentation optimisée Cython
Version de reproduction: maintient la logique algorithmique, mais avec des ajustements dans l'implémentation
"""
import numpy as np

cimport cython
cimport numpy as np

from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each(
    int[:,::1] path, 
    float[:,::1] value, 
    int t_x, 
    int t_y, 
    float max_neg_val
) nogil:
    """
    Calcule le chemin d'alignement monotone pour un seul échantillon
    
    Args:
        path: matrice de chemin de sortie [t_x, t_y]
        value: matrice de valeurs d'entrée [t_x, t_y] (sera modifiée)
        t_x: longueur de la séquence de texte
        t_y: longueur de la séquence audio
        max_neg_val: valeur négative maximale (pour le traitement des bords)
    """
    cdef int x, y
    cdef float v_prev, v_cur
    cdef int index = t_x - 1

    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[x, y - 1]
            
            if x == 0:
                if y == 0:
                    v_prev = 0.0
                else:
                    v_prev = max_neg_val
            else:
                v_prev = value[x - 1, y - 1]
            
            value[x, y] = max(v_cur, v_prev) + value[x, y]

    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index = index - 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c(
    int[:,:,::1] paths, 
    float[:,:,::1] values, 
    int[::1] t_xs, 
    int[::1] t_ys, 
    float max_neg_val=-1e9
) nogil:
    """
    Calcule les chemins d'alignement monotones par lots (traitement parallèle)
    
    Args:
        paths: matrice de chemins de sortie [batch, t_x, t_y]
        values: matrice de valeurs d'entrée [batch, t_x, t_y] (sera modifiée)
        t_xs: longueur de texte pour chaque échantillon [batch]
        t_ys: longueur audio pour chaque échantillon [batch]
        max_neg_val: valeur négative maximale (pour le traitement des bords)
    """
    cdef int b = values.shape[0]
    cdef int i
    
    for i in prange(b, nogil=True):
        maximum_path_each(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)
