"""
Monotonic Alignment Search (MAS) - Implémentation optimisée Cython
Version de reproduction: réimplémentation avec structure et nommage différents
"""
import numpy as np

cimport cython
cimport numpy as np

from cython.parallel import prange
from libc.stdlib cimport malloc, free


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_single_alignment(
    int[:,::1] alignment_matrix, 
    float[:,::1] score_matrix, 
    int text_length, 
    int audio_length, 
    float boundary_penalty
) nogil:
    """
    Calcule le chemin d'alignement monotone pour un seul échantillon
    
    Args:
        alignment_matrix: matrice d'alignement de sortie [text_length, audio_length]
        score_matrix: matrice de scores d'entrée [text_length, audio_length] (sera modifiée)
        text_length: longueur de la séquence de texte
        audio_length: longueur de la séquence audio
        boundary_penalty: pénalité pour les bords (valeur négative maximale)
    """
    cdef int x, y
    cdef int x_min, x_max
    cdef float from_same, from_prev
    cdef float best_prev
    cdef int idx = text_length - 1

    cdef float* dp_prev = <float*> malloc(text_length * sizeof(float))
    cdef float* dp_cur = <float*> malloc(text_length * sizeof(float))
    cdef unsigned char* take_diag = <unsigned char*> malloc(text_length * audio_length * sizeof(unsigned char))

    if dp_prev == NULL or dp_cur == NULL or take_diag == NULL:
        if dp_prev != NULL:
            free(dp_prev)
        if dp_cur != NULL:
            free(dp_cur)
        if take_diag != NULL:
            free(take_diag)
        return

    for x in range(text_length):
        dp_prev[x] = boundary_penalty

    for y in range(audio_length):
        for x in range(text_length):
            dp_cur[x] = boundary_penalty

        x_min = max(0, text_length + y - audio_length)
        x_max = min(text_length, y + 1)

        for x in range(x_min, x_max):
            if x == 0:
                from_prev = 0.0 if y == 0 else boundary_penalty
            else:
                from_prev = dp_prev[x - 1]

            if x == y:
                from_same = boundary_penalty
            else:
                from_same = dp_prev[x] if y > 0 else boundary_penalty

            if from_prev >= from_same or x == y:
                best_prev = from_prev
                take_diag[x * audio_length + y] = 1
            else:
                best_prev = from_same
                take_diag[x * audio_length + y] = 0

            dp_cur[x] = best_prev + score_matrix[x, y]

        # swap buffers
        for x in range(text_length):
            dp_prev[x] = dp_cur[x]
            score_matrix[x, y] = dp_prev[x]

    for y in range(audio_length - 1, -1, -1):
        alignment_matrix[idx, y] = 1
        if y == 0:
            break
        if idx > 0 and (idx == y or take_diag[idx * audio_length + y] == 1):
            idx -= 1

    free(dp_prev)
    free(dp_cur)
    free(take_diag)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void compute_batch_alignments(
    int[:,:,::1] alignment_matrices, 
    float[:,:,::1] score_matrices, 
    int[::1] text_lengths, 
    int[::1] audio_lengths, 
    float boundary_penalty=-1e9
) nogil:
    """
    Calcule les chemins d'alignement monotones par lots (traitement parallèle)
    
    Args:
        alignment_matrices: matrices d'alignement de sortie [batch, text_length, audio_length]
        score_matrices: matrices de scores d'entrée [batch, text_length, audio_length] (seront modifiées)
        text_lengths: longueur de texte pour chaque échantillon [batch]
        audio_lengths: longueur audio pour chaque échantillon [batch]
        boundary_penalty: pénalité pour les bords (valeur négative maximale)
    """
    cdef int batch_size = score_matrices.shape[0]
    cdef int sample_idx
    
    for sample_idx in prange(batch_size, nogil=True):
        compute_single_alignment(
            alignment_matrices[sample_idx], 
            score_matrices[sample_idx], 
            text_lengths[sample_idx], 
            audio_lengths[sample_idx], 
            boundary_penalty
        )
