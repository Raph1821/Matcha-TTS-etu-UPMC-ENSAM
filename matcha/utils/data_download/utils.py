# taken from https://github.com/pytorch/audio/blob/main/src/torchaudio/datasets/utils.py
# Copyright (c) 2017 Facebook Inc. (Soumith Chintala)
# Licence: BSD 2-Clause
# pylint: disable=C0123

import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Union

_LG = logging.getLogger(__name__)


def _extract_tar(from_path: Union[str, Path], to_path: Optional[str] = None, overwrite: bool = False) -> List[str]:
    if isinstance(from_path, Path):
        from_path = str(from_path)

    if to_path is None:
        to_path = os.path.dirname(from_path)

    import sys
    
    with tarfile.open(from_path, "r:*") as tar:
        members = tar.getmembers()
        total_files = sum(1 for m in members if m.isfile())
        extracted = 0
        skipped = 0
        
        bar_width = 50
        
        for file_ in members:
            file_path = os.path.join(to_path, file_.name)
            if file_.isfile():
                if os.path.exists(file_path):
                    skipped += 1
                    if not overwrite:
                        continue
                tar.extract(file_, to_path)
                extracted += 1
                
                if extracted % 10 == 0 or extracted == total_files:
                    progress = 100 * extracted // total_files if total_files > 0 else 0
                    filled = bar_width * extracted // total_files if total_files > 0 else 0
                    bar = '█' * filled + '░' * (bar_width - filled)
                    print(f"   [{bar}] {extracted}/{total_files} ({progress}%)", end='\r')
                    sys.stdout.flush()
            else:
                tar.extract(file_, to_path)
        
        progress = 100
        bar = '█' * bar_width
        print(f"   [{bar}] {extracted}/{total_files} ({progress}%)")
        
        if skipped > 0:
            print(f"   ({skipped} fichiers déjà existants, ignorés)")
        
        return [os.path.join(to_path, m.name) for m in members if m.isfile()]


def _extract_zip(from_path: Union[str, Path], to_path: Optional[str] = None, overwrite: bool = False) -> List[str]:
    if type(from_path) is Path:
        from_path = str(Path)

    if to_path is None:
        to_path = os.path.dirname(from_path)

    with zipfile.ZipFile(from_path, "r") as zfile:
        files = zfile.namelist()
        for file_ in files:
            file_path = os.path.join(to_path, file_)
            if os.path.exists(file_path):
                _LG.info("%s already extracted.", file_path)
                if not overwrite:
                    continue
            zfile.extract(file_, to_path)
    return files
