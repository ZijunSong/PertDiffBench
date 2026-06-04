#!/usr/bin/env bash
# One-time: copy existing Numba cache from ~/.cache/numba to /data/ppnm/cache/numba, then you can delete the old copy after verifying.
set -euo pipefail
SRC="${HOME}/.cache/numba"
DST="/data/ppnm/cache/numba"
if [[ ! -d "$SRC" ]]; then
  echo "Nothing to migrate: $SRC does not exist."
  exit 0
fi
mkdir -p "$DST"
echo "Syncing $SRC -> $DST ..."
rsync -a --info=stats2 "$SRC/" "$DST/"
echo "Done. After a successful training run using NUMBA_CACHE_DIR=$DST, you may free space with:"
echo "  rm -rf \"$SRC\""
