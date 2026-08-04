#!/usr/bin/env bash
set -euo pipefail

UNITY_SOURCE="${1:-/project/pi_roohie_umass_edu/fokkerplanckDeeponet}"
REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
DESTINATION="${REPOSITORY_ROOT}/FP_PINN/legacy_source"

if [[ ! -d "${UNITY_SOURCE}" ]]; then
    echo "Unity source directory not found: ${UNITY_SOURCE}" >&2
    exit 2
fi

mkdir -p "${DESTINATION}"

if find "${DESTINATION}" -mindepth 1 -print -quit | grep -q .; then
    echo "Destination is not empty; refusing to overwrite: ${DESTINATION}" >&2
    exit 3
fi

cd "${UNITY_SOURCE}"

find . \
    -type d \( \
        -name .git -o \
        -name .ipynb_checkpoints -o \
        -name __pycache__ -o \
        -name runs -o \
        -name results -o \
        -name output -o \
        -name outputs -o \
        -name checkpoints -o \
        -name training_data -o \
        -name particle_data -o \
        -name flow_fields -o \
        -name 'isolated_*' -o \
        -name 'cylinder147_DPSD_*' \
    \) -prune -o \
    -type f -size -20M \( \
        -name '*.py' -o \
        -name '*.ipynb' -o \
        -name '*.sh' -o \
        -name '*.slurm' -o \
        -name '*.sbatch' -o \
        -name '*.f' -o \
        -name '*.f90' -o \
        -name '*.f95' -o \
        -name '*.c' -o \
        -name '*.cpp' -o \
        -name '*.h' -o \
        -name '*.hpp' -o \
        -name '*.cu' -o \
        -name '*.cuh' -o \
        -name '*.md' -o \
        -name '*.rst' -o \
        -name '*.tex' -o \
        -name '*.yaml' -o \
        -name '*.yml' -o \
        -name '*.json' -o \
        -name '*.toml' -o \
        -name '*.cfg' -o \
        -name '*.ini' -o \
        -name '*.txt' \
    \) -print0 |
while IFS= read -r -d '' SOURCE_FILE; do
    RELATIVE_PATH="${SOURCE_FILE#./}"
    mkdir -p "${DESTINATION}/$(dirname "${RELATIVE_PATH}")"
    cp -p -- "${SOURCE_FILE}" "${DESTINATION}/${RELATIVE_PATH}"
done

echo "Source synchronization completed."
echo "Destination: ${DESTINATION}"
echo "Files copied: $(find "${DESTINATION}" -type f | wc -l)"
du -sh "${DESTINATION}"
echo
echo "Review before committing:"
echo "  git status --short"
echo "  find FP_PINN/legacy_source -type f -size +90M -print"
