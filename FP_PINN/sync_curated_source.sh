#!/usr/bin/env bash
set -euo pipefail

UNITY_SOURCE="${1:-/project/pi_roohie_umass_edu/fokkerplanckDeeponet}"
REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
MANIFEST="${REPOSITORY_ROOT}/FP_PINN/legacy_manifest.txt"
DESTINATION="${REPOSITORY_ROOT}/FP_PINN/legacy_source"

if [[ ! -d "${UNITY_SOURCE}" ]]; then
    echo "Unity source directory not found: ${UNITY_SOURCE}" >&2
    exit 2
fi

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Manifest not found: ${MANIFEST}" >&2
    exit 3
fi

mkdir -p "${DESTINATION}"

if find "${DESTINATION}" -mindepth 1 -print -quit | grep -q .; then
    echo "Destination is not empty; refusing to overwrite: ${DESTINATION}" >&2
    exit 4
fi

MISSING=0
COPIED=0

while IFS= read -r RELATIVE_PATH || [[ -n "${RELATIVE_PATH}" ]]; do
    [[ -z "${RELATIVE_PATH}" ]] && continue
    [[ "${RELATIVE_PATH}" == \#* ]] && continue

    SOURCE_FILE="${UNITY_SOURCE}/${RELATIVE_PATH}"
    DESTINATION_FILE="${DESTINATION}/${RELATIVE_PATH}"

    if [[ ! -f "${SOURCE_FILE}" ]]; then
        echo "Missing source file: ${RELATIVE_PATH}" >&2
        MISSING=1
        continue
    fi

    FILE_BYTES="$(stat -c '%s' "${SOURCE_FILE}")"
    if (( FILE_BYTES > 20971520 )); then
        echo "Refusing file larger than 20 MiB: ${RELATIVE_PATH}" >&2
        MISSING=1
        continue
    fi

    mkdir -p "$(dirname "${DESTINATION_FILE}")"
    cp -p -- "${SOURCE_FILE}" "${DESTINATION_FILE}"
    COPIED=$((COPIED + 1))
done < "${MANIFEST}"

if (( MISSING != 0 )); then
    echo "Curated import is incomplete. Resolve the warnings before committing." >&2
    exit 5
fi

echo "Curated source synchronization completed."
echo "Files copied: ${COPIED}"
echo "Destination: ${DESTINATION}"
du -sh "${DESTINATION}"

echo
echo "Files selected:"
find "${DESTINATION}" -type f -printf '%P\n' | sort

echo
echo "Review before committing:"
echo "  git status --short"
echo "  git diff --stat"
echo "  find FP_PINN/legacy_source -type f -size +20M -print"
