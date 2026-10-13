#!/usr/bin/env bash
# Download the calibration and held-out images that are not kept in the repo.
#
# Usage: scripts/fetch_images.sh
#
# python/image_lists.txt names them: "calibration <file>" goes to
# python/images_calibration/ and "heldout <file>" to python/images_heldout/.
# The files come from github.com/EliSchwartz/imagenet-sample-images, one image
# per ImageNet class. Files that are already there are not downloaded again.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
source_url=https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master

while read -r set_name name; do
    case "$set_name" in
        calibration|heldout) ;;
        *) continue ;;
    esac
    target="$root/python/images_$set_name"
    mkdir -p "$target"
    if [ ! -s "$target/$name" ]; then
        curl -fsSL --retry 3 -o "$target/$name.part" "$source_url/$name"
        mv "$target/$name.part" "$target/$name"
    fi
done < "$root/python/image_lists.txt"

for set_name in calibration heldout; do
    echo "$set_name: $(find "$root/python/images_$set_name" -name '*.JPEG' | wc -l) images"
done
