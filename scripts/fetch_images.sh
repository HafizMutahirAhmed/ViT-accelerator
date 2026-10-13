#!/usr/bin/env bash
# Download the ImageNet sample images. None of them is kept in the repo.
#
# Usage: scripts/fetch_images.sh
#
# python/image_lists.txt names them: "test <file>" goes to python/images/,
# "calibration <file>" to python/images_calibration/ and "heldout <file>" to
# python/images_heldout/. The files come from
# github.com/EliSchwartz/imagenet-sample-images, one image per ImageNet class.
# Files that are already there are not downloaded again.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
source_url=https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master

while read -r set_name name; do
    case "$set_name" in
        test) target="$root/python/images" ;;
        calibration|heldout) target="$root/python/images_$set_name" ;;
        *) continue ;;
    esac
    mkdir -p "$target"
    if [ ! -s "$target/$name" ]; then
        # an apostrophe in a file name is written %27 in the URL
        curl -fsSL --retry 3 -o "$target/$name.part" "$source_url/${name//\'/%27}"
        mv "$target/$name.part" "$target/$name"
    fi
done < "$root/python/image_lists.txt"

for directory in images images_calibration images_heldout; do
    echo "python/$directory: $(find "$root/python/$directory" -name '*.JPEG' | wc -l) images"
done
