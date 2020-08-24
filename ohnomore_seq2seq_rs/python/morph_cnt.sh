#!/bin/sh

in=${1:-/dev/stdin}

n=$(cut -f 6 $in | awk 'BEGIN {FS="\|"} {print NF}' | sort | uniq | tail -n 1)
cp config.toml config.toml.old

head -n -1 config.toml.old > config.toml
rm config.toml.old
echo "max_morph_tags = $n" >> config.toml
