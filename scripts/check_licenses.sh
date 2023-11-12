#!/bin/bash
set -e

pip-licenses --from=mixed  --ignore-packages `cat .libraries-whitelist.txt`> licenses.txt
cat licenses.txt

FOUND=$(tail -n +2 licenses.txt | grep -v -f .license-whitelist.txt | wc -l)

if [[ $FOUND -gt 0 ]]; then
  echo "Found licenses that are not on the whitelist."
  echo "$FOUND"
  tail -n +2 licenses.txt | grep -v -f .license-whitelist.txt
  exit 1
else
  echo "All licenses are on the whitelist."
  exit 0
fi
