#!/usr/bin/env bash
rsync -aPvz \
  --exclude=".git" \
  --filter=":- .gitignore" \
  root@141.0.85.200:/workspace/kamui \
  ~/workplaces/
