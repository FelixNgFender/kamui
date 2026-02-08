#!/usr/bin/env bash
rsync -aPv \
  --exclude=".git" \
  --filter=":- .gitignore" \
  --timeout=300 \
  --protocol=31 \
  --partial \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -o ConnectTimeout=60" \
  ~/workplaces/kamui/ \
  root@141.0.85.200:/workspace/kamui
# ssh -p 42423 root@141.0.85.200 -L 8080:localhost:8080
