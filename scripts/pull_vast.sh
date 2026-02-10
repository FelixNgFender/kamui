#!/usr/bin/env bash
rsync -aPvz \
  --exclude=".git" \
  --filter=":- .gitignore" \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -o ConnectTimeout=60" \
  root@felix.vast.ai:/workspace/kamui \
  ~/workplaces/
