#!/usr/bin/env bash
rsync -aPv \
  --exclude=".git" \
  --filter=":- .gitignore" \
  --timeout=300 \
  --protocol=31 \
  --partial \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -o ConnectTimeout=60" \
  ~/workplaces/kamui/ \
  tvnguyen@turing.wpi.edu:~/kamui
