#!/usr/bin/env bash
rsync -aPvz \
  --exclude=".git" \
  --filter=":- .gitignore" tvnguyen@turing.wpi.edu:~/kamui ~/workplaces/
