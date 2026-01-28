#!/usr/bin/env bash
rsync -aPvz --filter=":- .gitignore" tvnguyen@turing.wpi.edu:~/kamui ~/workplaces/
