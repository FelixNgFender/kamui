#!/usr/bin/env bash
rsync -aPvz --filter=":- .gitignore" ~/workplaces/kamui/ tvnguyen@turing.wpi.edu:~/kamui
