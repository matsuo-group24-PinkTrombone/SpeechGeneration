#!/bin/bash

echo "Working in $(pwd)"

REPOSITORY_URL="https://github.com/matsuo-group24-PinkTrombone/SpeechGeneration.git"

[ -d ".git" ] && git fetch && git pull
[ ! -d ".git" ] && git clone "$REPOSITORY_URL" ./

exec /bin/bash
