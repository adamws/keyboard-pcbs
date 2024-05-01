#!/bin/sh
set -xe

eeschema $1 &
EESCHEMA_PID=$!

# long sleeps because it take a while on circleci
sleep 10
xdotool key Return
sleep 2
xdotool key Return
sleep 2
xdotool key Return
sleep 2
xdotool key ctrl+s
sleep 2

kill -9 $EESCHEMA_PID
