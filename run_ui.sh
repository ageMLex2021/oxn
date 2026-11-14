#!/bin/bash
# OX-Neuro Hyper — запуск веб-интерфейса компрессора.
# Если сервер уже работает, просто открывает страницу в браузере.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
if ! curl -s -o /dev/null --max-time 2 http://127.0.0.1:8000/; then
  python3 ox_ui.py &
  sleep 2
fi
xdg-open http://127.0.0.1:8000
wait
