ModusMate bundle
================

Загальний опис
--------------
Цей репозиторій містить повний набір матеріалів для проєкту ModusMate:
- firmware для плати PSoC Edge (board-side),
- host-side інструменти на Python (GUI, бенчмарки, тренування, керування моделями),
- матеріали кваліфікаційної роботи (report).

Структура
---------
- host-side/
  - Python-частина: зв'язок із платою, GUI, бенчмарки, тренування, реєстр моделей.
- board-side/ws-camera-imgproc-usb/
  - workspace ModusToolbox для прошивки.
- report/
  - LaTeX звіт, дані, графіки, зображення та скрипти генерації.

Швидкий старт: host-side
------------------------
1. Перейдіть у host-пакет:
    cd host-side/host
2. Створіть віртуальне середовище:
    python3 -m venv .venv
3. Встановіть залежності:
    .venv/bin/pip install -e .
    .venv/bin/pip install kagglehub

Збірка firmware
---------------
1. Перейдіть у директорію firmware-проєкту:
    cd board-side/ws-camera-imgproc-usb/camera-imgproc-usb
2. Зберіть проєкт:
    make build -j

Прошивка моделі
---------------
1. Перейдіть у host-side:
    cd host-side
2. Запустіть прошивку вибраної моделі:
    python -m modusmate_host.models flash <name> --port <port> \
         --fw <absolute_path_to_board-side/ws-camera-imgproc-usb/camera-imgproc-usb>

Робота зі звітом
----------------
Усі матеріали звіту зосереджені в report/:
- report/test.tex та report/all_results_tables.tex,
- report/data/ (таблиці й проміжні дані),
- report/plots/ і report/images/,
- report/scripts/ (генерація графіків).

Оновлення графіків звіту:
cd report
python scripts/generate_report_plots.py
python scripts/generate_family_plots.py
