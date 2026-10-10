# Лабораторна робота №3
## "Машинне навчання з підкріпленням"
## Ходаков Максим Олегович ШІ-2

**Тема:** покращення алгоритму Deep Q-Learning для задачі
[CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/).

Бейзлайн курсу: [dqn_car_racing.py](https://github.com/PavloZakala/ReinforcementLearning/blob/main/dqn_car_racing.py),
заявлена якість 840.6 ± 103.2 на 50 тестових епізодах.
Умова: параметри середовища (`make_env`) та тестовий скрипт (`test_car_racing.py`) не змінюються.

### Що покращено (див. docstring у `dqn_car_racing.py` та розділ 4 звіту)
- Dueling-архітектура Q-мережі;
- ціль Double DQN з 3-кроковими поверненнями;
- коректний бутстреп при обрізанні епізоду за часом (truncated ≠ terminal);
- компактний replay-буфер (кожен кадр зберігається один раз, 120k переходів);
- ε-розклад за кроками, косинусне згасання learning rate, Polyak-оновлення цільової мережі;
- ранній обрив тренувальних епізодів без прогресу (лише в тренувальних середовищах);
- вибір чекпоінта за жадібною оцінкою на 16 нових трасах;
- асинхронний крок середовищ паралельно з оновленнями мережі на GPU (MPS/CUDA).

### Структура
| Файл | Призначення |
|---|---|
| `dqn_car_racing.py` | покращена реалізація (тренування, модель, буфер) |
| `test_car_racing.py` | тестовий скрипт курсу без змін |
| `checkpoint_CarRacing-v3_best.pth` | найкращий чекпоінт покращеної моделі |
| `baseline/` | оригінальні код, тест і чекпоінт курсу для порівняння |
| `plot_observation.py`, `plot_results.py` | рисунки для звіту |
| `build_report.py` | генерація звіту за ДСТУ 3008:2015 |
| `results/` | журнали тренування, результати тестів, рисунки |
| `report/` | звіт (DOCX + PDF) |

### Запуск
```bash
pip install torch "gymnasium[box2d]" opencv-python-headless matplotlib scipy python-docx pypdf
# box2d-py на Python 3.14 / macOS збирається так:
#   pip install swig setuptools wheel
#   SDKROOT=$(xcrun --show-sdk-path) CPLUS_INCLUDE_PATH=$(xcrun --show-sdk-path)/usr/include/c++/v1 \
#     pip install --no-build-isolation box2d-py

python dqn_car_racing.py --total-steps 1500000          # тренування (~3.5 год на Apple M1)
python test_car_racing.py checkpoint_CarRacing-v3_best.pth 50
cd baseline && python test_car_racing.py checkpoint_CarRacing-v3_best.pth 50   # бейзлайн
python plot_observation.py && python plot_results.py && python build_report.py
```
