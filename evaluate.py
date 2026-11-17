"""Запуск замеров: python evaluate.py --dataset eval/dataset

В PyCharm: правая кнопка на этом файле, Run; параметры (например --model base) вписываются
в Run, Edit Configurations, поле Parameters. Рабочая папка запуска должна быть корнем проекта.
"""
from eval.run_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
