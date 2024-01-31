# SHIFT train project
Этот проект - бейзлайн для обучения модели классификации, ваша задача - улучшить его.

## Окружение
Для начала склонируйте проект и настройте окружение
```bash
git clone https://github.com/Dragon181/SHIFT-intensive.git
cd SHIFT-intensive
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## Запуск проекта
Перед началом разбейте датасет на train, val, test для лучшего обучения.
Не забудьте поправить пути до новых датасетов в [конфигурации](conf/data/sign_train.yaml)
```bash
python3 train.py
```

## С чего начать?
Для начала просмотрите параметры конфигураций [configs](conf/).
Так же внимательно изучите, как формируется [dataloader](srcs/data_loader/data_loaders.py)
По умолчанию используется модель с полносвязными слоями, попробуйте что-то лучше

## Полезные ссылки
- Туториалы по обучению на Pytorch - https://pytorch.org/tutorials/
- Документация hydra - https://hydra.cc/docs/intro/