## Подготовка окружения



## Установка приложения из гитхаба

1. Клонируем репозиторий приложения

git clone https://github.com/serg-kas/textures2masks

2. Переходим в папку приложения и запускаем установку зависимостей

cd textures2masks
pip install -r requirements.txt

3. Находясь в этой же папке клонируем модель

git clone https://github.com/facebookresearch/sam2.git  

4. Переходим в папку модели и делаем локальную установку

cd sam2
pip install -e .

5. Скачиваем веса (идем в папку checkpoints, скачиваем и возвращаемся обратно)

сd checkpoints 
./download_ckpts.sh 
cd ..

6. Возвращаемся в папку приложения

cd ..

## Запуск приложения

