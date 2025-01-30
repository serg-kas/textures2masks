## Подготовка окружения



## Установка приложения из гитхаба

1. Клонируем репозиторий приложения

git clone https://github.com/serg-kas/textures2masks

2. Переходим в папку приложения и запускаем установку зависимостей и выходим

cd textures2masks
pip install -r requirements.txt
cd ..

3. Клонируем репозиторий модели

git clone https://github.com/facebookresearch/sam2.git  

4. Переходим в папку модели и делаем локальную установку

cd sam2
pip install -e .

5. Возвращаемся в папку приложения

cd ..
cd textures2masks

6. Идем в папку models и скачиваем веса

сd models
./download_ckpts.sh 
cd ..

7. Корректность установки нужно проверить тестовым запуском

python app.py test

7. Запуск приложения
python app.py masks

В папку source_files помещаем файл для обработки (можно несколько)
В папке out_files появится результат


ПРИМЕЧАНИЕ:
1. Окружение удобнее ставить / активировать с помощью Anaconda
conda create -n py310 -c conda-forge python=3.10 pip
conda activate py310
2. Потом необходимые пакеты устанавливать с помощью pip

