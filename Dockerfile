FROM python:3.10


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get update && apt-get install libgl1



WORKDIR /app


COPY requirements.txt .


RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu


RUN pip install --no-cache-dir -r requirements.txt


COPY . .


#EXPOSE 1883
#EXPOSE 5672


CMD [ "python3", "app.py"]
