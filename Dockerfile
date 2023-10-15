FROM python:3.8.10 AS base

RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Clean APT
RUN rm -rf /var/lib/apt/lists/*; rm -rf /var/cache/apt/archives/*

WORKDIR /app


COPY requirements.txt ./
RUN python -m pip install -r requirements.txt


COPY . ./


EXPOSE 8001

FROM base AS development

RUN echo "alias p='python'" >> ~/.bashrc
RUN echo "alias dm='python manage.py makemigrations --settings=cocktail.settings.local && python manage.py migrate --settings=cocktail.settings.local'" >> ~/.bashrc
RUN echo "alias dr='python manage.py runserver 0.0.0.0:8001 --settings=cocktail.settings.local'" >> ~/.bashrc
