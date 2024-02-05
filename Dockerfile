FROM python:3.8.10 AS base

RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install -y ca-certificates curl gnupg
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
ARG NODE_MAJOR=20
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list

RUN apt-get update -y
RUN apt-get install nodejs -y
RUN npm install -g pyright

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
