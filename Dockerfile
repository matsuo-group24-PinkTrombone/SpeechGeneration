FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ADD ./ /workspace

RUN apt-get update && apt-get install -y \
  curl \
  ffmpeg \
  git \
  make \
  unzip \
  && rm -rf /var/lib/apt/lists/*

COPY ./scripts/docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN conda config --add channels conda-forge && \
  git config --global --add safe.directory /workspace

RUN conda install -y nodejs libsndfile && \
  conda upgrade -y nodejs && \
  pip install poetry && \
  poetry config virtualenvs.create false && \
  poetry install

CMD [ "/bin/bash" ]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
