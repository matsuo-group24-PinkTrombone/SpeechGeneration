FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get install -y \
  ffmpeg \
  git \
  make \
  && rm -rf /var/lib/apt/lists/*

RUN conda config --add channels conda-forge
RUN conda install -y nodejs libsndfile poetry && conda upgrade -y nodejs
RUN poetry config virtualenvs.create false
CMD [ "/bin/bash" ]
