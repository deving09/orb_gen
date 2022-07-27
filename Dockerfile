#COMMENT
FROM ubuntu:18.04

RUN apt-get update -y && \
    apt-get install -y python3.7 python3-pip python3.7-dev pkg-config
RUN apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
RUN apt-get install -y software-properties-common &&\
    add-apt-repository -y ppa:jonathonf/ffmpeg-4

RUN apt-get install -y \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev

RUN apt-get install git -y

RUN pip3 install --upgrade pip

# Install system libraries required by environment
RUN pip3 install numpy \
    && pip3 install pillow \
    && pip3 install beautifulsoup4==4.10.0 \
    && pip3 install click==8.0.4 \
    && pip3 install gdown==4.4.0 \
    && pip3 install matplotlib==3.3.4 \
    && pip3 install pyparsing \
    && pip3 install pysocks \
    && pip3 install regex \
    && pip3 install requests \
    && pip3 install six \
    && pip3 install tokenizers==0.11.6 \
    && pip3 install torch==1.8.0 \
    && pip3 install torchvision==0.9.1 \
    && pip install thop==0.0.31-2005241907 \
    && pip install plotly==4.8.1   \
    && pip install tqdm==4.62.3 \ 
    && pip install ftfy regex \
    && pip install git+https://github.com/openai/CLIP.git 


#CMD ["python3"]
