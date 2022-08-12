# Comment
#FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
#FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
FROM singularitybase.azurecr.io/base/job/pytorch/rocm4.5.2_ubuntu18.04_py3.8_pytorch_1.8.1:20220509T151538593
#USER root

#RUN  apt-get update \
#  && apt-get install wget unzip zip -y
#RUN apt install wget
#RUN wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
#RUN dpkg -i packages-microsoft-prod.deb
#RUN apt-get -y update
#RUN apt-get -y install blobfuse libcurl3-gnutls


# Create a non-root user and switch to it.
#USER user

# All users can use /home/user as their home directory.
#ENV HOME=/home/user
#RUN chmod 777 /home/user

#RUN sudo apt update
#RUN apt install git -y

#WORKDIR /app

RUN pip install numpy \
    && pip install pillow \
    && pip install beautifulsoup4==4.10.0 \
    && pip install regex \
    && pip install six \
    && pip install ftfy \
    && pip install requests 

# Install system libraries required by environment
RUN pip install torchvision==0.9.1 \
  && pip install thop==0.0.31-2005241907 \
  && pip install plotly==4.8.1   \
  && pip install tqdm==4.62.3

RUN pip install  git+https://github.com/openai/CLIP.git 


#CMD ["python3"]
