# Comment
#FROM base/job/pytorch/1.8.0-cuda11.1-a100:20220509T151538446
#FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
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



#WORKDIR /app

# Install system libraries required by environment
RUN pip install torchvision==0.9.1 \
  && pip install thop==0.0.31-2005241907 \
  && pip install plotly==4.8.1   \
  && pip install tqdm==4.62.3


#CMD ["python3"]
