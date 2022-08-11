ARG BASE_IMAGE
ARG VALIDATOR_IMAGE

FROM $BASE_IMAGE as base
FROM $VALIDATOR_IMAGE as validator

FROM base

# install software needed for the workload
# this example is installing figlet
RUN apt-get update && \
    apt-get install --no-install-recommends --no-install-suggests -yq \
        figlet && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean && \
    figlet Singularity

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

RUN pip install  git+https://github.com/openai/CLIP.git \
    && python CLIP/setup.py develop

# get the validation scripts
COPY --from=validator /validations /opt/microsoft/_singularity/validations/

# run the validation
RUN /opt/microsoft/_singularity/validations/validator.sh
