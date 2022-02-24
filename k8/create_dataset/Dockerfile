FROM osgeo/gdal:ubuntu-small-latest

# avoid questions when installing stuff in apt-get
ARG DEBIAN_FRONTEND=noninteractive

# installing GCloud SDK
RUN apt-get update
RUN apt-get install -y apt-transport-https ca-certificates gnupg curl wget
RUN apt-get install -y python3.8-dev python3-distutils
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

# assuming ubuntu 20 has python3 only
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN python -m pip install --upgrade pip
RUN pip install rasterio geopandas tqdm

# setup work directory
WORKDIR /home

COPY src /home/src

WORKDIR /home/src
RUN chmod a+rwx inference.sh
# default argument 'surrey'
#CMD ./inference.sh surrey 1250 1100
CMD echo "hello"