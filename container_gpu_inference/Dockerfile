FROM gcr.io/cloud-ml-public/training/pytorch-gpu.1-6

RUN pip install rasterio tqdm

WORKDIR /home

COPY src /home/docker/src

WORKDIR /home/docker/src

RUN chmod a+rwx inference-gpu.sh
RUN chmod a+rwx tiff2jp2.sh

CMD echo"Hello World"
