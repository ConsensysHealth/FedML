FROM continuumio/miniconda3

COPY . /


COPY venv.yml .
RUN conda env create -f venv.yml
SHELL ["conda", "run", "-n", "venv", "/bin/bash", "-c"]

ENV PYTHONPATH="$PYTHONPATH:/"

RUN conda install setproctitle
RUN conda install h5py 
RUN conda install pytorch torchvision
RUN conda install -c anaconda mpi4py
RUN conda install scikit-learn 
RUN conda install numpy 

# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "venv", "python", "./FedML/fedml_experiments/distributed/my_fedavg/main.py"]