## Installation from Docker Containers

OpenFPM provides [Docker](https://www.docker.com/) containers with OpenFPM installed. 

Docker is an open platform for developers to build, ship, and run distributed applications.
Please refer to user manuals to find more information about this tool, e.g. [here](http://people.irisa.fr/Anthony.Baire/docker-tutorial.pdf)

OpenFPM provides the following CPU-only docker images with OpenFPM 
pre-installed `openfpm/ubuntu:install20.04`, `openfpm/fedora:install34`.

Or the images with the GPU code enabled `openfpm/ubuntu_cuda:install10.2-devel-ubuntu18.04`,
`openfpm/ubuntu_cuda:install11.2.2-devel-ubuntu20.04`.   

To start the container in Linux general command would be
```sh
[sudo] docker run --net=host \
 -dit -v /tmp/.X11-unix:/tmp/.X11-unix \
 -e DISPLAY=unix$DISPLAY \
 -v $HOME/<shared host folder>:<shared container folder> \
 -e GDK_SCALE -e GDK_DPI_SCALE \ 
 --gpus all `#for gpu containers, requires nvidia-container-toolkit` \ 
 --name openfpm openfpm/ubuntu:install20.04 bash

```

- where the networking of the container is not containerized, but connected to the host network
- X11 forwarding enabled
- enabled folder sharing between the host and the container file systems
- all GPU resources are shared between the host and the container

To connect to the running container use

    [sudo] docker exec -it openfpm /bin/bash

Inside the container the preinstalled dependencies are located in `/root`, 
the file with environment variables _openfpm_vars_ is located is `/root`,
the source code of OpenFPM located in `/openfpm/openfpm_pdata`, 
installation files of OpenFPM are located in `/usr/local`.

---