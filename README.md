# Getting Started with OptimusAI / Clisham

Welcome to OptimusAI!

## Overview
Welcome. You are looking at the repository of OptimusAI. Optimus is a tool for process plant diagnosis and optimization written in python.

## On using Docker
We provide a common base Docker image that is used throughout the Optimus code base. The `Dockerfile` that builds this image lives at the base or our code repository.

The Optimus QB Labs team uses this Docker image for our own CI/CD and feature development. So if you are comfortable with containerized development (or willing to invest some time to learn an awesome new tool), you get the benefit of the same consistency, reproducibility, and rapid startup that repo maintainers get. This will also make it much easier for you to hand over your project or run it in a different environment.

We try to abstract away most of the docker commands in a Makefile. To see all available commands, simply run
```
make
```
from your project's root.


## Getting started (Unix / Mac)

### Requirements
While the optimus pipeline should run natively on Mac (ideally in a conda environment), we highly recommend
using the above-described Docker setup.

At a minimum, running Optimus with docker requires the following tools:
- [git](https://git-scm.com/)
- [docker](https://www.docker.com/products/docker-desktop)
- [gnu make](https://www.gnu.org/software/make/) and [rsync](https://rsync.samba.org/) which can be installed via [homebrew](https://brew.sh/)


-----------

## Getting started (Windows)

### Requirements
Windows development is only supported in combination with Docker. You will still be able to use the windows development tools you are familiar with and we have tried to abstract most of the docker commands away behind a user-friendly make file.

At a minimum, running Optimus with docker requires the following tools:
- [git](https://git-scm.com/) or [github desktop](https://desktop.github.com/)
- [docker](https://www.docker.com/products/docker-desktop)
- a Windows-compatible version of `make` (more information below)

**Clone your folder into a local file**:
You will need to have a local copy of your Azure DevOps folder to run the following commands. Cloning information is available in the **clone** button.

- If you are using github:
Connect using the option for connecting through the web. You will require three inputs: a username, a password and a link. To get this information, click on the **clone** button on azure devops. Copy the HTTPS link provided there as the link; for the username and passwords, click on **generate git credentials**.

You can also use your preferred program for cloning the repository. 

**Docker**: 

- [Hyper-V](https://docs.microsoft.com/en-us/virtualization/hyper-v-on-windows/quick-start/enable-hyper-v) is required to run Docker Desktop for Windows and you need admin privledges to install it. GHD will be able to help with this.
- If you have Windows 10 Pro, Enterprise or Education, build 15063 or later then you want to install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- If you have Windows 7 64-bit to Windows 10 (< Build 15063) then we recommend you explore the _alternative windows native install_ that we describe in this `README.md`.
- We recommend you avoid [Docker Toolbox](https://docs.docker.com/toolbox/toolbox_install_windows/) on older firm laptops. Although this is theoretically available for install, the in practice complexity of mounting source code from the host machine to the container through a linux guest VM negates the simplifying benefits of using Docker in the first place.


**Make**: 

On Windows you have a few different options for running tasks in a GNU compatible `Makefile`:

- GNU make is available on [chocolatey](https://chocolatey.org/install) with `choco install make`
- [GNUWin32](http://gnuwin32.sourceforge.net/packages/make.html)
- [MinGW](http://www.mingw.org/)

You can find other options on the web.

One option that we have found to work well is to install `make` with Chocolatey:

- Chocolatey supports a [Non Adminstrative Install](https://chocolatey.org/docs/installation#non-administrative-install) so if you don't have administrative privilege on your machine you can still install the package manager without additional requests.
- [make](https://chocolatey.org/packages/make) is a well supported Chocolatey package that provides a clean conventional install on your windows host laptop. You will need to install it. 


### Installation process

**1) Clone your folder into a local file**:

You will need to have a local copy of your Azure DevOps folder to run the following commands. You can use your preferred method and program.

- If you have Github Desktop:
On Github Desktop, select clone repository -> URL. You will require three inputs: a username, a password and a link. To get the corresponding information, go to azure devops online repository and select the **clone** button. Copy the HTTPS link provided there as the link; for the username and passwords, click on **generate git credentials** and copy the corresponding data.



**2) Build the Optimus Base Image**:

Open Window´s power shell and move into the folder in which you placed the clone repository. 
 
```
cd <YOUR FILE LOCATION ABSOLUTE PATH>
make base-build
```

**3) Conda Dependency Setup**

We need to install all of the required python dependencies into the correct docker volume so they are persisted and avalilable to the container at run time.

We have a make task to help with this.

```
make dependencies-install
```

If you observe a docker connection error during the process (error 125), you will have to manually assign your local repository to Docker. To do this, open Docker Desktop and go to settings -> Resources -> file sharing, and assign your local path.
If you would like more information of what's happening here, take a look at the sections describing both poetry and docker volumes.


After this step, optimus is installed.


**4) Checking**

To check whether the installation process worked, get example data from your team and then run the following command:

```
make pipeline-run-name PIPE_NAME=de
```


### Alternative Windows Dev Setup

If you have a older Windows machine that does not support Docker, or you are encountering unforeseen obstacles getting the container based development environment installed on your machine, you can attempt to run the Optimus dev tools directly on your native windows machine. This is not a recommended or well-supported by the development team so consider alternative solutions before you attempt direct install on you Windows laptop. If you do decide to go that route the steps below may be useful:

#### Anaconda Setup

* Install Anaconda locally. Find [detailed instructions here](https://www.anaconda.com/)
* Run the following two commands to create a new conda environement with `python >=3.7`:

```
conda create -n "optimus" python=3.7
activate optimus
```

To exit this created conda environment, just run `deactivate`

#### Dependency Install with Poetry

Follow these [installation instructions](https://python-poetry.org/docs/) for the Windows platform and run the powershell command to install poetry.

Once the installation is completed, run the following commands to install all requirements inside the conda environment:

```
"C:\Users\[YOUR LOCAL USER]\.poetry\bin\poetry" config virtualenvs.create false
"C:\Users\[YOUR LOCAL USER]\.poetry\bin\poetry" install -n --no-root
```

## Docker Volumes

When you run docker-based commands from the `Makefile`, they run inside a new linux container spawned from the `optimus-base` image. This means that you get a consistent execution environment. However, there are times when you may want to persist state between different runs. In particular, you may want to persist updates to the python env, for example when installing a new python library. To enable this, we use [Docker volumes](https://docs.docker.com/storage/volumes/).

Development and troubleshooting with our Docker-based workflow will be easier if you know more about how volumes work.

Here is an example command from our `Makefile`:

```
docker run -t --workdir="/optimus/pipeline" -v miniconda:/miniconda -v ${PWD}:/optimus optimus-base-clisham /bin/bash -ci "kedro run"
```

Let's break down what's happening in this command:

* We are spawning a new Linux container from the docker image `optimus-base-clisham` to do a `kedro run` of the Optimus pipeline.
* The `kedro run` command is issued from the `/optimus/pipeline` directory within the container
* The `-v ${PWD}:/optimus` argument uses a _docker path based volume_  to `bind-mount` the source files of your host machine present working directory into the container's file system at the location `/optimus`
* The `-v miniconda:/miniconda` argument uses a _docker named volume_ to mount the volume `miniconda` into the container file system location `/miniconda`
  - you can think of this _named volume_ like a little hard drive that holds your entire `conda env`
  - this little hard drive is completely managed by docker and gets attached to the `/miniconda` location at container run time
  - when you kill the container this volume persists and is ready to be reattached next time you need it

## Contributing
Optimus is an "internal open source" project and we would love your contributions. Please see [our contribution guidelines](CONTRIBUTING.md) for code styles and standards.
