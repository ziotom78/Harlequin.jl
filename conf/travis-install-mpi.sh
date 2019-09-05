#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`
OMPIVER=openmpi-3.0.0
case "$os" in
    Darwin)
        brew update
        brew upgrade cmake
        brew install openmpi
        ;;

    Linux)
        sudo apt-get update -q
        sudo apt-get install -y gfortran ccache openmpi-bin libopenmpi-dev
        ;;
    
    *)
        echo "Unknown operating system: $os"
        exit 1
        ;;
esac
