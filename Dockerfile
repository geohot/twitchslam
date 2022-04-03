#FROM python
#FROM ubuntu:20.04
FROM nvidia/vulkan:1.1.121
RUN mkdir /app
WORKDIR /app
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y git 
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN /usr/bin/python3 -m pip install opencv-python
RUN /usr/bin/python3 -m pip install pygame
RUN /usr/bin/python3 -m pip install PyOpenGL
RUN /usr/bin/python3 -m pip install scikit-build

# Install CMake
#RUN apt install -y libprotobuf-dev protobuf-compiler
RUN apt-get install -y cmake
RUN apt-get install -y g++ 
RUN apt-get install -y freeglut3
RUN apt-get install -y freeglut3-dev
RUN apt-get install -y libglew-dev
RUN apt-get install -y binutils-gold
RUN apt-get install -y libsuitesparse-dev
RUN /usr/bin/python3 -m pip install scipy
RUN /usr/bin/python3 -m pip install scikit-image

# Install Eigen
WORKDIR /app
RUN git clone https://gitlab.com/libeigen/eigen.git
WORKDIR /app/eigen
RUN git checkout 3.3.4
RUN mkdir build
WORKDIR /app/eigen/build
RUN /usr/bin/cmake ..
RUN /usr/bin/make -j12
RUN make install

# Install pangolin
WORKDIR /app
#RUN git clone https://github.com/uoip/pangolin.git
RUN git clone  https://github.com/AdityaNG/pangolin.git
WORKDIR /app/pangolin
RUN mkdir build
WORKDIR /app/pangolin/build
RUN /usr/bin/cmake ..
RUN /usr/bin/make -j12
#RUN /usr/bin/make 
WORKDIR /app/pangolin/
RUN ls  pangolin*.so
RUN /usr/bin/python3 setup.py install

# Install g2opy
WORKDIR /app
RUN git clone https://github.com/uoip/g2opy.git
WORKDIR /app/g2opy
RUN mkdir build
WORKDIR /app/g2opy/build
RUN /usr/bin/cmake ..
RUN /usr/bin/make -j12
RUN /usr/bin/make install
WORKDIR /app/g2opy/
RUN /usr/bin/python3 setup.py install

# Install twitchslam
WORKDIR /app
#RUN git clone https://github.com/geohot/twitchslam
RUN git clone https://github.com/AdityaNG/twitchslam
WORKDIR /app/twitchslam
RUN git checkout docker

RUN touch /root/.Xauthority
RUN apt-get -y install xauth

EXPOSE 8887
WORKDIR /app/twitchslam
#CMD /usr/bin/python3 slam.py
CMD sh run.sh