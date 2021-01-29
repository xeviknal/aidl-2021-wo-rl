#/bin/sh

sudo apt-get install wget
sudo apt-get install python-opengl
sudo apt-get install -y xvfb python-opengl ffmpeg
sudo apt-get install xserver-xorg-core
sudo apt-get install tigervnc-standalone-server tigervnc-xorg-extension tigervnc-viewer
sudo apt-get install htop

## Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

cd idl-2021-wo-rl

conda create --name car-racing python=3.8
conda activate car-racing

pip install -r requirements
