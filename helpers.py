import base64
import glob
import io
import os

import torch
from IPython import display as ipythondisplay
from IPython.display import HTML
from pyvirtualdisplay import Display


def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


def print_hyperparams(config):
    print('----------------------------------------')
    print("{:<15} {:<10}".format('Param', 'Value'))
    for label, value in config.items():
        if label is not "device":
            print("{:<15} {:<10}".format(label, value))
    print('----------------------------------------')


def display_start():
    display = Display(visible=0, size=(1400, 900))
    display.start()
    return display


def save_model(model, path):
    torch.save(model.state_dict(), path)


def create_directory(path):
    try:
        os.mkdir(path)
        print(f'Directory {path} has been created.')
    except FileExistsError:
        print(f'Directory {path} already exists.')