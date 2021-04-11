from pyvirtualdisplay import Display
import torch
import glob
import io
import os
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay


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


def display_start():
    display = Display(visible=0, size=(1400, 900))
    display.start()


def save_model(model, path):
    torch.save(model.state_dict(), path)

def create_directory(path):
    try:
        os.mkdir(path)
        print(f'Directory {path} has been created.')
    except FileExistsError:
        print(f'Directory {path} already exists.')