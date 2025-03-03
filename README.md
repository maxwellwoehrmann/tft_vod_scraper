# tft_vod_scraper
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Git LFS to store a large model file. To clone this repository:
1. Install Git LFS:
brew install git-lfs

clone as normal.

here is how to try everything locally:

start a virtual environment
install requirements (if you later find any are missing pls add to requirements.txt)

from the root folder
run python main.py 
    this is currently set to just download and assign frames to players from the 10 most recent vods on the list
    importantly, it saves the frames (in temp/frames), which we can then run detection on

    you can easily edit it to run on only 1 game if preferred

run detect_and_label_augments.py
    this creates an output folder with labeled detections for all frames im temp/frames. saves the output into augment_results

that's it :D


if you want to train the models yourself:
from root directory run
    python generate_box_training.py
    python train_box_model.py

    python generate_augment_training.py
    python train_augment_model.py

they each have their own testing script you can run as well. just double check that all the paths look good.