# tft_vod_scraper

source venv/bin/activate

model exists in model_output/tft_augment_detector11

generate training data with generate_training.py, set number of desired images there

run train_model.py to train new model.

run test_model.py to test on whatever frames are currently in temp/frames 

if there are none, run main.py - this should be working atm.

long story short: basically the model sucks lol