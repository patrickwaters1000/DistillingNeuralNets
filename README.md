# Distilling Neural Networks

Knowledge distillation is a technique for improving the performance of small lightweight models.  This repository uses distillation to train a small MobileNet image classifier.

## Requirements

Keras, tensorflow and numpy are required.

## Scraping from ImageNet

Using the script Downloader.py, you can scrape ImageNet for images of some desired classes, for example
```
python Downloader.py cat dog bear horse squirrel
```
This will download the images and save them in directories data/train/cat, etc. and data/test/cat, etc.
Many of the ImageNet URLs do not point to valid images, purge any invalid images as follows:
```
python ValidImageTester.py
python CountImages.py
```

## Testing distillation

You can run an experiment comparing the performance of knowledge distillation on the image classes you downloaded as follows:
```
python Experiment1.py
```
