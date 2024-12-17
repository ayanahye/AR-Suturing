### Description 
This is apart of a research project building a low cost, low latency AR suturing training system

### Dataset Credits
The data in original_frames and segmented_frames are taken from this video: https://youtu.be/eDG2d69e96I?si=8ttznzcg0x0QobRU

The data from the Cut folder is taken from: https://www.kaggle.com/datasets/ibrahimfateen/wound-classification

The data from the train_images_sample and train_masks_sample is taken from: https://www.kaggle.com/datasets/leoscode/wound-segmentation-images 

### Structure
- **01-segment.py** contains the script for segmenting the video frames into patient skin, doctor gloves, tools, and cut based on color by creating masks for each component and combining them into single colored result image.
- **02-cut-detection.py** contains script for using wound detector pretrained model via api and saves results on 4 images in detected_Cut with exact positions saved to inference_result.json.
- **test_on_video.py** contains script for testing cut detector on live video surgery
- **Full_U_Net_Cuts.ipynb** contains the python notebook for UNET model trained on wounds.
- **Edge_Tracing.py** contains python script to generate green edge around cut on actual image using mask.