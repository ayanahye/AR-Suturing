### Description 
This is apart of a research project building a low cost, low latency AR suturing training system

### Dataset Credits
The data in original_frames and segmented_frames are taken from this video: https://youtu.be/eDG2d69e96I?si=8ttznzcg0x0QobRU

### Structure
- **01-segment.py** contains the script for segmenting the video frames into patient skin, doctor gloves, tools, and cut based on color by creating masks for each component and combining them into single colored result image.