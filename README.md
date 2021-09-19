# Classify Images

## classify_night_day.py
Classifies night and day images based on pixel values.

## cluster_images.py
There are two modes you can run the script: cluster or classify
1. In cluster mode, all the images are cluster into two categories (can change).
 The centroid values are automatically saved in centroids.json file.
2. In 'classify' mode, the centroid values are read from the json file and used to classify new images. 
 If the similarity measures (distance between two images) are too high, a warning message will be shown.


## convert_labels.py
Converts label files from one format to another.
Currently, supported formats are:

- CVAT xml to EdgeImpulse
- Kaggle xml to EdgeImpulse

Example:
```
python .\convert_labels.py -mode convert -path .\SkNetworks_CarDashboard_21036\01.rawData\2\BMW\BMW_day_0_1.xml -for
mat_in cvat_xml
```
