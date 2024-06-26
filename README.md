# Face recognizer
Project to learn more about Computer Vision, specifically recognition and AI training. Recognizes faces from a given dataset.

## To do list
- Emotion recognizer (medium)
- Lip reading (hard)

## Usage
First step is to install all the depedencies from *requirements.txt* using:
```
pip install -r requirements.txt
```
Then, you should train the model on pictures you uploaded in the folder *training* using:
```
detector.py --train
```
Then, you can validate your model on all images in folder *validation* using:
```
detector.py --validate
```
Or a specific image using:
```
detector.py --image [path_image]
```
Finally, you can also use your model on videos:
```
detector.py --video [path_video]
```
Or on your webcam:
```
detector.py
```

The folder tree should look like this:
```
..
└── face_recognizer/
    ├── code/
    │   └── detector.py
    ├── training/
    │   ├── person0/
    │   │   ├── image_person00
    │   │   ├── image_person01
    │   │   └── ...
    │   ├── person1/
    │   │   ├── image_person10
    │   │   ├── image_person11
    │   │   └── ...
    │   ├── person2/
    │   │   ├── image_person20
    │   │   ├── image_person21
    │   │   └── ...
    │   └── ...
    ├── validation/
    │   ├── image0
    │   ├── image1
    │   └── ...
    ├── output/
    │   └── encodings.pickle
    └── ...  
```

```
usage: detector.py [-h] [--video VIDEO] [--image IMAGE] [--model MODEL] [--encodings ENCODINGS] [--train] [--validate]

optional arguments:
  -h, --help            show this help message and exit
  --video VIDEO         Path to the video file
  --image IMAGE         Path to the image file
  --model MODEL         Model to use for face detection (hog or cnn), default=hog
  --encodings ENCODINGS
                        Path to the encodings file (.pickle)
  --train               Train the model (from the folder 'training')
  --validate            Validate the model (from the folder 'validation')
```

*Note*: press 'q' to exit video or image and press 'space' to pause the video