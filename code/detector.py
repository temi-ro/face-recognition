from pathlib import Path
import face_recognition
import pickle
import collections
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from sklearn.neighbors import KDTree
import time

DEFAULT_ENCODINGS_FILE = Path("output/encodings.pickle")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

def encode_known_faces(model = "hog", encodings_location = DEFAULT_ENCODINGS_FILE):
    """
    Loads the images from the training folder, finds and encodes the faces in the DEFAULT_ENCODINGS_FILE file.
    """

    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        # Gives the coordinates of the face in the image
        face_locations = face_recognition.face_locations(image, model=model)
        # Gives the encoding of the face in the image
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    # name_encodings = dict(zip(names, encodings))
    name_encodings = {"names": names, "encodings": encodings}
    with open(encodings_location, "wb") as f:
        pickle.dump(name_encodings, f)

def recognize_faces(image_path, model = "hog", encodings_location = DEFAULT_ENCODINGS_FILE):
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_img = face_recognition.load_image_file(image_path)

    face_locations = np.array(face_recognition.face_locations(input_img, model=model))
    face_encodings = np.array(face_recognition.face_encodings(input_img, face_locations))

    pillow_img = Image.fromarray(input_img)
    draw = ImageDraw.Draw(pillow_img)

    # Returns the name and bounding box of all the faces in the image
    for bb, unknown_encoding in zip(face_locations, face_encodings):
        name, probability = _recognize_face(unknown_encoding, loaded_encodings)
        
        print(f"Found {name} in the image at {bb}")
        _display_faces(draw, bb, name, probability)

    del draw
    pillow_img.show()

def validate(model = "hog"):
    for filepath in Path("validation").glob("*/*"):
        if filepath.is_dir():
            continue

        print(f"Validating {filepath}")
        recognize_faces(str(filepath), model)

def recognize_faces_video(video_path=None, model = "hog", encodings_location = DEFAULT_ENCODINGS_FILE):
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # If video_path is None, it will use the webcam
    flipped = False
    if video_path is None:
        video_path = 0
        flipped = True

    cap = cv2.VideoCapture(video_path)
    pause = False
    start = time.time()
    idx = 0
    old_name, old_probability = "Unknown", -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if pause:
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == 32: # Space bar
                pause = False


        # Time
        print(f"Frame time: {time.time() - start}")

        if flipped:
            frame = cv2.flip(frame, 1)

        start = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame, model=model)
        face_encodings = face_recognition.face_encodings(frame, face_locations)


        pillow_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pillow_img)
        
        for bb, unknown_encoding in zip(face_locations, face_encodings):
            # Only recognize the face every 10 frames
            if idx % 10 == 0:
                name, probability = _recognize_face(unknown_encoding, loaded_encodings)
                old_name, old_probability = name, probability

            _display_faces(draw, bb, old_name, old_probability)

        del draw
        frame = cv2.cvtColor(np.array(pillow_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == 32: # Space bar
            pause = not pause

        idx += 1

    cap.release()
    cv2.destroyAllWindows()


def _recognize_face(unknown_encoding, loaded_encodings):
    known_encodings = loaded_encodings["encodings"]
    names = loaded_encodings["names"]
    votes = collections.defaultdict(int)
    total_votes = 0
    # Compares the face encoding of the unknown face with the known faces

    # KNN 
    tree = KDTree(known_encodings)
    distances, indices = tree.query([unknown_encoding], k=10)
    print(f"Distances: {distances}")
    for i, index in enumerate(indices[0]):
        if distances[0][i] > 0.6:
            votes["Unknown"] += 1
            total_votes += 1
        else:
            votes[names[index]] += 1
            total_votes += 1
        
    # Vote
    # for known_encoding, name in zip(known_encodings, names):
    #     # Compares the face encoding of the unknown face with the known faces
    #     if face_recognition.compare_faces([known_encoding], unknown_encoding)[0]:
    #         votes[name] += 1
    #         total_votes += 1



    if votes and total_votes > 2:
        print(f"Most votes: {max(votes, key=votes.get)} with {votes[max(votes, key=votes.get)]} votes out of {total_votes}")
        return max(votes, key=votes.get), votes[max(votes, key=votes.get)] / total_votes

    # Unknown face (should return None?)
    return "Unknown", -1

def _display_faces(draw, bounding_box, name, probability):
    ratio = 0.05

    top, right, bottom, left = bounding_box

    size_font = int((right-left) * ratio)
    size_font = max(size_font, 10) # Minimum font size such that it is still readable for small images
    font = ImageFont.truetype("arial.ttf", size_font)

    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name, font=font)
    
    # Draw a rectangle to put the text in
    draw.rectangle(((left, bottom), (text_right, text_bottom)), fill=BOUNDING_BOX_COLOR, outline=BOUNDING_BOX_COLOR)
    draw.text((left, bottom), name, fill=TEXT_COLOR, font=font)

    # In case of unknown face, don't display the probability
    if name == "Unknown":
        return
    
    # Draw a rectange to put the probability in
    rect_length = max(0.1*(right-left), 20)
    draw.rectangle(((right-rect_length, bottom), (right, text_bottom)), fill=BOUNDING_BOX_COLOR, outline=BOUNDING_BOX_COLOR)
    draw.text((right-rect_length, bottom), f"{probability:.2f}", fill=TEXT_COLOR, font=font, align="right")


if __name__ == "__main__":
    # encode_known_faces()
    # recognize_faces("validation/elon_musk/161856.jpg")
    # recognize_faces("validation/temi/20230920_102357976_iOS.jpg") 
    # recognize_faces("training/temi/20240121_173037745_iOS.jpg") # Funny case (emoji as temi)
    # recognize_faces("validation/temi/IMG-20200706-WA0048.jpg")
    # recognize_faces("validation/elon_musk/161889.jpg")
    # recognize_faces("validation/unknown/ambroise.jpg")
    # validate()
    recognize_faces_video()
    