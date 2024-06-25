from pathlib import Path
import face_recognition
import pickle
import collections
from PIL import Image, ImageDraw, ImageFont

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

    face_locations = face_recognition.face_locations(input_img, model=model)
    face_encodings = face_recognition.face_encodings(input_img, face_locations)

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

def _recognize_face(unknown_encoding, loaded_encodings):
    known_encodings = loaded_encodings["encodings"]
    names = loaded_encodings["names"]
    votes = collections.defaultdict(int)
    total_votes = 0
    for known_encoding, name in zip(known_encodings, names):
        # Compares the face encoding of the unknown face with the known faces
        if face_recognition.compare_faces([known_encoding], unknown_encoding)[0]:
            votes[name] += 1
            total_votes += 1

    if votes:
        print(f"Most votes: {max(votes, key=votes.get)} with {votes[max(votes, key=votes.get)]} votes")
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
    if probability == -1:
        return
    
    # Draw a rectange to put the probability in
    rect_length = max(0.1*(right-left), 20)
    draw.rectangle(((right-rect_length, bottom), (right, text_bottom)), fill=BOUNDING_BOX_COLOR, outline=BOUNDING_BOX_COLOR)
    draw.text((right-rect_length, bottom), f"{probability:.2f}", fill=TEXT_COLOR, font=font, align="right")

if __name__ == "__main__":
    # encode_known_faces()
    # recognize_faces("validation/elon_musk/161856.jpg")
    # recognize_faces("validation/elon_musk/161850.jpeg")
    # recognize_faces("validation/unknown/ambroise.jpg")
    validate()