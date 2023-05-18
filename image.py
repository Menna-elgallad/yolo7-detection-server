from algorithm.object_detector import YOLOv7
from utils.detections import draw
import json
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
import cv2
import os
import re
from paddleocr import PaddleOCR, draw_ocr
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)
from matplotlib import pyplot as plt
from flask import Flask, request, jsonify
from algorithm.object_detector import YOLOv7
from utils.detections import draw
import numpy as np
import cv2
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


yolov7 = YOLOv7()
yolov7.load('best.weights', classes='classes.yaml',
            device='cpu')  # use 'gpu' for CUDA GPU inferenceprint

print("model loaded")


def detection(image):
    # image = cv2.imread(imgpath)
    detections = yolov7.detect(image)
    detected_image = draw(image, detections)
    cv2.imshow("detected", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('detected_image.jpg', detected_image)
    print(json.dumps(detections, indent=4))
    json_data = json.dumps(detections)
    parsed_dict = json.loads(json_data)[0]
    x, y, width, height = parsed_dict["x"], parsed_dict["y"], parsed_dict["width"], parsed_dict["height"]

    cropped_image = image[y:y+height, x:x+width]

    return cropped_image


def cropping(plate):
    res = cv2.resize(plate, dsize=(270, 200), interpolation=cv2.INTER_CUBIC)
    colorpart = res[0:80, 0:270]
    crop = res[80:200, 0:270]
    return colorpart, crop


def color_extraction(img):
    height, width = img.shape[:2]
    hc = height / 2
    wc = width/2
    color = img[np.int16(hc)][np.int16(wc)]
    red, green, blue = int(color[2]), int(color[1]), int(color[0])

    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    bars = []
    bars.append(bar)
    img_bar = np.hstack(bars)
    return (red, green, blue)


def convert_rgb_to_names(rgb_tuple):
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    rred = ["maroon", "darkred", "firebrick", "brown", "crimson", "red"]
    yellow = ["gold", "goldenrod", "khaki", "yellow", "sandybrown", "peru"]
    ggreen = ["yellowgreen", "olive", "darkkhaki", "olivedrab", "lawngreen", "chartreuse", "darkgreen",
              "green", "forestgreen", "lime", "limegreen", "lightgreen", "palegreen", "springgreen", "seagreen", "mediumseagreen"]
    bblue = ["mediumaquamarine", "lightseagreen", "teal", "darkcyan", "midnightblue", "aqua", "cyan", "darkturquoise", "turquoise", "mediumturquoise", "aquamarine", "cadetblue",
             "steelblue", "dodgerblue", "deepskyblue", "cornflowerblue", "skyblue", "lightskyblue", "navy", "darkblue", "mediumblue", "mediumblue", "royalblue", ]
    orange = ["orange", "darkorange", "orangered", "coral",
              "tomato", "chocolate", "saddle brown", "sienna"]
    gray = ["dimgray", "gray",  "darkolivegreen", "darkgray", "silver", "lightgray", "gainsboro", "dimgrey",
            "grey", "darkgrey", "lightgrey", "lavender", "powderblue", "paleturquoise", "darkseagreen"]
    col = names[index]
    print("col", col)
    if col in rred:
        col = "red"
    elif col in yellow:
        col = "yellow"
    elif col in ggreen:
        col = "green"
    elif col in bblue:
        col = "blue"
    elif col in gray:
        col = "gray"
    elif col in orange:
        col = "orange"
    else:
        col = "not found"
    return col


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal


def remove_noise(image):
    return cv2.medianBlur(image, 5)

# thresholding


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# dilation


def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# erosion


def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# opening - erosion followed by dilation


def openn(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection


def canny(image):
    return cv2.Canny(image, 100, 200)

# skew correction


def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# template matching


def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def remove_borders(image):
    contours, heiarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)


def read_arabic_image(img):
    ocr = PaddleOCR(lang="ar")
    result = ocr.ocr(img)
    # the image must be one dimention change this
    # Note: change this accourding to your image shape

    return result


def extract_strings(PaddleOCR_result):
    strings = []
    for element in PaddleOCR_result:
        if isinstance(element, str):
            strings.append(element)
        elif isinstance(element, list):
            strings += extract_strings(element)
        elif isinstance(element, tuple):
            strings += extract_strings(list(element))

    return strings


# number then character
def ValidateText(stringsArr):

    LicencePlateNum = " ".join(stringsArr)

    split_string = re.findall('\D+|\d+', LicencePlateNum)

    characters = []
    numbers = []

    for item in split_string:
        if item.isdigit():
            numbers.append(int(item))
            # numbers.append(int(item))
        else:
            characters.append(item.strip())

    validtext = ' '.join(characters + [str(number) for number in numbers])

    return validtext


def remove_space(PlateNumber):
    string = PlateNumber.replace(' ', '')
    LicencePlateNumber = ' '.join([char for char in string])
    return LicencePlateNumber


def data_dict(LicencePlateNumber, colortext):
    keys = ['Plate Number', 'Color']
    values = [LicencePlateNumber, colortext]
    data = {key: value for key, value in zip(keys, values)}
    return data


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    file.save(file.filename)
    img = cv2.imread(file.filename)
    cropped = detection(img)
    colorpart, textpart = cropping(cropped)
    color = color_extraction(colorpart)
    colortext = convert_rgb_to_names((color))

    img = cv2.cvtColor(textpart, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.medianBlur(img, 1)  # Median blur to remove noise
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.medianBlur(img, 1)
    img = noise_removal(img)
    img = remove_borders(img)
    PaddleOCR_result = read_arabic_image(textpart)
    stringsArr = extract_strings(PaddleOCR_result)
    ValidPlateNumber = ValidateText(stringsArr)

    LicencePlateNumber = remove_space(ValidPlateNumber)
    data = data_dict(LicencePlateNumber, colortext)
    print("dataaa", data)

    graphql_mutation = """
    mutation Validation($gateid: String!, $plateNumber: String!) {
      validateCar(gateId: $gateid, plateNumber: $plateNumber) {
        access
      }
    }
    """
    # Set the GraphQL endpoint URL
    graphql_url = 'http://localhost:5000/graphql'
    # Set the headers with the necessary content type
    headers = {
        'Content-Type': 'application/json',
    }
    # Set the variables for the mutation
    variables = {
        'gateid': '2a2eb6e7-6d35-4c8e-9788-2d6696c477ac',
        'plateNumber': data['Plate Number'],
    }
    # Send the GraphQL request
    response = requests.post(graphql_url, json={
                             'query': graphql_mutation, 'variables': variables}, headers=headers)

    # Check the response status code
    if response.status_code == 200:
        # If the request was successful, return the JSON response
        return jsonify(response.json())
    else:
        # If there was an error, return an error message
        return 'GraphQL request failed', response.status_code


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8000", debug=True)
