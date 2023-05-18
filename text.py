
def read_arabic_image(img):
    ocr = PaddleOCR(lang="ar")
    result = ocr.ocr(img)
    # the image must be one dimention change this
    # Note: change this accourding to your image shape
    read, _ = result[0][0][1]
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

    LicencePlateNum =" ".join(stringsArr)

    split_string = re.findall('\D+|\d+', LicencePlateNum)

    characters = []
    numbers = []

    for item in split_string:
        if item.isdigit():
            numbers.append(item)
            # numbers.append(int(item))
        else:
            characters.append(item.strip())

    validtext = ' '.join(characters + [str(number) for number in numbers])

    return validtext


arabic_to_english = {
    '٠': '0',
    '١': '1',
    '٢': '2',
    '٣': '3',
    '٤': '4',
    '٥': '5',
    '٦': '6',
    '٧': '7',
    '٨': '8',
    '٩': '9'
}

def ArabicNum_to_En(arabic_to_english , PlateNumber):
    LicencePlateNumber = ''
    for Num in PlateNumber:
        if Num in arabic_to_english:
            LicencePlateNumber += arabic_to_english[Num]
        else:
            LicencePlateNumber += Num

    return LicencePlateNumber


def remove_space(PlateNumber):
    string = PlateNumber.replace(' ', '')
    LicencePlateNumber = ' '.join([char for char in string])
    return LicencePlateNumber


def data_dict(LicencePlateNumber,colortext):
    keys = ['Plate Number', 'Color']
    values = [LicencePlateNumber, colortext]
    data  = {key: value for key, value in zip(keys, values)}
    return data


def Response():
    PaddleOCR_result = read_arabic_image(textpart)
    stringsArr = extract_strings(PaddleOCR_result)
    ValidPlateNumber = ValidateText(stringsArr)
    PlateNumber = ArabicNum_to_En(arabic_to_english , ValidPlateNumber)
    LicencePlateNumber = remove_space(PlateNumber)
    data = data_dict(LicencePlateNumber,colortext)
    return data
