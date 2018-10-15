import cv2
import os

rawImageFolder = "first_batch"
exportFolder = "face_images"
minimumSizeAllowed = 336 # face images smaller than X by X will be discarded
scaleFactor = 1.2
offsetFactor = 1.0
cascPath = "haarcascade_frontalface_default.xml"

def get_face_box_with_offset(x, y, w, h, imgw, imgh, offset_factor=1):
    # expand the initial box to a square
    sx, sy, sw, sh = x, y, w, h
    if sw < sh:
        expand = sh-sw
        sx = sx - expand // 2
        sw = sh
    elif sh < sw:
        expand = sw-sh
        sy = sy - expand // 2
        sh = sw
    # try to offset the borders
    fx = sx - int(sw * offset_factor)
    fy = sy - int(sh * offset_factor)
    size = sw + 2 * int(sw * offset_factor)
    # check if the area still lies within the image
    if fx < 0 or fy < 0 or fx + size >= imgw or fy + size >= imgh:
        return False, 0, 0, 0, 0
    else:
        return True, fx, fy, size, size
#    fx = x - w * offset_factor
#    if fx < 0:
#        fx = 0
#    fy = y - h * offset_factor
#    if fy < 0:
#        fy = 0
#    fw = w + 2 * w * offset_factor
#    if fx + fw > imgw:
#        fw = imgw - fx
#    fh = h + 2 * h * offset_factor
#    if fy + fh > imgh:
#        fh = imgh - fy


# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

img_folders = os.listdir(rawImageFolder)
#print(img_folders)
for f_index, folder_name in enumerate(img_folders):
    print("Folder {} out of {}".format(f_index, len(img_folders)))
    curr_dir = os.path.join(rawImageFolder, folder_name)
    img_names = os.listdir(curr_dir)
    for im_index, img_name in enumerate(img_names):
        img_path = os.path.join(curr_dir, img_name)
        image = cv2.imread(img_path)
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=scaleFactor,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        #print("Found {0} faces!".format(len(faces)))
        # Save faces to files
        for face_index, (x, y, w, h) in enumerate(faces):
            
            
            face_name = "face_{}-{}-{}.jpg".format(f_index, im_index, face_index)
            #face_name = "face_" + f_index + "-" + im_index + "-" + face_index + ".jpg";
            face_path = os.path.join(exportFolder, face_name)
            box_exists, fx, fy, fw, fh = get_face_box_with_offset(x, y, w, h,
                                                                  image.shape[1],
                                                                  image.shape[0],
                                                                  offset_factor=offsetFactor)
            #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #cv2.rectangle(image, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
            #cv2.imshow("Faces found", image)
            #cv2.waitKey(0)
            if box_exists and fw >= minimumSizeAllowed:
                face_img = image[fy:fy+fh, fx:fx+fw]
                cv2.imwrite(face_path, face_img)
        



# Get user supplied values
#imagePath = "C:\\Users\\olexd\\Programming\\Projects\\antispoofing-cv\\data\\raw_images\\first_batch\\1xt_pa9hfnM\\frame1857.jpg"

# Read the image
#image = cv2.imread(imagePath)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Draw a rectangle around the faces
#for (x, y, w, h) in faces:
#    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#cv2.imshow("Faces found", image)
#cv2.waitKey(0)
