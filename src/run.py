from keras.models import load_model
from modely import *
model = load_model("model_512_awe_3.hdf5", custom_objects={"awesomeq_loss": awesomeq_loss, 'dice_loss': dice_loss})
from modely import *
camera = cv2.VideoCapture(0)
img_back = cv2.imread("background.jpg")
img_back = padding(resize_image(img_back, 512), 512, 3)
img_zeros = np.zeros((512, 512, 3)).astype(np.uint8)
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (512,512))
while True:
    # Grab the current frame
    (grabbed, img) = camera.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = padding(resize_image(img, 512), 512, 3)
    img_back = padding(resize_image(img_back, 512), 512, 3)
    img = img.astype(np.float64)
    img /= 255.0
    imgx = np.expand_dims(img, axis=0)
    prediction = model.predict(imgx)[0]
    prediction *= 255
    prediction = np.squeeze(prediction, axis=[0, -1])
    prediction[prediction < 75] = 0
    prediction[prediction != 0] = 255
    img *= 255
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_zeros[prediction == 255] = img[prediction == 255]
    img_zeros[prediction != 255] = img_back[prediction != 255]
    cv2.imshow("Face", img_zeros)
    # out.write(img_zeros)
    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup the camera and close any open windows
# out.release()
camera.release()
cv2.destroyAllWindows()
