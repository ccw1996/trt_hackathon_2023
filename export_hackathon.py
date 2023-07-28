from canny2image_TRT import hackathon
import cv2
import datetime

model = hackathon()
model.initialize()


for i in range(5):
    path = "/home/player/pictures_croped/bird_"+ str(i) + ".jpg"
    img = cv2.imread(path)
    start = datetime.datetime.now().timestamp()
    new_img = model.process(img,
            "a bird", 
            "best quality, extremely detailed", 
            "longbody, lowres, bad anatomy, bad hands, missing fingers", 
            1, 
            256, 
            20,
            False, 
            1, 
            9, 
            2946901, 
            0.0, 
            100, 
            200)
    
    cv2.imwrite(f'out_imgs/bird_CTRL&UNET&DECODER_{i}.jpg', new_img[0])
    