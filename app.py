from tkinter import filedialog,Tk,Button,Canvas,LEFT
from PIL import ImageTk,Image  
from keras.models import load_model
import cv2
import numpy as np
import math

model = load_model('model.h5')
print("Model loaded...")



window=Tk()
window.geometry("500x500")

# global params
filename = ""
img = None
image = None
rgb = None
rgb_image = None
rgb_img = None

canvas = Canvas(window, width = 300, height = 300)  
canvas.pack(expand='yes')  



# Methods

def normalize(imgs):
    # convert image from range 0-256 to 
    imgs = imgs/255
    return imgs

def unnormalize(imgs):
    imgs = (imgs*255)
    return imgs.astype('uint8')

def get_rgb_image(l, ab):
    shape = (l.shape[0],l.shape[1],3)
    img = np.zeros(shape)
    img[:,:,0] = l[:,:,0]
    img[:,:,1:]= ab
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img


def findCloseProps(image_shape):
    mult = 224
    #fct = math.ceil(max(image_shape[0]/mult , image_shape[1]/mult))
    fct = min(image_shape[0]//mult , image_shape[1]//mult)
    if fct == 0: fct = 1
    new_shape = fct * mult
    return (new_shape,new_shape) 

def fillBorder(image):
    w0,h0 = image.shape
    w,h = findCloseProps(image.shape)
    return cv2.resize(image,dsize=(w,h))
    output = cv2.copyMakeBorder(
        image,
        top = 0,
        bottom= w - w0,
        right= h - h0,
        left= 0,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    return output


def predict_color_path(image_path,shape):
    original_img =  cv2.imread( image_path , 0 )
    gray_img = cv2.imread( image_path  )

    
    lab = cv2.cvtColor(gray_img,cv2.COLOR_BGR2LAB)
    gray_img = lab[:,:,0]

    original_shape = gray_img.shape
    
    #gray_img = fillBorder(gray_img)
    
    gray_img = cv2.resize(gray_img,dsize=(224,224))
    if shape[0] == 0:
        shapex,shapey = gray_img.shape
    else:
        shapex = shape[0]
        shapey = shape[1]

    
    norm_img = np.array([ normalize(gray_img).reshape((shapex,shapey,1)) ]) 

    res = model.predict(norm_img)[0]

    actual_ab = unnormalize(res)          
    actual_l = unnormalize(norm_img[0])
    actual_img = get_rgb_image(actual_l,actual_ab)

    #actual_img = actual_img[0:original_shape[0],0:original_shape[1]]

    actual_img = cv2.resize(actual_img,dsize = (original_shape[1],original_shape[0]),interpolation=cv2.INTER_CUBIC )

    actual_lab = cv2.cvtColor(actual_img,cv2.COLOR_RGB2LAB)
    actual_lab[:,:,0] = original_img
    actual_img = cv2.cvtColor(actual_lab,cv2.COLOR_LAB2RGB)

    return actual_img


def openImage():
    global filename, img, image,canvas
    filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("Image Files",["*.jpg","*.png","*.jfif"]),("all files","*.*")))  
    img = Image.open(filename)
    img = img.resize((300, 300), Image.ANTIALIAS) 
    img = ImageTk.PhotoImage(img)   
    image = canvas.create_image(0,0, anchor = 'nw', image = img)
    

def colorize():
    global filename,image,canvas,rgb,rgb_image,rgb_img
    rgb = predict_color_path(filename,(0,0))
    rgb_image = Image.fromarray(rgb)
    rgb_image = rgb_image.resize((300, 300), Image.ANTIALIAS) 
    rgb_img = ImageTk.PhotoImage(rgb_image)   
    canvas.delete("all")
    image = canvas.create_image(0,0, anchor='nw',image=rgb_img)


def save():
    global rgb
    file = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
    file = file.name
    if file:
        rgb = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        cv2.imwrite(file,rgb)

# Open Button
openBtn = Button(window, text ="Open", command = openImage)
openBtn.pack(side=LEFT,padx=10,pady=10)

# Convert Button
colorizeBtn = Button(window, text ="Colorize", command = colorize)
colorizeBtn.pack(side=LEFT,padx=10,pady=10)

# Save Button
saveBtn = Button(window, text ="Save", command = save)
saveBtn.pack(side=LEFT,padx=10,pady=10)


if __name__ == "__main__":
    window.mainloop()
    