from flask import Flask,redirect,url_for,render_template,request,flash

import numpy as np

from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input,decode_predictions
from keras.preprocessing import image

app=Flask(__name__)

Model=load_model('new_model.hd5')
Model.make_predict_function()

@app.route('/')
def Home():
    return render_template('home.html')

@app.route('/DR_info',methods=['POST'])
def Dataset_btn():
    if request.method == 'POST':
         return render_template('dataset.html')
    
@app.route('/',methods=['POST'])
def upload_image():
    imagefile=request.files['imagefile']
    image_path="./Upload_Image/"+imagefile.filename
    imagefile.save(image_path)

    image_size=((64,64))

    img=image.load_img(image_path, target_size=image_size)

    x = image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x/= 255.0  # Rescale pixel values to between 0 and 1, similar to training data
    # x=preprocess_input(x)

    result=Model.predict(x)
    # print(result)
    class_label = "Symptoms" if result[0][0] > 0.5 else "No-symptoms"
    
    # return render_template('test.html',result=class_label)
    if class_label=="Symptoms":
        return render_template('test.html',result=class_label)
    else:
        return render_template('nonsym.html',result=class_label)
    
@app.route('/Graph',methods=['POST'])
def Graphs():
    return render_template('graph.html')


if __name__=="__main__":
    app.run(debug=True)