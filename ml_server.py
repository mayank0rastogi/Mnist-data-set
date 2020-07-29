
import json
import tensorflow as tf
import numpy as np
import random

from flask import Flask, request

app=Flask(__name__)


model = tf.keras.models.load_model('model1.h5')

feature_model = tf.keras.models.Model(
model.inputs,
[layer.output for layer in model.layers])

_, (x_test,_)=tf.keras.datasets.mnist.load_data()
x_test=x_test/255,

def get_prediction():
    index = np.random.choice(x_test.shape[0])
    image = x_test[index,:,:]
    image_arr = np.reshape(image.(1.784))
    return feature_model.predoct(image_arr),image
@app.route('/',methods=['GET','POST'])
def index():
    if request.method =='POST':
        preds,image = get_prediction()
        final_preds = [p.tolist()for p in preds]
        return json.dumps({
            'prediction':final_preds
            'image': image.to_list()
        })
    return 'welcome to the model server'
if __name =='__main__':
    app.run()








