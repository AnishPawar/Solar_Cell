import cv2
import matplotlib.pyplot as plt
# import imp
import tensorflow as tf
import numpy as np


def make_preds(model,image):
    # image = cv2.resize(image,(224,224))
    image = np.expand_dims(image,axis=0)
    predVal = model.predict(image)
    predictions = tf.where(predVal < 0.5, 0, 1)
    predictions = predictions.numpy()[0][0]
    # print(predictions)
    predProb = predVal[0][0]
    print(predProb)
    if predictions == 0:
        classProb = 1-predProb
        
        res = "{:.2f}".format(classProb)
        print(f'probability is: {res}')


        return ['cracked',res]
    else:
        classProb = predProb
        
        res = "{:.2f}".format(classProb)
        print(f'probability is: {res}')
        return ['uncracked',res]

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None,inner_model=None):

    grad_model = tf.keras.models.Model(inputs=[inner_model.inputs],
                outputs=[inner_model.get_layer(last_conv_layer_name).output,
                inner_model.output])  

    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds

    print(class_channel)
    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    imgx = cv2.resize(heatmap,(224,224))
    colormap = plt.get_cmap('RdYlBu_r')
    heatmap = (colormap(imgx) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    # imgx = cv2.applyColorMap(imgx, cv2.COLORMAP_JET)
    # gray = cv2.cvtColor(img_array[0], cv2.COLOR_BGR2GRAY)
    # gray=gray/255.

    cv2.imwrite("static/processed_images/kk.png",heatmap)
    hm = cv2.imread('static/processed_images/kk.png')

    # heatmap = heatmap/255.
    s = img_array[0]
    # s=s/255.
    added = cv2.addWeighted(s,0.7,hm,0.9,0.0)

    return added