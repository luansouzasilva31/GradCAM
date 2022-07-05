import io
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model


class GradCAM :
    
    def __init__(self , model , layer_name=None) :
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.layer_name = layer_name
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layer_name is None :
            self.layer_name = self.find_target_layer()
    
    def find_target_layer(self) :
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers) :
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4 :
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
    
    def compute_heatmap(self , image , class_idx , eps=1e-8) :
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        grad_model = Model(
            inputs=[self.model.inputs] ,
            outputs=[self.model.get_layer(self.layer_name).output ,
                     self.model.output])
        
        # record operations for automatic differentiation
        with tf.GradientTape() as tape :
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image , tf.float32)
            (conv_outputs , predictions) = grad_model(inputs)
            loss = predictions[: , class_idx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss , conv_outputs)
        
        # compute the guided gradients
        cast_conv_outputs = tf.cast(conv_outputs > 0 , "float32")
        cast_grads = tf.cast(grads > 0 , "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]
        
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guided_grads , axis=(0 , 1))
        cam = tf.reduce_sum(tf.multiply(weights , conv_outputs) , axis=-1)
        
        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w , h) = (image.shape[2] , image.shape[1])
        heatmap = cv2.resize(cam.numpy() , (w , h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        
        return heatmap
    
    @staticmethod
    def plot(image , heatmap , show: bool = True , save: bool = False , savepath: str = './plot.png' , mode: int = 1 ,
             title: str = '' , interpolant: float = 0.5 , colormap='magma' , figsize=(16 , 12)) :
        
        assert 0 < interpolant < 1 , 'Heatmap Interpolation must be between 0 and 1'
        
        # heatmap = cv2.applyColorMap(heatmap , cv2.COLORMAP_VIRIDIS)
        
        if mode == 1 :
            # img = cv2.addWeighted(image , interpolant , heatmap , 1 - interpolant , 0)
            img = (image * interpolant + heatmap * (1 - interpolant)).astype(np.uint64)
        
        elif mode == 2 :
            img = np.concatenate((image , heatmap) , axis=1).astype(np.uint64)
        
        elif mode == 3 :
            img = np.concatenate((image , heatmap) , axis=0).astype(np.uint64)
        
        else :
            raise NotImplementedError(f'mode={mode} not implemented.')
        
        # Plotando
        if show :
            plt.rcParams['figure.dpi'] = 100
            
            fig , axis = plt.subplots(1 , 1 , num=title , figsize=figsize, facecolor='w' , edgecolor='k')
            axis.imshow(img , cmap='magma')
            plt.tight_layout()
            plt.show(block=True)
        
        # Salvando amostra
        if save :
            plt.imsave(savepath , img , cmap=colormap)
            
