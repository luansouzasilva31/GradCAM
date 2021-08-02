import tensorflow as tf
import numpy as np
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model, sample, save_path='./gradcam.png', alpha=0.5, pred_index=None, conv_layer_name=None):
        # model: modelo a ser utilizado
        # conv_layer_name: nome da camada convolucional a ser utilizada.
        self.model = model
        self.sample = sample
        self.save_path = save_path
        self.alpha = alpha
        self.pred_index = pred_index
        self.conv_layer_name = conv_layer_name
        
        # Se conv_layer_name=None, tenta achar automaticamente o ultimo
        # conv layer do modelo.
        if self.conv_layer_name is None:
            self.conv_layer_name = self.find_last_convlayer()
        
        # extract gradcam
        self.extract()
        
        # saving gradcam
        self.save()
        
    
    def find_last_convlayer(self):
        ''' Função para encontrar a última camada convolucional do modelo.
        Se não houver nenhuma, retorna um erro.'''
        
        # reversed: retorna um interator que acessa a sequência na ordem inversa.
        # isso é feito pois se quer analisar da última p/ primeira camada.
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4: # se é um tensor 4D...
                return layer.name
        # Caso não, GradCAM não pode ser aplicado. Assim, um erro é retornado:
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
    
    def extract(self):
        ''' Extrai gradcam da amostra baseado no modelo. '''
        # Definindo model para gradcam
        # insere uma segunda saída: camada convolucional
        grad_model = tf.keras.models.Model([self.model.inputs],
                                           [self.model.get_layer(self.conv_layer_name).output,
                                            self.model.output])
        
        # calculando gradiente para a amostra
        with tf.GradientTape() as tape:
            # retorna as predições para cada saída.
            self.conv_output, preds = grad_model(self.sample) 
            # se a predição não foi previamente informada...
            if self.pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        # gradiente do neurônio de saída
        grads = tape.gradient(class_channel, self.conv_output)
        
        # este é um vetor onde cada entrada é a intensidade média do gradiente
        # em um canal de mapa de recuros específico
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        
        # multplicando cada canal na matriz do gradcam
        # por "quão importante este canal é" em relação à melhor classe prevista
        # em seguida, some todos os canais para obter a ativação da classe do gradcam
        self.conv_output = self.conv_output[0]
        self.heatmap = self.conv_output @ pooled_grads[..., tf.newaxis]
        self.heatmap = tf.squeeze(self.heatmap)
        
        # normalizando o gradcam para fins de visualização
        self.heatmap = tf.maximum(self.heatmap, 0)/tf.math.reduce_max(self.heatmap)
        
        self.heatmap = self.heatmap.numpy()
    
    def save(self):
        ''' Salva e plota o gradcam sobreposto na imagem de base '''
        img = self.sample
        img = np.squeeze(img, axis=0)
        
        self.heatmap = np.uint8(255*self.heatmap)
        
        # colorindo mapa de calor
        jet = cm.get_cmap("jet")
        
        # usando o padrão RGB para as cores do gradcam
        jet_colors = jet(np.arange(256))[:, :3]
        self.jet_heatmap = jet_colors[self.heatmap]
        
        # criando uma imagem com o gradcam reajustado
        self.jet_heatmap = tf.keras.preprocessing.image.array_to_img(self.jet_heatmap)
        self.jet_heatmap = self.jet_heatmap.resize((img.shape[1], img.shape[0]))
        self.jet_heatmap = tf.keras.preprocessing.image.img_to_array(self.jet_heatmap)
        
        # sobrepondo o gradcam e a imagem em um mesmo ambiente de plotagem
        self.superimposed_img = self.jet_heatmap*self.alpha + img*(1-self.alpha)
        self.superimposed_img = tf.keras.preprocessing.image.array_to_img(self.superimposed_img)
        
        # salvando a imagem obtida
        self.superimposed_img.save(self.save_path)
        print('Saved at {}'.format(self.save_path))
        
        #return jet_heatmap, superimposed_img # retorna o gradcam resultante, e a sobreposição.        
            
        
        
        
        
        
        
        
