# Grad-CAM
Este repositório contém um script funcional e pronto para uso da técnica GradCAM.

É apenas uma classe contida no arquivo .py, sendo necessários os seguintes packages:
<ul>
  <li>tensorflow (versão no momento: 2.4.1)</li>
  <li>numpy (versão no momento: 1.20.2)</li>
  <li>matplotlib (versão no momento: 3.3.4)</li>
</ul>
  
O script necessita ser chamado apenas uma vez, passando todos os parâmetros necessários, que são:
<ul>
  <li> <em>model</em>: Modelo (com arquitetura e pesos já definidos)</li>
  <li> <em>sample</em>[: Amostra para avaliação no modelo (imagem)</li>
  <li> <em>save_path</em>: diretório para salvar imagem com gradcam sobreposto. Se não informado, será o diretório raiz de gradcam.py</li>
  <li> <em>alpha</em>: O quão o gradcam se destaca na imagem. Padrão 0.5</li>
  <li> <em>pred_index</em>: Predição da amostra no modelo. Se informado, acelera o processamento. Se não, é predito em execução.</li>
  <li> <em>conv_layer_name</em>]: Nome do layer que se deseja observar o gradcam. Se não informado, adota-se a última camada convolucional.</li>
</ul>

