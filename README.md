# Ajna

## Dependências

TODO

## Obtendo os dados

1. Siga as intruções aqui: https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/. **aparentemente está funcionando apenas no Windows**.
2. Salve os dados na pasta *datasets*

## Preprocessando os dados

1. Execute o script preprocess.py
2. A saída será salva em *preprocessed_data/*

## Treine o modelo

1. Execute o script train.py


## Avalie o modelo

1. Baixe os dados do [Kaggle]() utilizados num campeonato para estimar o vetor de visão.
2. Preprocesse os dados de maneira similar a mostrada anteriormente para o treino.
3. Avalie o modelo executando: ```python eval_cnn.py --input-path preprocessed_images/test/ --model-checkpoint checkpoints/best-cnn.ckpt```
