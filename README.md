![](imgs/ajna.png)

IA capaz de identificar onde um usuário está olhando na tela, a fim de facilitar o uso do computador.

## Dependências

Deve ser utilizado Python 3.6.

Todas as dependências necessárias estão no arquivo requirements.txt. Estas podem ser instaladas manualmente ou ainda instaladas na linha de comando utilizando o comando:

```bash
pip3 install -r requirements.txt
```

## Como executar o AJNA?

### 1. Treinar o modelo (não obrigatório)

Esse passo não é obrigatório, versões pre-treinadas do modelo podem ser [encontradas aqui](https://drive.google.com/open?id=11W2kSWEKYQrJXrpiodB4J73OCCgbe54S). Para ter acesso aos modelos pre-treinados é necessário ter um email @ccc.ufcg.edu.br.

* Para treinar o modelo primeiro siga as intruções disponíveis [aqui]( https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) para gerar as imagens da base unityEyes. **Aparentemente funciona apenas no Windows**.

* Salve os dados do unityEyes na pasta *datasets*

* Execute o arquivo *preprocess_data.py* da seguinte forma:

```bash
python preprocess_data.py --input-path datasets/ --output-path preprocessed_data/
```

* Finalmente, treine o modelo, para isso é necessário arquivos de treino e validação, porém nada impede utilizar os mesmos dados para validar o modelo (embora nesse caso não exista validação).

```bash
python train_cnn.py --train-path preprocessed_data/ --eval-path preprocessed_data/
```

Após treinar o modelo é possível visualizar métricas do treinamento utilizando tensorboard:

```bash
tensorboard --logdir checkpoints/
```

Além disso é também salvo um checkpoint do modelo (pesos aprendidos) que pode ser carregado novamente sem necessidade de treino. O checkpoint é intitulato **best_cnn.ckpt**.

### 2. Avaliar o modelo

Para obter predições do modelo para uma dada imagem basta executar:

```bash
python predict_cnn.py  --input-image preprocessed_data/x.pickle  --model-checkpoint checkpoints/best_cnn.ckpt
```

### 3. Estimar vetor da visão (Gaze estimation)

Para tal é necessário uma webcam.

Basta rodar:

```bash
python live_demo.py --model-checkpoint checkpoints/best_cnn.ckpt
```
