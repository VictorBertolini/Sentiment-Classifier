# Classificador de sentimento em texto 

## O que faz 
O classificador de sentimento a partir de uma base de dados pega do kaggle, faz o processamento de aprendizagem utilizando Naive Bayes e então consegue pegar textos e classificá-lo em 6 sentimentos, sendo eles:

- `Sadness`
- `Joy`
- `Love`
- `Anger`
- `Fear`
- `Surprise`

## Objetivo 
O objetivo desse programa é divulgar como um classificador de sentimentos pode ser feito de forma simples para entusiastas do tema

## Como compilar


## Como Funciona 
### 1 - Extrair os dados 
Primeiramente devemos extrair os dados da base de dados para manipulá-los:
```python
dataset = Excel("DataBase\emotions.xlsx")
node_list = dataset.getInformation()
```

Usamos uma classe Excel que irá acessar o arquivo em excel emotions.xlsx pego do kaggle 
E então ele é transformado em uma lista de Node's (outra classe criada para armazenar cada palavra e seus dados)

### 2 - Embaralha os dados e separa em treino e teste 

```python
size = len(node_list)
shuffle_index = sample(population=range(0, size), k=size)
node_list = [node_list[i] for i in shuffle_index] 

# Separate the nodes in 'train' and 'test' (80%)
trainNodes = node_list[:round(size * 0.8)]
testNodes = node_list[round(size * 0.8):]
```
Está sendo utilizado 80% dos dados para treino, como recomendado na literatura 

### 3 - Stopwords
Para impedir que o modelo fique confuso com palavras sem sentido de agregar no modelo é pego um .txt de palavras que devem ser retiradas das frases, com fito de melhorar a qualidade do modelo
```python
stopwords = getStopWords("DataBase\stopwords.txt")
```
### 4 - Treinamento
A frase é dividida em palavras, é retirada as stopwords e adicionado ao dicionário. Tal ação é feita com todas as frases da base de dados.
```python
dict = {}

for node in trainNodes:
    node.splitPhrase()
    node.cut_StopWords(stopwords)

    add_dict(dict, node)

occurrencies = getTotalOccurrencies(dict)

save(dict, occurrencies)
```
Após terminar o treino, é pego a quantidade de ocorrências de palavras de cada sentimento 
E salvo em um .txt (Param.txt) todos os resultados do treino 

#### 4.1 - add_dict()
Essa função pegará um dicionário com a base de dados sendo feita no treinamento e um novo Node para adicionar. Para cada palavra da frase contida no Node, ele verifica se existe no dicionário.
- Caso não exista, então ele cria um campo para a palavra e um vetor para cada sentimento
[Sadness, Joy, Love, Anger, Fear, Surprise]
- Caso exista ou já tenha criado o vetor no passo anterior, logo é somado 1 no sentimento respectivo da palavra, caso seja uma palavra derivada de uma frase de Joy, e Joy é a posição 1 do vetor, então ele somará em 1 esse campo, ficando [0, 1, 0, 0, 0, 0]
```python
def add_dict(dict, node):
    for word in node.phrase:
        if word not in dict:
            dict[word] = [0] * 6

        dict[word][node.emotion] += 1
```

### 5 - Pega-se todas as ocorrências de cada sentimento 
```python 
dict = {}
dict, occurrencies = getParams()
```
Todos os valores de Sadness de todas as palavras da base de dados serão somados e colocados na posição 0 do vetor, e isso para cada sentimento


### 6 - Teste e verificação de acurácia do modelo
```python
score = []
for node in testNodes:
    emotion = emotionClassify(dict, occurrencies, node.phrase)

    if node.emotion == emotion:
        score.append(1)
    else:
        score.append(0)

result = sum(score)/len(score)
print(f"Total Rate: {result}")
```

#### 6.1 - emotionClassify()
Para a classficação usa-se a fórmula de Naive Bayes:

$P(S|W) = \frac{P(W|S) \times P(S)}{P(W)}$

A qual deriva para a equação:

$P(S|W) = \frac{S + 1}{Total_s + Total}$

E então faz o cálculo para cada palavra e para cada sentimento:

$R_s = P(W_1|S)\times P(W_2|S) \times ... \times P(W_n|S)$

E então é pego a maior das probabilidades dos resultados obtidos:

$Emotion = max(R_s)$

<b> Index <b>
- $P(S|W) =$ Probabilidade de ser de um sentimento S dado uma palavra W
- $P(W|S) =$ Probabilidade de aparecer uma palavra W dado um sentimento S
- $P(S) =$ Probabilidade de ser um sentimento S
- $P(W) =$ Probabilidade de sair uma palavra da base de dados
- $ S + 1 =$ Quantidade de palavras do sentimento S para uma palavra W com suavização de laplace
- $Total_s =$ Total de palavras existentes do sentimento S
- $Total =$ Total de palavras da base de dados
- $R_s =$ Resultado do sentimento S

```python
def emotionClassify(dict, occurrencies, phrase):
    # Split the phrase in words
    words = phrase.split()

    # Get the total of all emotions in the train base 
    total = sum(occurrencies)

    result = [1] * 6

    # The calculus for each emotion
    for emotion in range(0,6):
        # For each word 
        for word in words:
            if word in dict:
                row = dict[word]
                naiveBayes = (int(row[emotion]) + 1) / (occurrencies[emotion] + total)
                result[emotion] *= naiveBayes

    return result.index(max(result))
```





