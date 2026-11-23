# Dados meteorologicos da Australia 


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import tensorflow as tf
# Desabilita a compilação JIT automática do TensorFlow
tf.config.optimizer.set_jit(False)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from tcn import TCN

# Função para abrir o arquivo com chunks 
def AbrirArquivo(path, sep):
    dados = []
    dados_chunks = pd.DataFrame()
    for chunk in pd.read_csv(path, sep=sep, encoding = 'utf-8-sig', chunksize = 100000):
        dados.append(chunk)
    dados_chunks = pd.concat(dados)
    return dados_chunks

#Filtrar dados pela região especifica 
def escolherLocalidade(data, local):
    data = data[data['Location']==local]
    return data 

# Plotar a serie original
def plotarSerie(data, region, variavel):
    plt.figure(figsize=(12, 8))
    plt.title(f'Regiao: {region}')
    plt.plot(data, label= variavel)
    plt.xlabel('Periodo')
    plt.ylabel(f'{variavel}')
    plt.legend()
    plt.show()

# funão para corrigir o dataset/preencher 
def corrigirFalhas(data, modo):
    colunas = ["Date", "Location", "Sunshine",  "WindGustDir", "RainToday", "RainTomorrow", "WindDir9am",  "WindDir3pm"] 
    colunas_nomes = data.columns
    #print(colunas_nomes)
    if modo == "media":
        for col in colunas_nomes:
            if col not in colunas: 
                data[col] = data[col].fillna(data[col].mean())
                data[col] = data[col].fillna(data[col].mean())
    if modo == "median":
        for col in colunas_nomes:
            if col not in colunas: 
                data[col] = data[col].fillna(data[col].median())
                data[col] = data[col].fillna(data[col].median())
    return data 


#Função para montar a janela de previsão temporal 

def montarJanela(data, janela, step, modo):
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    x, y = [], []

    if modo == '1-step':
        for i in range(len(data) - janela):
            x.append(data[i:i+janela, 0])
            y.append(data[i+janela, 0])

    elif modo == 'multi-step':
        for i in range(len(data) - janela - step + 1):
            x.append(data[i:i+janela, 0])
            y.append(data[i+janela:i+janela+step, 0])

    return np.array(x), np.array(y)

# Função para dividir o conjunto de dados para o modelo ARIMA 

def TreinoTesteArima(data, tm):
    tm_treino = int(len(data)*tm)
    y_train = data[:tm_treino]
    y_test = data[tm_treino:]

    return y_train, y_test

# Função para dividir o conjunto de dados para metodos de Machine Learning  
def TreinoTesteML(x, y, tm):

    tm_treino = int(len(x)*tm)
    print(tm_treino)

    x_train = x[:tm_treino]
    x_test = x[tm_treino:]
    
    y_train = y[:tm_treino]
    y_test = y[tm_treino: ]

    return x_train, x_test, y_train, y_test


#Função para avaliar os modelos 
def avaliaModelos(y_pred, y_real):
    y_pred = np.asarray(y_pred).flatten()
    y_real = np.asarray(y_real).flatten()

    print(f"Erro MAE: {mean_absolute_error(y_real, y_pred)}")
    print(f"Erro RMSE: {root_mean_squared_error(y_real, y_pred)}")
    print(f"Erro r2: {r2_score(y_real, y_pred)}")
    print(f"COR = {np.corrcoef(y_real, y_pred)[0, 1]}")
    print(f"MAPE = {mean_absolute_percentage_error(y_real, y_pred)}")
                  
#Plota o resultado dos modelos 
def plotGraficoPR(real, prev, modelo, STEP):

    # Converter os dados para arrays NumPy
    real = np.asarray(real)
    prev = np.asarray(prev)

    # Transformar vetores 1D em matrizes 2D
    if real.ndim == 1:
        real = real.reshape(-1, 1)

    if prev.ndim == 1:
        prev = prev.reshape(-1, 1)

    # Verificar se possuem o mesmo formato
    if real.shape != prev.shape:
        raise ValueError(
            f"Dimensões incompatíveis: "
            f"real={real.shape}, previsto={prev.shape}"
        )

    n_outputs = real.shape[1]

    # Caso tenha mais de uma saída
    if n_outputs > 1:

        fig, axs = plt.subplots( nrows=n_outputs, figsize=(16, 5 * n_outputs), squeeze=False)
        fig.suptitle(f"Modelo: {modelo} - Horizonte: {STEP}", fontsize=16)
        for i in range(n_outputs):
            axs[i, 0].plot(real[:, i],label='Dados Reais')
            axs[i, 0].plot(prev[:, i], label='Dados Previstos')
            axs[i, 0].set_title(f'Horizonte {i+1}')
            axs[i, 0].set_xlabel('Observações')
            axs[i, 0].set_ylabel('Temperatura Maxima')
            axs[i, 0].legend()
            axs[i, 0].grid()

    # Caso seja previsão de um passo
    else:

        plt.figure(figsize=(16, 5))
        plt.title(f"Modelo: {modelo} - Horizonte: {STEP}")
        plt.plot(real[:, 0],label='Dados Reais')
        plt.plot(prev[:, 0],label='Dados Previstos')
        plt.xlabel('Observações')
        plt.ylabel('Temperatura Maxima')
        plt.legend()
        plt.grid()

    # Salvar gráfico
    arquivo = f"METEOROLOGICAS_{modelo}_STEP_{STEP}.png"

    plt.tight_layout()
    plt.savefig(arquivo, dpi=300)
    plt.show()



# Funçao para criar ARIMA
def preverARIMA(y_train, y_test, janela=30, ordem=(2,1,1)):

    y_train = np.asarray(y_train, dtype=float).flatten()
    y_test = np.asarray(y_test, dtype=float).flatten()

    # Junta treinamento e teste para permitir previsões rolling
    serie = np.concatenate([y_train, y_test])

    previsoes = []

    for i in range(len(y_train), len(serie)):

        # Seleciona as últimas observações disponíveis
        dados_janela = serie[i-janela:i]

        # Cria e ajusta o ARIMA
        modelo = ARIMA(
            dados_janela,
            order=ordem
        )

        modelo_ajustado = modelo.fit()

        # Prevê um passo à frente
        pred = modelo_ajustado.forecast(steps=1)

        previsoes.append(float(np.asarray(pred).flatten()[0]))

    return np.array(previsoes)



#Função para criar e treinar o Random Forest 
def criaRF(x_train, y_train):
    RF = RandomForestRegressor()
    RF.fit(x_train, y_train)
    return RF


# Função para criar uma LSTM
def criaLSTM(x_train, y_train):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    n_outputs = 1 if len(y_train.shape) == 1 else y_train.shape[1]

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, activation='tanh',  input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(LSTM(units=25, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['mean_absolute_error'])
    model.summary()
    model.fit(x_train,y_train,epochs=800,batch_size=128,verbose=0)

    return model


# Função para criar uma CNN-LSTM
def criaCNNLSTM(x_train, y_train):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    n_outputs = 1 if len(y_train.shape) == 1 else y_train.shape[1]
    
    regressor = Sequential()
    regressor.add(tf.keras.Input( shape=(x_train.shape[1], x_train.shape[2])))
    regressor.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    regressor.add(MaxPooling1D(pool_size=2))
    regressor.add(LSTM(units=100, return_sequences=True, activation='tanh'))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=False, activation='tanh'))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(25, activation='elu'))
    regressor.add(Dense(10, activation='relu'))
    regressor.add(Dense(5, activation='relu'))
    regressor.add(Dense(n_outputs, activation='linear'))
    regressor.compile(loss='mean_squared_error',optimizer='rmsprop', metrics=['mean_absolute_error'])
    regressor.summary()
    regressor.fit(x_train,y_train,epochs=800,batch_size=128,verbose=0)
    return regressor



# Função para criar uma TCN
def criaTCN(x_train, y_train):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    n_outputs = 1 if len(y_train.shape) == 1 else y_train.shape[1]

    model = Sequential()

    model.add(tf.keras.Input(shape=(x_train.shape[1], 1)))
    #model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(TCN(nb_filters=64,kernel_size=3,dilations=[1, 2, 4, 8], nb_stacks=1, padding='causal', dropout_rate=0.1, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(25, activation='sigmoid'))
    model.add(Dense(n_outputs, activation='linear'))
    #model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mse',metrics=['mae'],jit_compile=False,run_eagerly=True)
    model.summary()
    model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=0)
    return model



# Abrir Arquivo e montar arquivo
path = "weatherAUS.csv"
df =  AbrirArquivo(path, ',')
print(df)

# Ordenar os dados pelas datas
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)


# Filtrar o DataSet por uma região 
df_analise = escolherLocalidade(df, 'Sydney')
df_analise  = corrigirFalhas(df_analise, "median")


# Plota a série completa 
#plotarSerie(df_analise['MaxTemp'].values, 'Sydney', 'MaxTemp')

print(df_analise['MinTemp'].describe())

# Dividir dados para ARIMA
y_train, y_test = TreinoTesteArima(df_analise['MinTemp'].values, 0.7) 


# Aplica o modelo Arima (Treina e Avalia)
#modelo_arima = criarARIMA(y_train)
y_pred = preverARIMA(
    y_train,
    y_test,
    janela=30,
    ordem=(2,1,0)
)

# Avaliar modelo Arima
print("\nRESULTADOS ARIMA")
avaliaModelos(y_pred, y_test)

plotGraficoPR(y_test, y_pred, "ARIMA MinTemp", 1)

#y_pred = modelo_arima.forecast(steps=len(y_test))
#avaliaModelos(y_pred, y_test)


# Montar a janela para os modelos de Machine Learning DataFrame 
x, y = montarJanela(df_analise[['MinTemp']].values, 3, 1, '1-step')
x_train, x_test, y_train, y_test = TreinoTesteML(x, y, 0.7)

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

# Aplica Normalização ao conjunto de treino e teste 
x_train_model = scaler1.fit_transform(x_train)
y_train_model = scaler2.fit_transform(y_train.reshape(-1, 1)).ravel()

x_test_model = scaler1.transform(x_test)


# Aplica o modelo de Random Forest 
RF = criaRF(x_train_model, y_train_model)
y_pred_rf =  RF.predict(x_test_model)

y_pred_rf = scaler2.inverse_transform(y_pred_rf.reshape(-1, 1)).ravel()
y_real = y_test.copy()

print("Avalia modelo Random Forest")
avaliaModelos(y_pred_rf, y_real)
plotGraficoPR(y_real, y_pred_rf, "RF MinTemp", 1)


# Aplica o modelo LSTM
modelo_lstm = criaLSTM(x_train_model, y_train_model)
x_test_lstm = x_test_model.copy()
x_test_lstm = np.reshape(x_test_lstm,(x_test_lstm.shape[0], x_test_lstm.shape[1], 1))
y_pred_lstm = modelo_lstm.predict(x_test_lstm)

y_pred_lstm = scaler2.inverse_transform(y_pred_lstm.reshape(-1, 1)).ravel()
print("Avalia modelo LSTM")
avaliaModelos(y_pred_lstm, y_real)
plotGraficoPR(y_real, y_pred_lstm, "LSTM MinTemp", 1)



# Aplica o modelo CNN-LSTM

CNNLSTM = criaCNNLSTM(x_train_model, y_train_model) 
x_test_cnn_lstm = x_test_model.copy()

x_test_cnn_lstm = np.reshape(x_test_cnn_lstm, (x_test_cnn_lstm.shape[0], x_test_cnn_lstm.shape[1], 1))
#x_test_lstm = x_test_lstm.reshape(x_test_lstm[0], x_test_lstm[1], 1)
y_pred_cnn_lstm = CNNLSTM.predict(x_test_cnn_lstm)

y_pred_cnn_lstm = scaler2.inverse_transform(y_pred_cnn_lstm.reshape(-1, 1)).ravel()
print("Avalia modelo CNN-LSTM")
avaliaModelos(y_pred_cnn_lstm, y_real)
plotGraficoPR(y_real, y_pred_cnn_lstm, "CNN-LSTM MinTemp", 1)


# Aplica o modelo TCN
modelo_tcn = criaTCN(x_train_model, y_train_model)
x_test_tcn = x_test_model.copy()
x_test_tcn = np.reshape(x_test_tcn,(x_test_tcn.shape[0], x_test_tcn.shape[1], 1))
y_pred_tcn = modelo_tcn.predict(x_test_tcn)

y_pred_tcn = scaler2.inverse_transform(y_pred_tcn.reshape(-1, 1)).ravel()
print("Avalia modelo TCN")
avaliaModelos(y_pred_tcn, y_real)
plotGraficoPR(y_real, y_pred_tcn, "TCN MinTemp", 1)

