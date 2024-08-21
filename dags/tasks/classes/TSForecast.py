import pandas as pd
import matplotlib.pyplot as plt
import logging
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tasks.classes.SSA import SSA
import math



class TimeSeriesForecast:
    def __init__(self, series) -> None:
        if isinstance(series, pd.Series):
            self.series = series
        
        else:
            try:
                self.series = pd.Series(series)
            
            except Exception as e:
                logging.error("Ошибка при инициализации класса: ", e)
        
    
    def visual_graph(self, title: str = None) -> None:
        plt.figure(figsize=(14, 5))
        self.series.plot()

        if title is None:
            plt.title("Дневное кол-во заболевших COVID-19")
        
        else:
            plt.title(str(title))

        plt.show()
    

    def create_ssa_series(self, L: int = 100) -> None:
        try:
            self.series_ssa = SSA(self.series, L=L)
        
        except Exception as e:
            logging.error("Ошибка во время создния SSA временного ряда: ", e)
        
    
    def plot_wcorr(self, max: int = None, title: str = None) -> None:
        try:
            if title is None:
                title = "W-Correlation for Walking Time Series"

            self.series_ssa.plot_wcorr(max=max)
            plt.title(title)
            plt.show()
        
        except Exception as e:
            logging.error("Ошибка во время отрисовки SSA Heatmap: ", e)

    
    def visual_reconstructed_elements(self, title: str = None, reconstructions: list[list] = []) -> None:
        try: 
            if title is None:
                title = "Walking Time Series: Groups"

            plt.figure(figsize=(12, 5))

            for reconstruct in reconstructions:
                self.series_ssa.reconstruct(reconstruct).plot()

            self.series_ssa.orig_TS.plot(alpha=0.4)
            plt.title(title)
            plt.xlabel(r"$t$ (s)")
            plt.ylabel("Acceleration (G)")
            legend = [
                r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(len(reconstructions))] + ["Original TS"]
            plt.legend(legend);

        except Exception as e:
            logging.error("Ошибка во время группировки элементов: ", e)


    def __replace_negative_values(self):
        self.series = self.series.abs()

    def __normalize_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.series = np.array(self.series).reshape(-1, 1)

        self.series = self.scaler.fit_transform(self.series)


    def __create_dataset(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset)-self.look_back-1):
            a = dataset[i:(i+self.look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + self.look_back, 0])
        return np.array(dataX), np.array(dataY)
    
    
    def train_test_split(self, train_size: float = 0.8, look_back: int = 1,
                         positive: bool = True, normal: bool = True) -> None:
        if not isinstance(train_size, float) or train_size >= 1 or train_size <= 0:
            print("Размер тренировочных данных должен быть представлен в виде 0.0 < train_size < 1.0")
            return
        
        try:
            if positive:
                self.__replace_negative_values()

            if normal:
                self.__normalize_data()
                
            self.train_size = int(len(self.series) * train_size)
            self.test_size = len(self.series) - self.train_size
            self.train = self.series[0:self.train_size,:]
            self.test = self.series[self.train_size:len(self.series),:]
            
            self.look_back = look_back
            self.trainX, self.trainY = self.__create_dataset(self.train)
            self.testX, self.testY = self.__create_dataset(self.test)

            self.trainX = np.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
            self.testX = np.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))
        
        except Exception as e:
            logging.error("Ошибка во время train_test_split: ", e)
    

    def create_model(self, model = None):
        if not model is None:
            self.model = model

        else:
            self.model = keras.models.Sequential()
            self.model.add(keras.layers.LSTM(150, activation="relu", return_sequences=True, input_shape=(1, self.look_back)))
            self.model.add(keras.layers.LSTM(50))
            self.model.add(keras.layers.Dense(1, activation="relu"))
            self.model.compile(loss='mean_squared_error', optimizer='adam')
    

    def fit_model(self, val_split: float = 0.15, batch_size: int = 1, early_stopping: bool = True):
        try:
            if early_stopping:
                early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
            
            else:
                early_stopping = None

            self.model.fit(
                self.trainX, self.trainY, epochs=100, 
                batch_size=batch_size, verbose=2, validation_split=val_split, callbacks=early_stopping)
    
        except Exception as e:
            logging.error("Ошибка во время обучения модели: ", e)

    
    def get_score(self) -> list[float]:
        try:
            self.trainPredict = self.model.predict(self.trainX)
            self.testPredict = self.model.predict(self.testX)

            self.trainY = self.trainY.reshape(-1, 1)
            self.testY = self.testY.reshape(-1, 1)

            # invert predictions
            self.trainPredict = self.scaler.inverse_transform(self.trainPredict)
            self.trainY = self.scaler.inverse_transform(self.trainY)
            self.testPredict = self.scaler.inverse_transform(self.testPredict)
            self.testY = self.scaler.inverse_transform(self.testY)

            trainScore = math.sqrt(mean_squared_error(self.trainY[:, 0], self.trainPredict[:,0]))
            
            testScore = math.sqrt(mean_squared_error(self.testY[:, 0], self.testPredict[:,0]))

            return [trainScore, testScore]

        except Exception as e:
            logging.error("Ошибка во время оценки модели: ", e)
        
    
    def visual_model_pred(self):
        try:
            plt.figure(figsize=(8, 6))
            # shift train predictions for plotting
            self.trainPredictPlot = np.empty_like(self.series)
            self.trainPredictPlot[:, :] = np.nan
            self.trainPredictPlot[self.look_back:len(self.trainPredict)+self.look_back, :] = self.trainPredict
            # shift test predictions for plotting
            self.testPredictPlot = np.empty_like(self.series)
            self.testPredictPlot[:, :] = np.nan
            self.testPredictPlot[len(self.trainPredict)+(self.look_back*2)+1:len(self.series)-1, :] = self.testPredict
            # plot baseline and predictions
            plt.plot(self.scaler.inverse_transform(self.series))
            plt.plot(self.trainPredictPlot)
            plt.plot(self.testPredictPlot)
            plt.show()
        
        except Exception as e:
            logging.error("Ошибка во время визуализации предсказаний модели: ", e)
    

    def get_future_values(self, num_predictions: int = 100):
        last_sequence = self.testX[-1:]
        self.predicted_values = []

        for _ in range(num_predictions):
            # Прогнозирование следующего значения
            next_value = self.model.predict(last_sequence)  # (1, 5, 1) если look_back = 5
            
            next_value = next_value.reshape(1, 1, 1)
            
            # Объединение последних look_back-1 значений с новым предсказанием
            last_sequence = last_sequence[:, :, 1:]

            last_sequence = np.concatenate((last_sequence, next_value), axis=2)  # (5, 1)

            # Обратное масштабирование и добавление в список предсказанных значений
            next_value = self.scaler.inverse_transform(next_value.reshape(1, 1))
            self.predicted_values.append(next_value[0][0])
    

    def visual_future_values(self):
        self.predicted_values = np.array(self.predicted_values).reshape(-1, 1)
        extended_length = len(self.series) + len(self.predicted_values)
        self.futurePredictPlot = np.empty((extended_length, 1))
        self.futurePredictPlot[:, :] = np.nan

        self.futurePredictPlot[len(self.series):, :] = self.predicted_values  # Новые предсказания добавляются в конец

        plt.figure(figsize=(10, 6))
        plt.plot(self.scaler.inverse_transform(self.series), label='Исходные данные', color='blue')
        plt.plot(self.trainPredictPlot, label='Предсказанные значения (обучающая выборка)', color='green')
        plt.plot(self.testPredictPlot, label='Предсказанные значения (тестовая выборка)', color='red')
        plt.plot(self.futurePredictPlot, label='Новые предсказанные значения', color='orange')

        plt.title('Сравнение исходных и предсказанных значений')
        plt.xlabel('Время')
        plt.ylabel('Количество случаев')
        plt.legend()
        plt.show()
