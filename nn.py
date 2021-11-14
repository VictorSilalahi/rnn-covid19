import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from keras.callbacks import CSVLogger

path = "d:\\project\\python\\naibaho\\"

class Utama():
	def __init__(self):

		scaler = MinMaxScaler(feature_range = (0, 1))
		
		df = pd.read_csv(path+"covid19\\covid_19_indonesia_time_series_all.csv")
		lokasi = df["Location"].unique()
			
		for l in lokasi:
			# testing untuk indonesia 
			data_pilih = df.loc[df["Location"]==l, "New Cases"]
			dat_proses = data_pilih.to_numpy()
			dat_proses = dat_proses.reshape(-1,1)
				
			dat_scaled = scaler.fit_transform(dat_proses)
			features_set = []
			labels = []
			for i in range(10, len(dat_proses)):
				features_set.append(dat_scaled[i-10:i, 0])
				labels.append(dat_scaled[i, 0])				
			features_set, labels = np.array(features_set), np.array(labels)
			features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
			mod = self.prosesModel(features_set, labels, l)
				
			# ambil data untuk prediksi
			dat_input = scaler.fit_transform(dat_proses)
			X_test = []
			for i in range(10, len(dat_proses)):
				X_test.append(dat_input[i-10:i, 0])
			X_test = np.array(X_test)
			X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))				
				
			# prediksi
			prediksi_covid = mod.predict(X_test)
			prediksi_covid = scaler.inverse_transform(prediksi_covid)
				
			# grafik
			self.prosesGrafik(dat_proses, prediksi_covid,l, "New Cases")
				
	def prosesModel(self, fs, lbls, l):
		model = Sequential()
		# layer 1
		model.add(LSTM(units=50, return_sequences=True, input_shape=(fs.shape[1], 1)))
		model.add(Dropout(0.2))
		# layer 2
		model.add(LSTM(units=50, return_sequences=True))
		model.add(Dropout(0.2))
		# layer 3
		model.add(LSTM(units=50, return_sequences=True))
		model.add(Dropout(0.2))
		# layer 4
		model.add(LSTM(units=50))
		model.add(Dropout(0.2))				
		# layer 5
		model.add(Dense(units = 1))
		# compiling
		model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse', 'mae', 'mape'])
		# pembuatan file log history
		csv_logger = CSVLogger(path+"logs\\"+l+"-New Cases.log", append=True)
		# melakukan fitting
		print("Proses dimulai untuk : ",l)
		model.fit(fs, lbls, epochs = 100, batch_size = 32, callbacks=[csv_logger])
		# evaluasi model
		scores = model.evaluate(fs, lbls, verbose=0)
		# model di simpan
		nama_file = path+"model\\"+l+"-New Cases.h5"
		model.save(nama_file)				
		print("Daerah : ", l, "; model : ",nama_file," telah disimpan.")
		return model
	
	def prosesGrafik(self, asli, prediksi, nama, jenis):
		# ditampilkan di matplotlib
		plt.clf()
		plt.plot(asli, color = "red", label = "Data Real")
		plt.plot(prediksi, color = "blue", label = "Data Prediksi")
		plt.title('Prediksi Covid19 untuk :'+nama)
		plt.xlabel('Waktu')
		plt.ylabel('Jumlah')
		plt.legend()
		plt.savefig(path+"grafik\\"+nama+"-"+jenis+".png")
		print("Grafik data : ",nama," telah disimpan.")
		#plt.show()			
		
if __name__=="__main__":
	Utama()
	sys.exit()
