import os
import sys
import datetime

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

import pandas as pd
import numpy as np

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

path = "d:\\project\\python\\naibaho\\"

class MainWin(QtWidgets.QWidget):
    
    def __init__(self):
        super().__init__()
        self.setWin()

    def setWin(self):
        self.setWindowTitle("Recurrent Neural Network")
        # layout
        self.winLayout = QtWidgets.QGridLayout(self)
        
        # table
        self.tblData = QtWidgets.QTableWidget()
        self.tblData.setRowCount(1)
        self.tblData.setColumnCount(6)
        self.tblData.setHorizontalHeaderLabels( ["Date","Location","New Cases","New Deaths", "New Recovered", "New Active Cases"] )
        
        self.lblJenis = QtWidgets.QLabel("Jenis Data")
        self.cmbJenis = QtWidgets.QComboBox()
        self.cmbJenis.setFont( QtGui.QFont("Arial",14 ))
        self.lblDaerah = QtWidgets.QLabel("Pilih Daerah")
        self.cmbDaerah = QtWidgets.QComboBox()
        self.cmbDaerah.setFont( QtGui.QFont("Arial",14 ))
        #self.lblPeramalan = QtWidgets.QLabel("Pilih Peramalan")
        #self.cmbPeramalan = QtWidgets.QComboBox()
        #self.cmbPeramalan.setFont( QtGui.QFont("Arial",14 ))
        self.btnProses = QtWidgets.QPushButton("Proses")
        self.btnProses.clicked.connect(self.peramalanData)
        
        self.fig = plt.figure()
        self.canv = FigureCanvas(self.fig)

        self.winLayout.addWidget(self.tblData, 0, 0, 10, 2)
        self.winLayout.addWidget(self.lblJenis, 0, 4)
        self.winLayout.addWidget(self.cmbJenis, 1, 4)
        self.winLayout.addWidget(self.lblDaerah, 0, 5)
        self.winLayout.addWidget(self.cmbDaerah, 1, 5)
        #self.winLayout.addWidget(self.lblPeramalan, 2, 4)
        #self.winLayout.addWidget(self.cmbPeramalan, 3, 4)
        self.winLayout.addWidget(self.btnProses, 3, 4, 1, 2)
        self.winLayout.addWidget(self.canv,4,4,6,2)
        self.ax1 = self.fig.add_subplot(111)
        
        # show window
        self.showMaximized()
        self.tampilDataCovid()
        self.tampilJenis()
        self.tampilDaerah()
        #self.parameterPeramalan()
        
    
    def tampilDataCovid(self):
        self.df = pd.read_csv(path+"covid19\\covid_19_indonesia_time_series_all.csv")
        self.tblData.setRowCount(self.df.shape[0])
        no=0;
        for i in range(0,self.df.shape[0]):
            self.tblData.setItem( no,0,QtWidgets.QTableWidgetItem( self.df.iloc[i,0] ) )
            self.tblData.setItem( no,1,QtWidgets.QTableWidgetItem( self.df.iloc[i,2] ) )
            self.tblData.setItem( no,2,QtWidgets.QTableWidgetItem( str(self.df.iloc[i,3]) ) )
            self.tblData.setItem( no,3,QtWidgets.QTableWidgetItem( str(self.df.iloc[i,4]) ) )
            self.tblData.setItem( no,4,QtWidgets.QTableWidgetItem( str(self.df.iloc[i,5]) ) )
            self.tblData.setItem( no,5,QtWidgets.QTableWidgetItem( str(self.df.iloc[i,6]) ) )
            no=no+1
        
        
    def tampilJenis(self):
        self.cmbJenis.addItem("New Cases")
        self.cmbJenis.addItem("New Deaths")
        self.cmbJenis.addItem("New Recovered")
        self.cmbJenis.addItem("New Active Cases")
    
    def tampilDaerah(self):
        lokasi = self.df["Location"].unique()
        for l in lokasi:
            self.cmbDaerah.addItem(l)
    
    def parameterPeramalan(self):
        self.cmbPeramalan.addItem("7 Hari")
        self.cmbPeramalan.addItem("14 Hari")
    
    def searchFile(self, fileName):
        for root, dirs, files in os.walk(path+"model\\"):
            for Files in files:
                try:
                    found = Files.find(fileName)
                    if found != -1:
                        return True
                except:
                    return False
    
    def peramalanData(self):

        # cari file model machine learning
        namaFile = self.cmbDaerah.currentText()+"-"+self.cmbJenis.currentText()+".h5"
        hasilCari = self.searchFile(namaFile)
        if hasilCari==True:
            
            model = load_model(path+"\\model\\"+namaFile)
            scaler = MinMaxScaler(feature_range = (0, 1))
            #pesan = model.summary()
            #print(pesan)
            
            self.data_pilih = self.df.loc[self.df["Location"]==self.cmbDaerah.currentText(), self.cmbJenis.currentText()]
            X_test=[]
            X_p=[]
            for c in self.data_pilih:
                X_test.append(c)
            sx = np.arange(0,self.data_pilih.shape[0])
            
            dat_proses = np.array(X_test)
            dat_proses = dat_proses.reshape(-1,1)            
            X_in = scaler.fit_transform(dat_proses)
            
            # prediksi
            for i in range(10, len(X_in)):
                X_p.append(X_in[i-10:i, 0])

            X_p = np.array(X_p)
            X_p = np.reshape(X_p, (X_p.shape[0], X_p.shape[1], 1))
            prediksi = model.predict(X_p)
            prediksi = scaler.inverse_transform(prediksi)
        
            # Visualisasi
            self.ax1.clear()
            judul = "Prediksi Covid19 ("+self.cmbJenis.currentText()+") untuk daerah "+self.cmbDaerah.currentText()
            self.ax1.plot(sx, X_test, color="red", label="Data Real")
            self.ax1.plot(prediksi, color="blue", label="Data Peramalan")
            self.ax1.set_xlabel("Hari")
            self.ax1.set_ylabel("Jumlah")
            self.ax1.legend()
            self.ax1.set_title(judul)
            self.canv.draw()
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("File Model Machine Learning tidak ditemukan!")
            msg.setWindowTitle("Error")
            msg.exec_()        

        
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWin()
    sys.exit(app.exec_())