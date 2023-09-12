"""
napari doesn't work for me so this is an extremely basic alternative
"""
from pyqtgraph import ImageItem
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider
from PyQt5 import QtWidgets
import pyqtgraph as pg
from .qt_util import needs_qt

import numpy as np

class ImageViewWindow(QtWidgets.QMainWindow):

    def __init__(self,img, *args, **kwargs):
        super(ImageViewWindow, self).__init__(*args, **kwargs)
        
        self.setWindowTitle('Image Viewer')
                
        w = pg.GraphicsLayoutWidget()
        
        ltop = QtWidgets.QHBoxLayout()
        
        layout = QtWidgets.QVBoxLayout()
        ltop.addLayout(layout)
        #btnstart = QtWidgets.QPushButton("Take ZStack")
        #btnstart.setMaximumSize(100,32)
        #btnstart.clicked.connect(self.recordZStack)

        ctlLayout = QtWidgets.QGridLayout()
        #ctlLayout = QtWidgets.QHBoxLayout()

        sliders=[]        
        for i in range( len(img.shape)-2 ):
            slider = QSlider(Qt.Horizontal)
            slider.setFocusPolicy(Qt.StrongFocus)
            slider.setTickPosition(QSlider.TicksBothSides)
            slider.setTickInterval(10)
            slider.setSingleStep(1)
            slider.setMaximum(img.shape[i]-1)
            slider.valueChanged.connect(self.sliderChange)
            layout.addWidget(slider)#, 0, 0)
            sliders.append(slider)
        self.sliders=sliders

        #layout.addWidget(btnstart)#, 0, 0)

        self.info = QtWidgets.QLabel()
        ctlLayout.addWidget(self.info, 0, 2, Qt.AlignLeft)

        layout.addLayout(ctlLayout)
        layout.addWidget(w)

        central = QtWidgets.QWidget()
        central.setLayout(ltop)
        self.setCentralWidget(central)
        
        v = w.addViewBox(row=0, col=0)
        self.imv = ImageItem()
        v.addItem(self.imv)
        
        #v = w.addViewBox(row=0, col=1)
        #self.zstackImv = ImageItem()
        #v.addItem(self.zstackImv)

        self.imv.setImage(img)
        self.data = img
        
        """
        h = img.shape[-2]
        w = img.shape[-1]
        
        if h > 600 or w > 600:
            scale = 600 / max(w,h)
            w *= scale
            h *= scale
            
        #if h < 50 and w < 
            
        x_space = self.width() - v.width()
        y_space = self.height() - v.height()
        
        self.resize(x_space+w, y_space+h)
        """
        
        self.sliderChange()

    def sliderChange(self):
        d = self.data
        ix=[]
        for s in self.sliders:
            ix.append(s.value())
            d=d[s.value()]
        self.imv.setImage(d.T)
        self.info.setText("["+','.join([str(i) for i in ix])+']')

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        #self.closeCamera()
        ...

        
    def closeEvent(self, event):
        #print('close event')
        #self.closeCamera()
        event.accept()



@needs_qt
def array_view(img):
    if getattr(img, "cpu", None): # convert torch tensors without importing torch
        img = img.cpu().numpy()
    w = ImageViewWindow(np.array(img))
    w.show()
    app = QtWidgets.QApplication.instance()
    app.exec()
    return w


if __name__ == '__main__':
    img = np.random.uniform(0, 100, size=(20,5,200,200)).astype(np.uint8)
    array_view(img)


    