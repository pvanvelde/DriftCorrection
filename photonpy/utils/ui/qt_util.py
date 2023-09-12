# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5 import QtCore
import sys

def error_box(txt, title='Error'):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("Error")
    msg.setInformativeText(txt)
    msg.setWindowTitle(title)
    msg.exec_()

def needs_qt(func):
    """
    Decorator to make sure QApplication is only instantiated once
    """
    def run(*args,**kwargs):
        app = QApplication.instance()
        appOwner = app is None
        if appOwner:
            app = QApplication(sys.argv)
        
        r = func(*args,**kwargs)
        
        if appOwner:
            del app
            
        return r
    return run

def force_top(window):
    # bring window to top and act like a "normal" window!
    window.setWindowFlags(window.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)  # set always on top flag, makes window disappear
    window.show() # makes window reappear, but it's ALWAYS on top
    window.setWindowFlags(window.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint) # clear always on top flag, makes window disappear
    window.show() # makes window reappear, acts like normal window now (on top now but can be underneath if you raise another window)