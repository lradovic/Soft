import cv2 
import numpy as lr
import scipy
import math

class Objekat:
    x=0
    y=0
    w=0
    h=0
    id= 0
    presaoLiniju=0
    vrednost = 0

    def __init__(self, id, x,y,w,h):
        self.id = id
        self.x=x
        self.y = y
        self.w= w
        self.h = h
        self.presaoLiniju = 0
        self.vrednost=0

    def euklid(self, contour):
        
        M = cv2.moments(contour) #centar 
        pozicija = lr.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        #izračunaj euklidsku udaljenost
        dist = math.sqrt( (pozicija[0] - self.x)**2 + (pozicija[1] - self.y)**2 )
        return dist       

     #def namesti(self, x,y,w,h):
      #  self.x=x
       # self.y = y
        #self.w= w
        #self.h = h       
