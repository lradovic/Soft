import cv2
import os
import numpy as lr
import matplotlib
import matplotlib.pyplot as plt
import scipy
from PIL import Image
import objekat as objekat
import vector as v
import struct
from sklearn import neighbors, metrics
#from sklearn.datasets import *
#from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def read_idx(filename): 
  with open(filename,'rb') as f:

    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    return lr.fromstring(f.read(), dtype=lr.uint8).reshape(shape)

def load_image(path):
  return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

nbins = 9 # broj binova (unutar histograma ima 9 elemenata)
cell_size = (8, 8) # broj piksela po celiji
block_size = (3, 3) # broj celija po bloku

def reshape_data(input_data):
  nsamples, nx, ny = input_data.shape
  return input_data.reshape((nsamples, nx*ny))

def istrenirajKNN():

  #train = read_idx("data/train-images.idx3-ubyte")
  #train_data = lr.reshape(train,(60000,28*28))
  #train_label= read_idx("data/train-labels.idx1-ubyte")
  labels = []
  images = []
 
  for i in range (0,9):
    image_dir_pos = 'data/trainingSet/'+str(i)
    for img_name in os.listdir(image_dir_pos):
      img_path = os.path.join(image_dir_pos, img_name)
      img = load_image(img_path)
      hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                    img.shape[0] // cell_size[0] * cell_size[0]),
                          _blockSize=(block_size[1] * cell_size[1],
                                      block_size[0] * cell_size[0]),
                          _blockStride=(cell_size[1], cell_size[0]),
                          _cellSize=(cell_size[1], cell_size[0]),
                          _nbins=nbins)
      images.append(hog.compute(img))
      labels.append(i)

  images = lr.array(images)
  labels = lr.array(labels)

  x_train = reshape_data(images)

  print(x_train.shape)
  print(labels.shape)
  knn = neighbors.KNeighborsClassifier(n_neighbors=10).fit(x_train,labels)

  #digit = load_digits()
  #dig = pd.DataFrame(digit['data'][0:1700])

  #train_x = digit['data'][:1700]

  #print(train_x[3].shape)
  #train_y =  digit['target'][:1700]

  #knn = KNeighborsClassifier(10)
  #knn.fit(train_x, train_y)

  return knn

knn = istrenirajKNN()

def knnMetoda(img):

  #plt.imshow(slika)
  #plt.savefig('asdasdsd.png')
  #slika=cv2.resize(slika, (28,28)).flatten()
  #slika=slika.reshape(1,-1)
  #plt.imshow(slika)
  #plt.savefig('KNNKNN.png')
  images=[]
  #image_dir_pos = 'data/testSample/'
          #for img_name in os.listdir(image_dir_pos):
     
     # img_path = os.path.join(image_dir_pos, 'img_639.jpg')
      #img = load_image(img_path)
  hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                    img.shape[0] // cell_size[0] * cell_size[0]),
                          _blockSize=(block_size[1] * cell_size[1],
                                      block_size[0] * cell_size[0]),
                          _blockStride=(cell_size[1], cell_size[0]),
                          _cellSize=(cell_size[1], cell_size[0]),
                          _nbins=nbins)
  images.append(hog.compute(img))

  images = lr.array(images)

  x_train = reshape_data(images)

  vrednost = knn.predict(x_train)
  print(vrednost)

  return vrednost[0]

def najmanjeRastojanje(contour, objekti):
  if len(objekti) is 0:
    return {'objekat': 0, 'euklid': 0}

  minO = objekti[0]
  minD = minO.euklid(contour)

  for o in objekti:
    dist = o.euklid(contour)
    if dist < minD:
      minD = dist
      minO = o

  return {'objekat': minO, 'euklid': minD}


def dodajOkvir(x,y,w,h,saOkvirom):
  
  prosirenjeW = 0
  prosirenjeH = 0

  if w <28:
      prosirenjeW = int((28-w)/2)

  if h < 28:
      prosirenjeH = int((28-h)/2)

  top = prosirenjeH+((28-h)%2)
  bottom = prosirenjeH
  left = prosirenjeW
  right = prosirenjeW+((28-w)%2)

  gotovo= cv2.copyMakeBorder(saOkvirom,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])

  return gotovo

def srediTackice(slikaTackice,uspelo):
  #namestamo kernel
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

  brIteracijaErozija = 1
  brIteracijaDilacija = 1

  #radimo otvaranje
  #for i in range(0,100):
  slikaErozija = cv2.erode(slikaTackice, kernel, brIteracijaErozija) #erozija

  slikaFinal = cv2.dilate(slikaErozija, kernel, brIteracijaDilacija) #dilacija

  #primenjujemo treshold
  if uspelo is True:
    #sivaSlika = cv2.cvtColor(slikaFinal, cv2.COLOR_RGB2GRAY)

    #namestamo parametre za treshold
    tresh = 35
    maxVal = 255

    sivaSlika = cv2.cvtColor(slikaFinal, cv2.COLOR_RGB2GRAY)
    
    retSivo, sivaSlika = cv2.threshold(sivaSlika, tresh, maxVal, cv2.THRESH_BINARY) 
    #plt.imshow(sivaSlika,'gray')
    #plt.savefig('sivaslika.png')
   
  
  return sivaSlika

def koordinateLinije(frejm,uspelo):
  #prebacujemo u RGB jer je prvo u BGR
  slika = cv2.cvtColor(frejm, cv2.COLOR_BGR2RGB)

  print(slika.shape)
  #uzimamo samo plavi kanal 
  #linija na svim video zapisima je plave boje
  plavaSlika=slika
  plavaSlika[:,:,0]=0
  plavaSlika[:,:,1]=0
  #plavaSlika=slika[:,:,2]
  print(plavaSlika.shape)


  #plt.imshow(plavaSlika)
  #plt.savefig('plava.png')

  #hocemo da sklonimo tackice
  #primenjujemo eroziju i dilaciju
  #zovemo nasu metodu za sredjivanje

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

  brIteracijaErozija = 1
  brIteracijaDilacija = 1

  #radimo otvaranje
  slikaErozija = cv2.erode(plavaSlika, kernel, brIteracijaErozija) #erozija

  slikaFinal = cv2.dilate(slikaErozija, kernel, brIteracijaDilacija) #dilacija

  #namestamo parametre za treshold
  tresh = 65
  maxVal = 255

  retSivo, sivaSlika = cv2.threshold(slikaFinal, tresh, maxVal, cv2.THRESH_BINARY) 

  #plt.imshow(sivaSlika,'gray')
  #plt.savefig('slikaSLIKA.png')

  #plt.imshow(sredjenaSlika)
  #plt.savefig('sacuvano2.png')

  #print(sredjenaSlika)

  granicaMin = 50
  granicaMax = 200
  
  #izdvajamo ivice canny metodom
  ivice = cv2.Canny(sivaSlika,granicaMin,granicaMax)

  #plt.imshow(sivaSlika,'gray')
  #plt.savefig('ivice.png')

  #parametri za Hough
  threshold = 60
  minLineLength = 100
  maxLineGap = 10

  #dobijamo koordinate linije
  linije = cv2.HoughLinesP(ivice, 1, lr.pi/180, threshold, 0, minLineLength, maxLineGap)

  return linije

def obradiSliku(frejm,uspelo):
  #prebacujemo u RGB jer je prvo u BGR
  slika = cv2.cvtColor(frejm, cv2.COLOR_BGR2RGB)

  #uzimamo samo crveni kanal 
  #cifre na svim video zapisima su bele boje
  crvenaSlika=slika
  crvenaSlika[:,:,2]=0
  crvenaSlika[:,:,1]=0

  #crvenaSlika=slika[:,:,0]

  #hocemo da sklonimo tackice
  #primenjujemo eroziju i dilaciju
  #zovemo nasu metodu za sredjivanje
  obradjenaSlika = srediTackice(crvenaSlika,uspelo)

  return obradjenaSlika

zbir=[]

def obradiFrejmove(videoFajl,pocetakX,pocetakY,krajX,krajY):

  video =  cv2.VideoCapture(videoFajl)

  
  objekti=[]
  sledeciId= 0
  presli=[]
  brFrejmova = 0
  while(video.isOpened()):

    uspelo, frejm = video.read() 

    #cv2.imshow('jaqhkjdhakj',frejm)


    if uspelo is not True:
      break
    else:

      #brFrejmova=(brFrejmova+1)%11
      #if brFrejmova is not 0:
       # continue

      obradjena = obradiSliku(frejm, uspelo)

      img, contours, hierarchy = cv2.findContours(obradjena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      print(len(contours))

      lll = frejm.copy()
      #cv2.drawContours(lll, contours, -1, (255, 0, 0), 1)
      #cv2.imshow('hhj',lll)
      #cv2.waitKey(0)
      
      
      #plt.imshow(lll)
      #plt.savefig('p.png')

      #region = img[y:y+h+1,x:x+w+1]

      for i in range(0,len(contours)):
        
        kontura = contours[i]
        x,y,w,h = cv2.boundingRect(kontura)
        if ((w > 1 and h > 14) or (w>14 and h>5)) and (w<=28 and h<=28) and hierarchy[0][i][3] == -1:
          
          mini = najmanjeRastojanje(kontura,objekti)

          if mini['euklid'] is 0:
            obj = objekat.Objekat(sledeciId,x,y,w,h) 
            saOkvirom = img[y:(y+h),x:(x+w)]
            proba = dodajOkvir(x,y,w,h,saOkvirom)
           
            obj.vrednost=knnMetoda(proba)  
            sledeciId+=1
            objekti.append(obj)

          elif (mini['euklid']<30): #43
            mini['objekat'].x=x
            mini['objekat'].y=y
            mini['objekat'].w=w
            mini['objekat'].h=h

            #sredi poziciju i nastavi dalje
            print('Promena pozicije '+str(x)+' '+str(y))
          else:
            obj = objekat.Objekat(sledeciId,x,y,w,h) 
            saOkvirom = img[y:(y+h),x:(x+w)]
            proba = dodajOkvir(x,y,w,h,saOkvirom)
            obj.vrednost=knnMetoda(proba)
            sledeciId+=1
            objekti.append(obj)

      print('OBJEKTI: '+str(len(objekti)))

      for o in objekti:
        
        linija = [(pocetakX,pocetakY), (krajX, krajY)]
        pozicija= [o.x+o.w/2,o.y+o.h/2]
        distance, nearest = v.pnt2line(pozicija,linija[0],linija[1])
        
        if(distance<8): #10
          if o.presaoLiniju is 0:
            o.presaoLiniju=1
            presli.append(o)
      
      print('PRESLI: '+str(len(presli)))


  video.release()
  cv2.destroyAllWindows()

  return presli





#main funkcija
#od nje krecemo
def main():
  #ucitavamo video zapise
  
  print('UCITAVAMO VIDEO ZAPISE')
  print('----------------------')

  for brojac in range(0,10): #imamo 10 video zapisa
      putanja='driveVideo/video-'+str(brojac) #putanja u projektu sa nazivom videa
      ekstenzija='.avi' #ekstenzija video zapisa
      videoFajl = (putanja+ekstenzija) #videoFajl je putanja do konkretnog videa u zavisnosti od brojaca

      #ispisujemo putanju sa kojom trenutno radimo
      print('Stigli smo do: '+videoFajl)

      #nakon sto specifiramo putanju
      #ucitavamo video sa kojim radimo u petlji
      video =  cv2.VideoCapture(videoFajl)

      uspelo, frejm = video.read() #citamo prvi frejm, da bi dosli do koordinata plave linije
      #uspelo nam vraca da li je uspelo
      #frejm je frejm sa kojim radimo
      
      #plt.imshow(frejm)
      #plt.savefig('sacuvano.png')
      
      #Hough vrati nekad vise koordinata ako se radi sa konturama
      nizLinija = koordinateLinije(frejm,uspelo)

      if nizLinija is not None:
        #tako da imamo niz linija i moramo izvuci iz niza koordinate 
        #uzecemo prve koje je nasao

        x0 = 0
        y0 = 1

        x1 = 2
        y1 = 3

        pocetakX=int(nizLinija[0,0,x0]) #pocetak X koord
        #Pocetak Y koordinata
        pocetakY=int(nizLinija[0,0,y0])

        #kraj linije
        krajX=int(nizLinija[0,0,x1])
        krajY=int(nizLinija[0,0,y1])
        print(pocetakX,pocetakY,krajX,krajY)
      
      #img = frejm.copy()
      #cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
      #plt.imshow(img)
      #plt.savefig('p.png')
      video.release()

      #poziv frejmove
      presli=obradiFrejmove(videoFajl,pocetakX,pocetakY,krajX,krajY)

      z=0

      for p in presli:
        z+=p.vrednost

      print(z)
      zbir.append(z)

     
      #img = cv2.imread('data/testSample/img_17.jpg', 1)
      #plt.imshow(img)
      #plt.savefig('novi.jpg')
      #hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
      #                           # img.shape[0] // cell_size[0] * cell_size[0]),
       #                 _blockSize=(block_size[1] * cell_size[1],
        #                            block_size[0] * cell_size[0]),
         #               _blockStride=(cell_size[1], cell_size[0]),
          #              _cellSize=(cell_size[1], cell_size[0]),
           #             _nbins=nbins)
      
      
      #vrednost = knnMetoda(lr.array(hog.compute(img)))
      #print(vrednost)
      


      #if brojac is 0:
       # return
      #plt.imshow(proba)
      #plt.savefig('z.png')
  fajl = open("out.txt",'w')
    
  fajl.write('RA 115/2015 Luka Radovic\n'
  'file\tsum\n'
  'video-0.avi\t'+str(zbir[0])+'\n'
  'video-1.avi\t'+str(zbir[1])+'\n'
  'video-2.avi\t'+str(zbir[2])+'\n'
  'video-3.avi\t'+str(zbir[3])+'\n'
  'video-4.avi\t'+str(zbir[4])+'\n'
  'video-5.avi\t'+str(zbir[5])+'\n'
  'video-6.avi\t'+str(zbir[6])+'\n'
  'video-7.avi\t'+str(zbir[7])+'\n'
  'video-8.avi\t'+str(zbir[8])+'\n'
  'video-9.avi\t'+str(zbir[9])+'\n')

  

     


#ako se pokrece start onda ce ovo name
#biti main
if __name__== "__main__":
  main()



