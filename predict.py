import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from sklearn.svm import SVC
import numpy as np
import glob
import imageio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
files=[]
for i in range(1,101):
    files.append('./train/'+str(i)+'.jpg')
Data=np.zeros((100,30000))
count=0
for names in files:
    img=imageio.imread(names)
    img=(img.reshape(1,-1)).astype(float) 
    Data[count,:]=img
    count=count+1
EtiqHom=np.arange(2,102,2)
EtiqHom=np.append(EtiqHom,[59,67])
Label=np.zeros(100)
Label[EtiqHom-1]=1
x_train,x_test,y_train,y_test= train_test_split(Data, Label, train_size=0.7)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
Score=np.array([])
ScoreN=np.array([])
Cvs=np.linspace(-1,2,100)
VC=np.exp(Cvs)
for Cv in VC:
    clf=SVC(C=Cv,kernel='linear')#,gamma='scale')
    clf.fit(x_train,y_train)
    yt_pred= clf.predict(x_test)
    F1=f1_score(y_test,yt_pred)
    Score=np.append(Score,F1)
VCO=VC[np.argmax(Score)]
clf=SVC(C=VCO,kernel='linear')#,gamma='scale')
clf.fit(x_train,y_train)
files_val = glob.glob('./test/*.jpg')
Data_val=np.zeros((len(files_val),30000))
count=0
for names in files_val:
    img=imageio.imread(names)
    img=(img.reshape(1,-1)).astype(float) 
    Data_val[count,:]=img
    count=count+1
Data_val = scaler.transform(Data_val)
ypred=clf.predict(Data_val)
count=0
out=open("test/predict_test.csv","w")
out.write("Name,Target\n")
for names in files_val:
    print(names.split("/")[-1],ypred[count])
    out.write("{},{}\n".format(names.split("/")[-1],ypred[count]))
    count=count+1
out.close()
