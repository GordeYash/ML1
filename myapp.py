from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
lb=pickle.load(open('FDS Practice/lb.pkl','rb'))
lb1=pickle.load(open('FDS Practice/lb1.pkl','rb'))
model=pickle.load(open('FDS Practice/model.pkl','rb'))

@app.route('/',methods=['POST','GET'])
def main():
    return render_template('index.html')

@app.route('/index',methods=['POST','GET'])
def home():
    degree=request.args.get('Degree')
    Jyear=request.args.get('jyear')
    tier=request.args.get('tier')
    age=request.args.get('age')
    benched=request.args.get('eben')
    experiance=request.args.get('Experiance')
    
    a=np.array([[degree,Jyear,age,tier,benched,experiance]])
    a[:,0]=lb.fit_transform(a[:,0])
    a[:,-2]=lb1.fit_transform(a[:,-2])
    predicta=model.predict(a)
    if predicta==0:
        return render_template('index.html',name="EMployee Leave")
    elif predicta==1:
        return render_template('index.html',name="Employee stay")
        
        

if __name__=='__main__':    
    app.run(debug=True,port=6788)