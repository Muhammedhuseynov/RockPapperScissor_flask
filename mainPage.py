from flask import Flask,jsonify,url_for
from flask import render_template
from flask import request

from werkzeug.utils import redirect, secure_filename
from predictor import predict
app = Flask(__name__)

import random


classes = {0:"nothing",1:"Papper",2:"Rock",3:"Scissor"}

def playGame(model_predicted):
    
    if model_predicted is not "nothing":
        
        global winner
        comp = random.randrange(1,4)
        comp_hand = classes[comp].lower()
        human_hand = model_predicted.lower()
        # print(f"Comp: {comp_hand}")
        # print(f"Human: {human_hand}")
        if (human_hand == "papper" and comp_hand == "rock") or (human_hand == "rock" and comp_hand == "scissor") or (human_hand == "scissor" and comp_hand == "papper"):
            winner = "Human"
        elif (human_hand == "papper" and comp_hand == "papper") or (human_hand == "rock" and comp_hand == "rock") or (human_hand == "scissor" and comp_hand == "scissor"):
            winner = "Draw"
        else:
            winner = "Computer"
        return [winner,human_hand,comp_hand]
    else:
        return None
        # print("Please only show your HAND!!!")





humand_hand = None
@app.route("/",methods=["GET","POST"])
def showPage():
    if request.method == "POST":
        
        if len(request.data) != 0: 
            global humand_hand
            data = request.data
            # print("Done")
            img_name = "./static/imgs/im.jpg"
            img_res = open(img_name,"wb")
            img_res.write(data)

            
            humand_hand = img_name
            # predicted = predict(img_name,classes=classes)

            # gameResults = playGame(predicted)
            
            # if gameResults is not None:
            #     print("Result Work")
            #     w,winner,comp_hand = gameResults
            #     # return redirect()
            #     # return render_template("index3.html",winner="nothand")
            # else:
            #     print("Result not Work")
            #     winner = None
            #     # return render_template("index3.html",winner="nothand")
    return render_template("index.html")



# @app.route("/")
# def home():
#     return render_template("index.html")
import time
@app.route("/result",methods=["GET","POST"])
def resultShow():
    if request.method == 'POST':
        print("Poooost")

        
        predicted = predict(humand_hand,classes=classes)

        gameResults = playGame(predicted)
        
        if gameResults is not None:
            print("Result Work")
            
            winNer,predicted_hand,robot_hand = gameResults
            print(f"Robot: {robot_hand}")
            print(f"Human: {predicted_hand}")
            print(winNer)
            # return redirect()
            rb_hand = f"../static/imgs/robotHand_{robot_hand}.jpg"
            return render_template("index3.html",robotHand=rb_hand,
                                                resultRobotHand=robot_hand.upper(),
                                                predictedHand=predicted_hand.upper(),
                                                winner=winNer.lower())
        else:
            print("Result not Work")
            winner = None
            return render_template("index.html",winner="notHand")
        

    elif request.method == 'GET':
        print("gettttt")
        
        return render_template('/')
        # return redirect('')
    else:
        return 'Not a valid request method for this route'





if __name__ == "__main__":
    app.run(port=8080,debug=True)