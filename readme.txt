Used pytorch pretrained Model "mobilenet_v2"
trained 20 epochs
-----------
for training, used codes in "trainModel" folder

-----------
how it is work:
	for webcam used webcamjs library
	it take a picture your hand,and to check your hand(img), used torch mobilenet_v2 model
	bot it is just random number to generate hand
NOTE: hand's backround should be white, if not white model won't predict :)

-----------
how to use:

conda create -n rpsEnv python=3.8.11
conda install flask=2.0.1

pip install requests

conda install torch=1.5.0+cu101
conda install torchvision=0.6.0+cu101
or 
pip install torch-1.5.0+cu101-cp38-cp38-win_amd64.whl
pip install  torchvision-0.6.0+cu101-cp38-cp38-win_amd64.whl 
//if problem in installtion torch and torchvision. Then install whl: https://download.pytorch.org/whl/torch_stable.html

-----------
run:
	python mainPage.py
	Running on http://127.0.0.1:8080/



//by: MUHAMMED_HUSEYNOV_{M.H}