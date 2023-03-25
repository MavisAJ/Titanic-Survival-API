# Titanic-Survival-API(Using FASTAPI)
The pupose of this project is to build an API (using FastAPI) which is able to interact with a machine learning model that is able to predict rightly the possibility of a passenger on the Titanic ship surviving.
The train and test dataset as well as the submision file containing the predictions of the model are all included in this repo.

Setup

First install required packages and set your virtual environment to be able to run API on your local machine 
You need to have Python 3 on your system (a Python version lower than 3.10). Then you can clone this repo and 
being at the repo's root :: repository_name> ... follow the steps below:


Windows

     python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  
  
 MacOS and Linux
    
    python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  
    
 
 To run API locally
 Using file demo
   
        uvicorn src.demo_01.api:app --reload 


FastAPI documentation is available on


https://fastapi.tiangolo.com/




Quick view of how my fastapi looks like


![MicrosoftTeams-image (6)](https://user-images.githubusercontent.com/105258546/227718626-81ee34a8-6f25-499e-8896-446dc03eacfd.png)

Here is alos a link  to my article on this project

https://medium.com/@mavisdorkenu32/machine-learning-api-using-fastapi-792635eaf063

Feedback on the project would much appreciated



