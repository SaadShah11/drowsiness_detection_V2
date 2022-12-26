# Drowsiness Detection V2

## Installation:

1): After cloning the project, goto the directory and run following command 

`pip install -r requirements.txt`

2): For Linux: Make sure you have conda installed, then run the command,

`./dlib.sh`

For Windows: Make sure you have conda installed, then run the following commands,

`conda install -c conda-forge dlib`

`pip install cmake`

`pip install face_recognition`

## Running

Navigate to the project directory and run the following commands,

1): Start rabbitmq in new a terminal window

`rabbitmq-server`

2): Start Celery Worker in a new terminal window

`celery -A my_app:celery worker --loglevel=INFO`

3): Start the flask app in a new terminal window

`gunicorn -w 1 -k eventlet --bind localhost:5000 wsgi:app`