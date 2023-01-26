import eventlet
from my_app import app

eventlet.monkey_patch()

if __name__ == "__main__":
	app.run()
