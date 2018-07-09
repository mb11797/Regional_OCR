from flask import Flask

app = Flask(__name__)


########################## APP ROUTES ##################################

@app.route('/')
def index():
    return "Hello Manas"

app.route()






app.run()
