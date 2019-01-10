from flask import Flask
from flask import request
from flask import redirect
from flask import session
from flask import jsonify
from keras.models import load_model
import os
import test_network
import train_network
from apscheduler.scheduler import Scheduler
import time
from werkzeug import secure_filename

app = Flask(__name__)

sched = Scheduler()
sched.start()

def job_function():
    print("Hello World")

def train():
    train_network.dataTraining()

# Schedules job_function to be run on the third Friday
# of June, July, August, November and December at 00:00, 01:00, 02:00 and 03:00
sched.add_cron_job(train, hour='19',minute='43')

@app.route("/train")
def hello():
    train()
    return "Success"

@app.route("/classify-image", methods=['GET','POST'])
def classifyImage():
    if request.method=='POST':
        imagefile = request.files['image']
        imagefile.save(os.path.join("examples", secure_filename(imagefile.filename)))
        image_path = "uploads/"+secure_filename(imagefile.filename)
        return test_network.imageClassification(image_path)
        # return jsonify(status=200,str=imagefile)

# schedule.every(2).minutes.do(train)
# schedule.every().day.at("19:10").do(train)

# while True:
#     schedule.run_pending()
#     time.sleep(1)

if __name__ == '__main__':
    app.debug = True
    app.run()
