import os
import base64
import psycopg2
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from flask import Flask, request, render_template, redirect, Response
from flask_mysqldb import MySQL
from dotenv import load_dotenv
from pathlib import Path
import glob

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
FOOTAGE_DIR = os.path.join(ROOT_DIR, "static/footage")

# 1: Lift Gerbang Barat, 2: Selasar Gerbang Barat, 11: Lab KCKS Belakang, 3: Selasar Lab KCKS
CREATE_CCTVS_TABLE = (
    "CREATE TABLE IF NOT EXISTS cctvs (id SERIAL PRIMARY KEY, cctv_number INTEGER, name TEXT);"
)

CREATE_FOOTAGES_TABLE = (
    """CREATE TABLE IF NOT EXISTS footages (cctv_id INTEGER, image_path TEXT, 
    num_detections INTEGER, date TIMESTAMP, FOREIGN KEY(cctv_id) REFERENCES cctvs(id) ON DELETE CASCADE);"""
)

# INSERT_CCTV_RETURN_ID = "INSERT INTO cctvs (name, cctv_number) VALUES (%s, %s) RETURNING id;"
INSERT_CCTV = "INSERT INTO cctvs (name, cctv_number) VALUES (%s, %s)"
RETURN_LAST_ID = "SELECT LAST_INSERT_ID();"

INSERT_FOOTAGE = (
    "INSERT INTO footages (cctv_id, image_path, num_detections, date) VALUES (%s, %s, %s, %s);"
)

GET_CCTV_ID = (
    "SELECT id FROM cctvs WHERE cctv_number = (%s) LIMIT 1;"
)

GET_CCTV_NAME = (
    "SELECT name FROM cctvs WHERE id = (%s) LIMIT 1;"
)

GET_FOOTAGE_FROM_CCTV = (
    "SELECT * FROM footages WHERE cctv_id = (%s) ORDER BY image_path DESC;"
)

GET_LATEST_FOOTAGE_FROM_CCTV = (
    "SELECT image_path FROM footages WHERE cctv_id = (%s) ORDER BY image_path DESC LIMIT 1;"
)

GLOBAL_NUMBER_OF_DAYS = (
    """SELECT COUNT(DISTINCT DATE(date)) AS days FROM footages"""
)

load_dotenv()
app = Flask(__name__)
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')

mysql = MySQL(app)


@app.get('/')
def home():
    return render_template('dashboard.html', page='home')

@app.route('/cctv/<int:cctv_id>')
def cctv_footages(cctv_id):
    cursor = mysql.connection.cursor()
    
    cursor.execute(GET_FOOTAGE_FROM_CCTV, (cctv_id,))
    footages = cursor.fetchall()
    cursor.execute(GET_CCTV_NAME, (cctv_id,))
    name = cursor.fetchone()[0]

    mysql.connection.commit()
    cursor.close()
    return render_template('cctv.html', footages=footages, name=name, page='log')

@app.post('/api/cctv')
def create_cctv():
    data = request.get_json()
    name = data["name"]
    cctv_number = data["cctv_number"]
    # mysql
    cursor = mysql.connection.cursor()
    cursor.execute(CREATE_CCTVS_TABLE)
    cursor.execute(INSERT_CCTV, (name, cctv_number)) #pass a tuple
    cursor.execute(RETURN_LAST_ID)
    cctv_id = cursor.fetchone()[0]
    mysql.connection.commit()
    cursor.close()

    return {"id": cctv_id, "message": f"CCTV {name} dengan no cctv {cctv_number} telah didaftarkan"}

# uncomment after
@app.post("/api/footage")
def store_footage():
    data = request.get_json()
    cctv_number = str(data["cctv_number"])
    num_detections = data["num_detections"]
    date = datetime.now(tz=ZoneInfo("Asia/Jakarta"))

    # determine filename
    filename = f"{date.strftime('%Y-%m-%d_%H%M%S%f')[:-3]}.jpeg"
    date_folder = str(filename.split("_")[0])
    
    # create folder for saving
    Path(os.path.join(FOOTAGE_DIR, cctv_number, date_folder)).mkdir(parents=True, exist_ok=True)
    abs_path = os.path.join(FOOTAGE_DIR, cctv_number, date_folder, filename)

    # to store in db 11/23-06-2023/asdf.jpeg
    image_path = f"{cctv_number}/{date_folder}/{filename}"

    # save the image
    encoded_image = data["image"]
    decoded_image = base64.b64decode((encoded_image))
    img_file = open(abs_path, 'wb')
    with img_file as f:
        f.write(decoded_image)
        f.close()

    cursor = mysql.connection.cursor()
    cursor.execute(GET_CCTV_ID, (cctv_number,))
    cctv_id = cursor.fetchone()[0]
    cursor.execute(CREATE_FOOTAGES_TABLE)
    cursor.execute(INSERT_FOOTAGE, (cctv_id, image_path, num_detections, date))
    mysql.connection.commit()
    cursor.close()

    return {"message": f"Footage for cctv number {cctv_number} added. date: {date}, Image path: {image_path}, detected person: {num_detections}"}

def stream(img_path):
    while True:
        time.sleep(0.3)
        image = get_latest_images(img_path)
        im = open('static/footage/' + image, 'rb').read()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n')

def get_latest_images(img_path):
    img_dir = "/".join(img_path.split("/", 2)[:2])

    #get latest image saved
    list_of_files = glob.glob(f'static/footage/{img_dir}/*.jpeg') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    
    filename = os.path.basename(latest_file)
    
    full_path = os.path.join(img_dir, filename).replace("\\","/")
    return full_path

@app.route('/slideshow/<int:cctv_id>')
def slideshow(cctv_id):
    cursor = mysql.connection.cursor()
    
    cursor.execute(GET_LATEST_FOOTAGE_FROM_CCTV, (cctv_id,))
    img_path = cursor.fetchone()[0]
    
    mysql.connection.commit()
    cursor.close()
    return Response(stream(img_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<int:cctv_id>')
def video_feed(cctv_id):
    cursor = mysql.connection.cursor()

    cursor.execute(GET_CCTV_NAME, (cctv_id,))
    name = cursor.fetchone()[0]
    
    mysql.connection.commit()
    cursor.close()
    return render_template('video_feed.html', cctv_id=cctv_id, name=name, page='live_detection')
