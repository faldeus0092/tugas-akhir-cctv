# Website for Showing Inference result
This website is meant to use in conjunction with [this repository](https://github.com/faldeus0092/amlogic-s905x-human-detection/tree/main). Please read it before doing this.

# Setup
0. Install Postgresql. Based on your OS, refer to [this guide](https://www.codecademy.com/article/installing-and-using-postgresql-locally)
1. Install requirements ```pip install -r requirements.txt```
2. Use [schema.sql](https://github.com/faldeus0092/tugas-akhir-cctv/blob/main/schema.sql) to generate schema.
   - from the command line ```psql -f schema.sql```
   - from the ```psql``` prompt ```\i schema.sql```
3. Run the website using ```flask run --host=0.0.0.0```
4. Create a cctv on the table. Each cctv contains name and number. You can do this either by using psql prompt, psql GUI (like postbird), or using built in API

example using API POST: ```/api/cctv``` with json loads as follows:
```
{
    "name": "Selasar Lab KCKS",
    "cctv_number": 3
}
```
5. Run the program on [this repository](https://github.com/faldeus0092/amlogic-s905x-human-detection/tree/main). Detection results should be saved on /static/footages/ and can be seen on website (adjust the IP according to your host) ```http://localhost:5000/video_feed/[cctv_id]```
