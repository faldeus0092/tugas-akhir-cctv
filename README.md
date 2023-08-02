# Website for Showing Inference result

This website is meant to use in conjunction with [this repository](https://github.com/faldeus0092/amlogic-s905x-human-detection/tree/main). Please read it before doing this.

  

# Setup

0. Install Postgresql. Based on your OS, refer to [this guide](https://www.codecademy.com/article/installing-and-using-postgresql-locally)
	- login to postgresql cli ```sudo -u postgres psql postgres```
	- create new user from postgresql cli ```CREATE ROLE root WITH LOGIN SUPERUSER PASSWORD '1234';```
	- create the database ```CREATE DATABASE cctv;```
	- Exit postgresql cli using ```exit```. Use [schema.sql](https://github.com/faldeus0092/tugas-akhir-cctv/blob/main/schema.sql) to generate schema ```sudo -u root psql cctv < schema.sql ```
	- create the ```.env``` file so that it matches the user you created before
		```DATABASE_URL = "postgresql://root:1234@localhost:5432/cctv"```

1. Install requirements ```pip install -r requirements.txt```

2. Run the website using ```flask run --host=0.0.0.0```

3. Create a cctv on the table. Each cctv contains name and number. You can do this either by using psql prompt, psql GUI (like postbird), or using built in API

  

example using API POST: ```/api/cctv``` with json loads as follows:

```

{

"name": "Selasar Lab KCKS",

"cctv_number": 3

}

```

5. Run the program on [this repository](https://github.com/faldeus0092/amlogic-s905x-human-detection/tree/main). Detection results should be saved on /static/footages/ and can be seen on website (adjust the IP according to your host) ```http://localhost:5000/video_feed/[cctv_id]```
