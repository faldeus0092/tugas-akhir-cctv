import cv2
import glob
import os

# Target FPS. Isi sesuai rata-rata fps dari model
fps = 3

# Get the width and height of the video frames
width = 960
height = 480

folder = './static/footage'
cctv_numbers = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
for cctv_number in cctv_numbers:
    dates = [name for name in os.listdir(os.path.join(folder, cctv_number)) if os.path.isdir(os.path.join(folder, cctv_number, name))]
    for date in dates:
        mp4file = date+".mp4"
        if not os.path.exists(os.path.join(folder, cctv_number, mp4file)):
            output_filename = os.path.join(folder, cctv_number, mp4file)
            output_video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
            list_of_files = [name for name in os.listdir(os.path.join(folder, cctv_number, date))]
            for filename in list_of_files:
                # print(filename)
                filename_abs_path = os.path.join(folder, cctv_number, date, filename)
                img = cv2.imread(filename_abs_path)
                output_video.write(img)

print("Video saved successfully.")