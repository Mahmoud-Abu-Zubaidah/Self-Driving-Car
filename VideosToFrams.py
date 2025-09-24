import cv2
import os
import uuid
def create_frams_from_videos():
    print("Extracting frames from videos...")
    VIDEOS_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'video') # Path to the directory containing video files
    # List of .avi video files in the directory. you can change the extension if needed based on your video format
    videos = [vid for vid in os.listdir(VIDEOS_PATH) if vid.endswith(".avi")] 
    num_videos = 2 # Number of videos to process

    # Create a directory to save extracted frames if it doesn't exist
    if not os.path.exists(os.path.join(VIDEOS_PATH, 'new_frames')):
        print("Creating directory for extracted frames...")
        os.makedirs(os.path.join(VIDEOS_PATH, 'new_frames'))
    #-----------------------------------------------
    # Loop through each video file
    # 

    for i in range(num_videos):
        print(f"Processing video {i+1}/{num_videos}: {videos[i]}")
        fram_num = -1
        video = cv2.VideoCapture(os.path.join(VIDEOS_PATH, videos[i]))
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)
        while True:
            ret, frame = video.read()
            fram_num += 1
            if not ret:
                break
            if fram_num % 20 == 0:
                cv2.imwrite(os.path.join(VIDEOS_PATH, 'new_frames', f'{str(uuid.uuid1())}.jpg'),frame)
        print(f"Finished processing video {i}")
        print("Number of frames extracted:", int(fram_num/20))
        video.release()