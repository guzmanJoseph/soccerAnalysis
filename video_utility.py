import cv2

def read_video(video_path):
    capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        flag, frame = capture.read()
        if not flag:
            break
        frames.append(frame)
    return frames

def save_video(frames_of_output_video, path_of_output_video):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(path_of_output_video, fourcc, 24, (frames_of_output_video[0].shape[1], frames_of_output_video[0].shape[0]))
    for frame in frames_of_output_video:
        output.write(frame)
    output.release()