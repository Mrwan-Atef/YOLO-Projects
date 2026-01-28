import yt_dlp

# Example: A generic professional training drill video
video_url = 'https://www.youtube.com/watch?v=ZvcM6ekmXIw'

ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': 'test_video_cones.mp4',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

print("Video downloaded. You can now run your YOLO tracker on 'test_video_cones.mp4'")