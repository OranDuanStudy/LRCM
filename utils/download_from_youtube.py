"""
YouTube Audio Downloader

Download audio from YouTube videos using youtube_dl.

Usage:
    Add YouTube video URLs to the ydl.download() call below.

Requirements:
    pip install youtube_dl ffmpeg
"""

import youtube_dl

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    # Add YouTube URLs here: ydl.download(['https://www.youtube.com/watch?v=VIDEO_ID'])
    pass
