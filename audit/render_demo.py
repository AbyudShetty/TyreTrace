"""Encode actual browser frames and local synthetic narration into an MP4."""
import json
import subprocess
import wave
from pathlib import Path
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'out/demo'


def main():
    capture=json.loads((OUT/'capture.json').read_text());segments=json.loads((OUT/'narration.json').read_text())
    frames=capture['frames'];lines=['ffconcat version 1.0']
    if not frames or frames[0]['time']>.1:raise ValueError('capture lacks the static opening frame; rerun demo:record')
    for i,f in enumerate(frames):
        following=frames[i+1]['time'] if i+1<len(frames) else capture['duration_seconds']
        duration=max(following-f['time'],.001)
        lines.extend([f"file '{f['filename']}'",f'duration {duration:.6f}'])
    lines.append(f"file '{frames[-1]['filename']}'")
    concat=OUT/'frames.ffconcat';concat.write_text('\n'.join(lines)+'\n')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();inputs=[];filters=[]
    for i,segment in enumerate(segments):
        path=OUT/f'voice-{i:02d}.wav'
        with wave.open(str(path)) as w:duration=w.getnframes()/w.getframerate()
        if duration>segment['seconds']-.15:raise ValueError(f'narration {i} is too long: {duration:.2f}s')
        inputs.extend(['-i',str(path)])
        filters.append(f"[{i}:a]apad=pad_dur={segment['seconds']},atrim=duration={segment['seconds']},asetpts=PTS-STARTPTS[a{i}]")
    filters.append(''.join(f'[a{i}]' for i in range(len(segments)))+f'concat=n={len(segments)}:v=0:a=1[a]')
    audio=OUT/'narration.wav'
    subprocess.run([ffmpeg,'-y','-loglevel','error',*inputs,'-filter_complex',';'.join(filters),'-map','[a]','-ar','48000','-ac','2',str(audio)],check=True)
    video=OUT/'TyreTrace-demo.mp4'
    subprocess.run([ffmpeg,'-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-i',str(audio),
        '-vf','fps=15,format=yuv420p','-c:v','libx264','-preset','veryfast','-crf','24','-threads','2',
        '-c:a','aac','-b:a','128k','-shortest','-movflags','+faststart',str(video)],check=True)
    import cv2
    cap=cv2.VideoCapture(str(video));count=cap.get(cv2.CAP_PROP_FRAME_COUNT);fps=cap.get(cv2.CAP_PROP_FPS)
    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH);height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT);cap.release()
    if fps<=0 or count/fps<sum(s['seconds'] for s in segments)-2:raise ValueError('encoded demo is incomplete')
    report={'file':str(video.relative_to(ROOT)),'duration_seconds':count/fps,'width':width,'height':height,
        'size_bytes':video.stat().st_size,'source':'actual local browser screencast',
        'narration':'local Microsoft David Desktop synthetic voice; review before submission'}
    (OUT/'video-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
