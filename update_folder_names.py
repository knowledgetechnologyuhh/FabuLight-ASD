import os, glob
for filename in glob.glob("/informatik3/wtm_archive/carneiro/WASD/clips_audios/train/*/**.wav"):
    print(filename)
    print(filename.replace("", "_"))
    break
