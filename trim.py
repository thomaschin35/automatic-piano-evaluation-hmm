import librosa
import soundfile as sf

# Load the file
y, sr = librosa.load("c_major_staccato_trim.wav", sr=22050, mono=True)

# Ascending scale is from 0.0s to 1.35s
start_sample = 0
end_sample = int(2.75 * sr)

y_asc = y[start_sample:end_sample]
sf.write("c_major_asc_only.wav", y_asc, sr)
print(f"Created c_major_asc_only.wav ({len(y_asc)/sr:.2f} seconds)")