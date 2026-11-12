Task 3: Sound Data Collection and Processing
Audio part of Formative 2 (Multimodal Data Preprocessing). It prepares the data for the voiceprint verification model.
Contents
Path	Description
`task3_audio_processing.ipynb`	Full pipeline with outputs: cleaning, visualisation, augmentation, feature extraction
`audio_features.csv`	Training data for the voiceprint model: 48 rows × 55 columns
`unauthorized_audio_features.csv`	1 row from a speaker outside the group (Semana), for the "access denied" demo
`audio/raw/`	Original recordings: 4 members × 2 phrases ("Yes, approve", "Confirm transaction")
`audio/clean/`	16 kHz mono WAV, silence and button clicks removed, peak-normalised
`audio/augmented/`	5 versions per clean clip: `pitch_up`, `pitch_down`, `stretch_slow`, `stretch_fast`, `noise`
`audio/unauthorized/`	The outsider's original and cleaned recording
`plots/audio/`	Waveforms, spectrograms, augmentation examples, PCA plots
Using `audio_features.csv` (Task 4: voiceprint model)
Label: `member` (`david`, `junior`, `rolande`, `sheila`)
Features: every numeric column except `take`, which is 50 features in total:
`duration_s`
`mfcc_*`, `delta_mfcc_*`
`rolloff_*`, `rms_*`, `centroid_*`, `bandwidth_*`, `zcr_*`
Do not use as features: `file`, `phrase`, `take`, `augmentation`
Train/test split: keep all versions of the same recording on the same side. Each recording has 6 rows: the original and its 5 augmented versions. If some of them end up in training and others in testing, the test accuracy will be unrealistically high. Group the split by member + phrase, for example with `GroupShuffleSplit` using `groups = member + "_" + phrase`, or leave one phrase out for testing.
```python
import pandas as pd
df = pd.read_csv("audio_features.csv")
X = df.drop(columns=["file", "member", "phrase", "take", "augmentation"])
y = df["member"]
groups = df["member"] + "_" + df["phrase"]
```
Using it in the command-line app (Task 6)
A new voice input must go through exactly the same processing before it is given to the model:
Convert it to 16 kHz mono WAV.
Clean it with `clean_audio()`.
Turn it into a feature row with `extract_features()`.
Both functions are in the notebook, in sections 1 and 4. The model must reject `unauthorized_audio_features.csv` (Semana) as an unknown speaker. A simple way is to set a threshold on the prediction probability, for example denying access if `max(predict_proba) < 0.6`.
