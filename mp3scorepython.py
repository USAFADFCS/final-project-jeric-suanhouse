# %% [markdown]
# 
# ## Audio to MIDI Transcriber using Basic Pitch
# This script converts audio stems (WAV files) to MIDI files

# %%
!pip install torchcodec

# %%
!pip install demucs
!pip install torch torchaudio
!pip install librosa soundfile
!pip install basic-pitch

# %% [markdown]
# Imports:

# %%
import os
import sys
import torch
from pathlib import Path
from IPython.display import Audio, display
import IPython.display as ipd
print("✓ Libraries imported successfully")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# %% [markdown]
# Definition of function

# %%
def separate_stems(input_file, output_dir="separated_stems", model_name="htdemucs"):
    """
    Separate an audio file into stems using Demucs
    
    Args:
        input_file (str): Path to the input MP3 file
        output_dir (str): Directory where separated stems will be saved
        model_name (str): Demucs model ('htdemucs', 'htdemucs_6s', etc.)
    
    Returns:
        dict: Paths to the separated stem files
    """
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import save_audio
    import torchaudio
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading audio file: {input_file}")
    
    # Load the audio file
    import librosa, torch, numpy as np
    print(f"Loading audio file (librosa): {input_file}")
    y, sr = librosa.load(input_file, sr=44100, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]  # ensure shape (channels, samples)
    wav = torch.from_numpy(y).float()
    
    # Load the Demucs model
    print(f"Loading Demucs model: {model_name}")
    model = get_model(model_name)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    
    # Ensure audio is in the correct format
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    
    # Resample if necessary
    if sr != model.samplerate:
        print(f"Resampling from {sr}Hz to {model.samplerate}Hz")
        resampler = torchaudio.transforms.Resample(sr, model.samplerate)
        wav = resampler(wav)
        sr = model.samplerate
    
    wav = wav.to(device)
    
    # Apply the model to separate stems
    print("Separating stems... (this may take a while)")
    with torch.no_grad():
        sources = apply_model(model, wav, device=device)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get the base name of the input file
    base_name = Path(input_file).stem
    
    # Save each stem
    stem_paths = {}
    sources = sources.cpu()
    
    print(f"\nSaving stems to: {output_dir}")
    for i, source_name in enumerate(model.sources):
        output_file = output_path / f"{base_name}_{source_name}.wav"
        save_audio(sources[0, i], str(output_file), sr)
        stem_paths[source_name] = str(output_file)
        print(f"  ✓ {source_name}: {output_file}")
    
    print("\nSeparation complete!")
    return stem_paths

print("✓ Stem separation function loaded")

# %%
!pip install librosa numpy torch demucs

# %%
import librosa
import torch
import numpy as np

y, sr = librosa.load("dreams.mp3", sr=44100, mono=False)
if y.ndim == 1:
    y = y[np.newaxis, :]  # shape (1, N) for mono
wav = torch.from_numpy(y).float()

# %%
INPUT_MP3 = "dreams.mp3"
import soundfile as sf
import torch
from demucs import audio

# Monkey-patch Demucs's save_audio to use soundfile instead of torchaudio
def safe_save_audio(wav, path, samplerate, **kwargs):
    wav_np = wav.detach().cpu().numpy()
    if wav_np.ndim == 1:
        wav_np = wav_np[np.newaxis, :]
    sf.write(path, wav_np.T, samplerate)
    print(f"✅ Saved safely: {path}")

audio.save_audio = safe_save_audio
# Try the fine-tuned model instead
stem_paths = separate_stems(INPUT_MP3, output_dir="separated_stems", model_name="htdemucs_6s")

# Display results
print("\n" + "="*60)
print("SEPARATED STEMS:")
print("="*60)
for stem_name, path in stem_paths.items():
    print(f"{stem_name:12s}: {path}")

# %% [markdown]
# # Play each stem to verify quality

# %%
print("\nPlayback stems:")
for stem_name, path in stem_paths.items():
    print(f"\n{stem_name.upper()}:")
    display(Audio(path))

# %% [markdown]
# # Audio to MIDI Transcription

# %%
def transcribe_to_midi(input_file, output_dir="midi_output", 
                       onset_threshold=0.5, frame_threshold=0.3,
                       minimum_note_length=127.70, minimum_frequency=None,
                       maximum_frequency=None, melodia_trick=True):
    """
    Transcribe an audio file to MIDI using Basic Pitch
    """
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Transcribing: {input_file}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get the base name
    base_name = Path(input_file).stem
    
    # Run Basic Pitch prediction
    print("Running transcription...")
    
    model_output, midi_data, note_events = predict(
        input_file,
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        melodia_trick=melodia_trick
    )
    
    # Save MIDI file
    midi_file = output_path / f"{base_name}.mid"
    midi_data.write(str(midi_file))
    
    note_count = len(note_events)
    print(f"  ✓ MIDI saved: {midi_file}")
    print(f"  ✓ Detected {note_count} notes")
    
    return {
        'midi_file': str(midi_file),
        'note_count': note_count,
        'note_events': note_events
    }


def transcribe_all_stems(stem_dir="separated_stems", output_dir="midi_output"):
    """
    Transcribe all stem files to MIDI with optimized settings per instrument
    """
    
    # Optimized settings for different instruments
    stem_settings = {
        'vocals': {
            'onset_threshold': 0.5,
            'frame_threshold': 0.3,
            'melodia_trick': True,
        },
        'bass': {
            'onset_threshold': 0.4,
            'frame_threshold': 0.3,
            'minimum_frequency': 32.70,  # E1
            'maximum_frequency': 392.00,  # G4
        },
        'drums': {
            'onset_threshold': 0.7,
            'frame_threshold': 0.5,
            'minimum_note_length': 50,
        },
        'other': {
            'onset_threshold': 0.5,
            'frame_threshold': 0.3,
        },
        'guitar': {
            'onset_threshold': 0.5,
            'frame_threshold': 0.3,
        },
        'piano': {
            'onset_threshold': 0.5,
            'frame_threshold': 0.3,
        }
    }
    
    stem_path = Path(stem_dir)
    wav_files = list(stem_path.glob("*.wav"))
    
    print(f"\nFound {len(wav_files)} stem(s) to transcribe")
    print("="*60)
    
    all_midi = {}
    
    for wav_file in wav_files:
        # Determine stem type
        stem_type = None
        for stype in stem_settings.keys():
            if stype in wav_file.stem.lower():
                stem_type = stype
                break
        
        settings = stem_settings.get(stem_type, {}) if stem_type else {}
        
        print(f"\n[{wav_file.name}]")
        if stem_type:
            print(f"  Type: {stem_type}")
        
        try:
            result = transcribe_to_midi(str(wav_file), output_dir, **settings)
            all_midi[wav_file.stem] = result
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print("\n" + "="*60)
    print("Transcription complete!")
    
    return all_midi

print("✓ Transcription functions loaded")


# %%
!pip install basic-pitch

# %%
try:
    import basic_pitch
    from importlib.metadata import version, PackageNotFoundError

    print("✓ basic-pitch installed successfully")

    try:
        print(f"  Version: {version('basic-pitch')}")
    except PackageNotFoundError:
        print("  (Version information not available)")

except ImportError:
    print("✗ basic-pitch not found - run: pip install basic-pitch")

# %%
midi_results = transcribe_all_stems("separated_stems", "midi_output")

# Display summary
print("\n" + "="*60)
print("TRANSCRIPTION SUMMARY:")
print("="*60)
for stem_name, result in midi_results.items():
    print(f"{stem_name:30s}: {result['note_count']:4d} notes")
    print(f"  → {result['midi_file']}")

# %%
try:
    import pretty_midi
    import matplotlib.pyplot as plt
    
    def visualize_midi(midi_file):
        """Display a piano roll visualization of a MIDI file"""
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        
        # Create piano roll
        piano_roll = midi_data.get_piano_roll(fs=100)
        
        plt.figure(figsize=(14, 4))
        plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='hot', 
                   interpolation='nearest')
        plt.colorbar(label='Velocity')
        plt.xlabel('Time (10ms)')
        plt.ylabel('MIDI Note Number')
        plt.title(f'Piano Roll: {Path(midi_file).stem}')
        plt.tight_layout()
        plt.show()
    
    # Visualize each MIDI file
    for stem_name, result in midi_results.items():
        print(f"\n{stem_name}:")
        visualize_midi(result['midi_file'])

except ImportError:
    print("Install pretty_midi and matplotlib to visualize MIDI files:")
    print("!pip install pretty_midi matplotlib")

# %%
print("\n" + "="*70)
print("PIPELINE COMPLETE!")
print("="*70)
print(f"\nInput file: {INPUT_MP3}")
print(f"\nGenerated files:")
print(f"  • Stems: {len(stem_paths)} files in 'separated_stems/'")
print(f"  • MIDI:  {len(midi_results)} files in 'midi_output/'")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Listen to the separated stems (Cell 5)")
print("2. Check the MIDI files in a DAW or notation software")
print("3. Adjust transcription thresholds if needed:")
print("   - Higher onset_threshold = fewer notes")
print("   - Lower onset_threshold = more notes")
print("\nFor sheet music generation, we'll use music21 in the next phase!")

# %%
!pip install omnizart tensorflow pretty_midi

# %%
!pip install omnizart tensorflow pretty_midi

# %%
import os
from pathlib import Path

def transcribe_to_midi_omnizart(input_file, output_dir="midi_output", mode="music"):
    """
    Transcribe an audio file to MIDI using Omnizart (Music Transcription)
    mode options: 'music', 'melody', 'piano', or 'drum'
    """
    from omnizart.music import app as m_app
    from omnizart.vocal import app as v_app

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"🎶 Transcribing with Omnizart ({mode} mode): {input_file}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = Path(input_file).stem
    midi_file = output_path / f"{base_name}_{mode}.mid"

    # Choose which app to use based on mode
    if mode in ["music", "piano", "drum"]:
        model_app = m_app
    elif mode in ["vocal", "melody"]:
        model_app = v_app
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Perform transcription
    midi_path = model_app.transcribe(input_file, model_path=None, output=midi_file)
    print(f"  ✓ MIDI saved: {midi_path}")

    return str(midi_path)

# %%
transcribe_to_midi_omnizart("dreams_vocals.wav", mode="melody")


