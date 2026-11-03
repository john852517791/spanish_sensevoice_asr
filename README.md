# Spanish SenseVoice ASR

This folder contains tokenizer assets, model code, training, and inference entry points for spoken Spanish SenseVoiceSmall ASR experiments.

## Tokenizer Variants

We will compare three vocabularies under the same acoustic model and data splits:

- `es_char`: coverage baseline for spoken Spanish.
- `es_unigram_1000`: primary SentencePiece unigram tokenizer.
- `es_unigram_2000`: larger SentencePiece unigram comparison.

All variants reserve CTC blank at id `0`. SenseVoice control tokens are appended after text tokens:

```text
<|es|>
<|ASR|>
<|nospeech|>
```

## Normalization

Tokenizer training text is normalized for spoken Spanish:

- lowercase text
- keep `á é í ó ú ü ñ`
- remove punctuation and symbols
- normalize digits, dates, and amounts to Spanish words before this step
- keep spoken fillers as normal words, such as `eh`, `mmm`, `pues`, `bueno`, `vale`, `o sea`

## Input Formats

The tokenizer builder accepts `.jsonl`, `.txt`, `.csv`, and `.tsv` files or directories.

Recommended training manifests are JSONL files with one utterance per line:

```json
{"audio": "/data/spanish_asr/wavs/8882/11802/8882_11802_001531.flac", "text": "sudanés historia del negko bakhita tercer eunuco sudanés", "utt_id": "8882_11802_001531", "duration": 3.42, "source": "librispeech_es"}
```

Convert a LibriSpeech-ES style txt file where each line is `<utt_id><tab><transcription>`:

```bash
python spanish_sensevoice_asr/scripts/convert_librispeech_es_manifest.py \
  --input train.txt \
  --output spanish_sensevoice_asr/data/manifests/train.jsonl \
  --audio-root /data/spanish_asr/wavs \
  --compute-duration
```

For an id like `8882_11802_001531`, the script writes audio path `/data/spanish_asr/wavs/8882/11802/8882_11802_001531.flac` by default. Use `--audio-ext .wav` if your audio files are WAV. Omit `--compute-duration` to write `duration: 0.0` without reading audio headers. Duration extraction uses `soundfile`, not `torchaudio`.

For `.jsonl`, it checks these transcript fields in order:

```text
target, text, transcript, sentence, normalized_text
```

This is compatible with the SenseVoice sample format where transcript text is stored in `target`.

## Build

Install SentencePiece for the unigram tokenizers:

```bash
pip install -r spanish_sensevoice_asr/requirements.txt
```

Generate all three tokenizer variants:

```bash
python spanish_sensevoice_asr/scripts/build_tokenizers.py \
  --input path/to/train_manifest_or_text \
  --output-dir spanish_sensevoice_asr/data/vocab \
  --normalized-text spanish_sensevoice_asr/data/text/tokenizer_train.es.txt
```

If SentencePiece is not available and you only want the char baseline:

```bash
python spanish_sensevoice_asr/scripts/build_tokenizers.py \
  --input path/to/train_manifest_or_text \
  --skip-sentencepiece
```

## Comparison Rule

Use the same train/dev/test split, audio preprocessing, model architecture, and decoding settings for all three vocabularies. Do not change tokenizer when comparing WAV, MP3 codec augmentation, 8 kHz, or 16 kHz audio conditions.

## Model And Training

The trainable model uses a local SenseVoiceSmall SANM encoder implementation and the FunASR pip CTC module. It does not require the external `SenseVoice/` source folder at runtime:

- 8 kHz mono audio by default
- 80-bin Kaldi fbank features
- `SenseVoiceEncoderSmall` SANM encoder
- FunASR CTC projection/loss with blank id `0`
- no multilingual language query, ITN switch, or emotion/rich-label auxiliary objective
- total loss is the Spanish transcript CTC loss

During training, the dataloader can lazily normalize source audio formats. By default, the first read of `/path/utt.flac` or `/path/utt.mp3` decodes it, converts to mono 8 kHz WAV, and writes `/path/utt_8kwav.wav`. Later epochs read that cached WAV directly. Disable this with `--no-cache-resampled-wav`.

The Spanish adaptation does not add ITN tokens or train auxiliary SenseVoice control tasks. Training targets are the normalized Spanish transcript tokens only. Decoding ignores special tokens and returns clean Spanish text.

Manifest files can be `.jsonl`, `.tsv`, `.csv`, or `.txt`. They must contain one audio path field and one transcript field.

Audio fields checked in order:

```text
audio, audio_path, audio_filepath, wav, path, file
```

Transcript fields checked in order:

```text
target, text, transcript, sentence, normalized_text
```

Train with the primary unigram tokenizer:

```bash
python spanish_sensevoice_asr/scripts/train_ctc.py \
  --train-manifest path/to/train.tsv \
  --dev-manifest path/to/dev.tsv \
  --audio-root path/to/audio_root \
  --exp-dir spanish_sensevoice_asr/exp/es_unigram_1000_ctc \
  --scheduler cosine \
  --warmup-steps 1000 \
  --grad-accum-steps 1 \
  --auto-resume
```

Multi-GPU DDP training uses `torchrun`:

```bash
torchrun --nproc_per_node 4 spanish_sensevoice_asr/scripts/train_ctc.py \
  --train-manifest path/to/train.jsonl \
  --dev-manifest path/to/dev.jsonl \
  --audio-root path/to/audio_root \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --scheduler cosine \
  --warmup-steps 4000 \
  --exp-dir spanish_sensevoice_asr/exp/es_unigram_1000_ctc \
  --auto-resume
```

Resume explicitly from a checkpoint:

```bash
python spanish_sensevoice_asr/scripts/train_ctc.py \
  --train-manifest path/to/train.jsonl \
  --dev-manifest path/to/dev.jsonl \
  --resume spanish_sensevoice_asr/exp/es_unigram_1000_ctc/last.pt
```

Run greedy inference from a checkpoint:

```bash
python spanish_sensevoice_asr/scripts/infer_ctc.py \
  --checkpoint spanish_sensevoice_asr/exp/es_unigram_1000_ctc/best.pt \
  --audio path/to/audio.wav
```

For `es_char`, pass `--tokens spanish_sensevoice_asr/data/vocab/es_char/tokens.json --sp-model ''`. For `es_unigram_2000`, pass its matching `tokens.json` and `.model` paths.

## Training Server Validation

Run these checks in order on the training server from the workspace root.

### 1. Install Dependencies

```bash
python -m pip install -r spanish_sensevoice_asr/requirements.txt
```

### 2. Verify Python Imports

```bash
python - <<'PY'
import torch
import torchaudio
import sentencepiece
import funasr
from importlib.metadata import version

print('torch', torch.__version__)
print('torchaudio', torchaudio.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
  print('cuda_device', torch.cuda.get_device_name(0))
print('sentencepiece', sentencepiece.__version__)
print('funasr', version('funasr'))
PY
```

### 3. Verify Tokenizer Assets

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path('spanish_sensevoice_asr/data/vocab')
for name in ['es_char', 'es_unigram_1000', 'es_unigram_2000']:
  tokens = json.loads((base / name / 'tokens.json').read_text(encoding='utf-8'))
  print(name, len(tokens), tokens[:4], tokens[-3:])
  assert tokens[0] == '<blank>'
  assert '<|withitn|>' not in tokens
  assert '<|woitn|>' not in tokens
  assert all(ch in tokens for ch in ['á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ'])
PY
```

### 4. Verify Spanish Text Normalization And Tokenization

```bash
python - <<'PY'
from spanish_sensevoice_asr.model.tokenizer import SpanishAsrTokenizer

tokenizer = SpanishAsrTokenizer(
  'spanish_sensevoice_asr/data/vocab/es_unigram_1000/tokens.json',
  'spanish_sensevoice_asr/data/vocab/es_unigram_1000/es_unigram_1000.model',
)
text = 'Tengo 500 pesos pendientes para el 30/07/2024.'
token_ids = tokenizer.encode(text)
print(tokenizer.normalize(text))
print('num_tokens', len(token_ids))
print('decoded', tokenizer.decode_ctc(token_ids))
PY
```

Expected normalized text:

```text
tengo quinientos pesos pendientes para el treinta de julio de dos mil veinticuatro
```

### 5. Verify The Training Manifest

Set the manifest and audio root for your server layout:

```bash
export TRAIN_MANIFEST=train.tsv
export AUDIO_ROOT=/path/to/audio_root
```

Check required columns and whether the first audio files exist:

```bash
python - <<'PY'
import csv
import os
from pathlib import Path

manifest = Path(os.environ['TRAIN_MANIFEST'])
audio_root = Path(os.environ['AUDIO_ROOT'])
audio_fields = ('audio', 'audio_path', 'audio_filepath', 'wav', 'path', 'file')
text_fields = ('target', 'text', 'transcript', 'sentence', 'normalized_text')

with manifest.open('r', encoding='utf-8', newline='') as stream:
  reader = csv.DictReader(stream, delimiter='\t' if manifest.suffix == '.tsv' else ',')
  print('columns', reader.fieldnames)
  rows = [row for _, row in zip(range(20), reader)]

audio_field = next((field for field in audio_fields if rows and rows[0].get(field)), None)
text_field = next((field for field in text_fields if rows and rows[0].get(field)), None)
print('audio_field', audio_field)
print('text_field', text_field)
assert audio_field is not None, 'No supported audio path field found'
assert text_field is not None, 'No supported transcript field found'

missing = []
for row in rows:
  audio_path = Path(row[audio_field])
  if not audio_path.is_absolute():
    audio_path = audio_root / audio_path
  if not audio_path.exists():
    missing.append(str(audio_path))

print('checked_rows', len(rows))
print('missing_audio_files', len(missing))
if missing:
  print('\n'.join(missing[:5]))
  raise SystemExit(1)
PY
```

### 6. Verify Model Forward Pass

This uses synthetic fbank features and does not require audio files. It verifies the SenseVoice SANM encoder and Spanish CTC loss path.

```bash
python - <<'PY'
import torch
from spanish_sensevoice_asr.model.ctc_model import SenseVoiceSmallCTC
from spanish_sensevoice_asr.model.tokenizer import SpanishAsrTokenizer

tokenizer = SpanishAsrTokenizer(
  'spanish_sensevoice_asr/data/vocab/es_unigram_1000/tokens.json',
  'spanish_sensevoice_asr/data/vocab/es_unigram_1000/es_unigram_1000.model',
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SenseVoiceSmallCTC(
  vocab_size=tokenizer.vocab_size,
  encoder_dim=64,
  num_layers=2,
  num_heads=4,
  feedforward_dim=128,
  blank_id=tokenizer.blank_id,
  ignore_id=tokenizer.ignore_id,
).to(device)
features = torch.randn(2, 120, 80, device=device)
lengths = torch.tensor([120, 100], dtype=torch.long, device=device)
targets = [
  torch.tensor(tokenizer.encode('hola mundo'), dtype=torch.long),
  torch.tensor(tokenizer.encode('buenos dias'), dtype=torch.long),
]
text = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=tokenizer.ignore_id).to(device)
text_lengths = torch.tensor([target.numel() for target in targets], dtype=torch.long, device=device)
loss, stats, weight = model(features, lengths, text, text_lengths)
print('device', device)
print('loss', float(loss.detach().cpu()))
print('loss_ctc', float(stats['loss_ctc'].detach().cpu()))
print('weight', int(weight))
assert torch.isfinite(loss)
assert weight == 2
PY
```

### 7. Verify One Batch Training Smoke Test

Use very small settings first. This proves audio loading, fbank extraction, tokenization, SenseVoice CTC loss, backprop, and checkpoint writing.

```bash
python spanish_sensevoice_asr/scripts/train_ctc.py \
  --train-manifest "$TRAIN_MANIFEST" \
  --audio-root "$AUDIO_ROOT" \
  --exp-dir spanish_sensevoice_asr/exp/smoke_es_unigram_1000_ctc \
  --encoder-dim 64 \
  --num-layers 2 \
  --num-heads 4 \
  --feedforward-dim 128 \
  --batch-size 2 \
  --epochs 1 \
  --num-workers 0 \
  --log-interval 1
```

Check that a checkpoint was written:

```bash
ls -lh spanish_sensevoice_asr/exp/smoke_es_unigram_1000_ctc/best.pt
```

### 8. Verify Greedy Inference

Pick one audio file from the manifest or set it manually:

```bash
export TEST_AUDIO=/path/to/audio.wav

python spanish_sensevoice_asr/scripts/infer_ctc.py \
  --checkpoint spanish_sensevoice_asr/exp/smoke_es_unigram_1000_ctc/best.pt \
  --audio "$TEST_AUDIO" \
  --encoder-dim 64 \
  --num-layers 2 \
  --num-heads 4 \
  --feedforward-dim 128
```

The smoke checkpoint is not expected to produce good transcription quality. It only verifies that the full inference path runs end to end.

### 9. Start Full Training

After the smoke test passes, start the primary training run:

```bash
python spanish_sensevoice_asr/scripts/train_ctc.py \
  --train-manifest "$TRAIN_MANIFEST" \
  --audio-root "$AUDIO_ROOT" \
  --exp-dir spanish_sensevoice_asr/exp/es_unigram_1000_ctc \
  --batch-size 8 \
  --epochs 20 \
  --num-workers 4
```
