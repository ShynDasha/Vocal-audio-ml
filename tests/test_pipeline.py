import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

import numpy as np
import pandas as pd

import audio_features
import prepare_vocalset
import train


class PipelineTests(unittest.TestCase):
    def test_extraction_repairs_partial_and_manifest_uses_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, raw = root / 'input.zip', root / 'raw'
            name = 'VocalSet/male1/scales/belt/example.wav'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr(name, b'audio fixture')
                output.writestr('__MACOSX/._example.wav', b'ignored')
            prepare_vocalset.extract_audio(archive, raw)
            (raw / name).write_bytes(b'broken')
            prepare_vocalset.extract_audio(archive, raw)
            self.assertEqual((raw / name).read_bytes(), b'audio fixture')
            rows = prepare_vocalset.build_manifest(raw, root / 'manifest.csv')
            self.assertEqual(rows, [dict(path=name, technique='belt', singer='male1', sample_type='scales')])

    def test_rejects_traversal_and_wrong_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('../outside.wav', b'bad')
            with self.assertRaises(ValueError):
                prepare_vocalset.extract_audio(archive, root / 'raw')
            with self.assertRaises(ValueError):
                prepare_vocalset.verify_archive(archive)
            self.assertFalse((root / 'outside.wav').exists())

    def test_alternative_checksum_requires_opt_in_and_valid_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'alternative.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('example.wav', b'audio fixture')
            with self.assertRaises(ValueError):
                prepare_vocalset.verify_archive(archive)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_vocalset.verify_archive(archive, allow_different_checksum=True)
            archive.write_bytes(b'not a zip')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(zipfile.BadZipFile):
                prepare_vocalset.verify_archive(archive, allow_different_checksum=True)

    def test_vibrado_folder_normalizes_label_without_renaming_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'raw/female2/arpeggios/vibrado/f2_arpeggios_vibrato_a.wav'
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b'fixture')
            rows = prepare_vocalset.build_manifest(root / 'raw', root / 'manifest.csv')
            self.assertEqual(rows[0]['technique'], 'vibrato')
            self.assertIn('/vibrado/', rows[0]['path'])
            self.assertTrue(audio.exists())

    def test_selection_and_disjoint_split(self):
        frame = pd.DataFrame([dict(path=f'{s}-{t}-{n}.wav', singer=f'male{s}', technique=t)
                              for s in range(4) for t in ['belt', 'vibrato'] for n in range(4)])
        chosen = audio_features.select_records(frame, 16)
        self.assertEqual(len(chosen), 16)
        self.assertEqual(len(chosen.groupby(['technique', 'singer'])), 8)
        pd.testing.assert_frame_equal(chosen, audio_features.select_records(frame, 16))
        with self.assertRaises(ValueError):
            audio_features.select_records(frame, 4)
        left, right = train.split_dataset(chosen)
        self.assertFalse(set(chosen.iloc[left].singer) & set(chosen.iloc[right].singer))
        self.assertEqual(set(chosen.iloc[right].technique), {'belt', 'vibrato'})

    def test_real_audio_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tone.wav'
            signal = (np.sin(2 * np.pi * 220 * np.arange(16000) / 16000) * 10000).astype('<i2')
            with wave.open(str(path), 'wb') as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(signal.tobytes())
            features = audio_features.process_audio(path)
            self.assertEqual(len(features), 27)
            self.assertTrue(all(np.isfinite(v) for v in features.values()))
            self.assertAlmostEqual(features['f0_mean'], 220, delta=10)

    def test_training_writes_model_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / 'features.csv'
            rows = [dict(path=f'{s}-{t}-{n}.wav', singer=f'male{s}', technique=t,
                         sample_type='scales', mfcc_1=label + n * .01, f0_mean=200 + label * 100)
                    for s in range(4) for label, t in enumerate(['belt', 'vibrato']) for n in range(4)]
            pd.DataFrame(rows).to_csv(features, index=False)
            features.with_suffix('.json').write_text(json.dumps(dict(
                sample_rate=16000, duration=5,
                features_sha256=hashlib.sha256(features.read_bytes()).hexdigest())))
            with patch.object(train, 'FEATURES', features), patch.object(train, 'OUTPUT_DIR', root / 'outputs'):
                with contextlib.redirect_stdout(io.StringIO()):
                    train.main()
            self.assertTrue((root / 'outputs/models/vocal_model.joblib').is_file())
            self.assertTrue((root / 'outputs/reports/confusion_matrix.png').is_file())
            metrics = pd.read_csv(root / 'outputs/reports/metrics.csv')
            self.assertEqual(len(metrics), 2)


if __name__ == '__main__':
    unittest.main()
