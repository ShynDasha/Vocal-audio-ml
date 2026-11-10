"""Prepare the singer-organized VocalSet 1.1 archive without copying audio."""
import argparse
import csv
import hashlib
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from config import ARCHIVE, RAW_DIR, MANIFEST

URL = 'https://zenodo.org/records/1442513/files/VocalSet11.zip?download=1'
EXPECTED_MD5 = '0c09396242f946e7111ad7d8fc649b81'
SINGER = re.compile(r'(?:male|female)\d+', re.IGNORECASE)
# Observed folder typo: female2/arpeggios/vibrado contains vibrato recordings.
TECHNIQUE_ALIASES = {'vibrado': 'vibrato'}


def verify_archive(path, allow_different_checksum=False):
    digest = hashlib.md5()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != EXPECTED_MD5:
        if not allow_different_checksum:
            raise ValueError(
                f'MD5 архіву: {actual}; очікувана на Zenodo: {EXPECTED_MD5}. '
                'Це не обов’язково означає пошкодження. Якщо ви довіряєте джерелу архіву, '
                'повторіть запуск із --allow-different-checksum для повної перевірки CRC. '
                'Ця перевірка не підтверджує тотожність офіційному датасету.'
            )
        print(f'MD5 {actual} відрізняється від Zenodo. Перевірка CRC усіх файлів...', flush=True)
        with zipfile.ZipFile(path) as archive:
            bad_file = archive.testzip()
        if bad_file is not None:
            raise ValueError(f'Пошкоджений файл у ZIP: {bad_file}')
        print('Цілісність ZIP підтверджено; тотожність офіційному архіву не підтверджена.', flush=True)


def extract_audio(archive, raw_dir):
    """Check every member; repair missing or truncated files on repeat runs."""
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            path = PurePosixPath(item.filename)
            if path.suffix.lower() != '.wav' or '__MACOSX' in path.parts or path.name.startswith('._'):
                continue
            if path.is_absolute() or '..' in path.parts or '\\' in item.filename:
                raise ValueError(f'Небезпечний шлях в архіві: {item.filename}')
            dest = raw_dir.joinpath(*path.parts)
            if not dest.resolve().is_relative_to(raw_dir.resolve()):
                raise ValueError(f'Шлях виходить за межі raw: {dest}')
            if dest.is_file() and dest.stat().st_size == item.file_size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix('.wav.part')
            with source.open(item) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            temporary.replace(dest)


def build_manifest(raw_dir, manifest_path):
    rows, skipped = [], []
    for path in sorted(raw_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() != '.wav':
            continue
        parts = path.relative_to(raw_dir).parts
        if '__MACOSX' in parts or path.name.startswith('._'):
            continue
        index = next((i for i, part in enumerate(parts[:-1]) if SINGER.fullmatch(part)), None)
        if index is None or len(parts) != index + 4:
            skipped.append('/'.join(parts))
            continue
        rows.append(dict(path=path.relative_to(raw_dir).as_posix(),
                         singer=parts[index].lower(), sample_type=parts[index + 1],
                         technique=TECHNIQUE_ALIASES.get(parts[index + 2], parts[index + 2])))
    if skipped:
        raise ValueError('Нерозпізнана структура WAV-файлів; manifest не оновлено:\n' + '\n'.join(skipped[:10]))
    if not rows:
        raise ValueError('Не знайдено WAV у структурі виконавець/тип/техніка/файл.')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix('.csv.part')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'technique', 'singer', 'sample_type'])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest_path)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Завантажити архів, якщо його немає')
    parser.add_argument('--allow-different-checksum', action='store_true',
                        help='Дозволити наявний ZIP з іншою MD5 після повної перевірки CRC')
    args = parser.parse_args()
    if not ARCHIVE.exists():
        if not args.download:
            parser.error(f'Додайте VocalSet11.zip у {ARCHIVE.parent} або використайте --download')
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        temporary = ARCHIVE.with_suffix('.zip.part')
        print('Завантаження VocalSet11.zip (~2.1 GB)...', flush=True)
        urllib.request.urlretrieve(URL, temporary)
        verify_archive(temporary)
        temporary.replace(ARCHIVE)
    else:
        print('Перевірка контрольної суми архіву...', flush=True)
        verify_archive(ARCHIVE, allow_different_checksum=args.allow_different_checksum)
    extract_audio(ARCHIVE, RAW_DIR)
    rows = build_manifest(RAW_DIR, MANIFEST)
    from collections import Counter
    print(f'Записів: {len(rows)}; виконавців: {len({r["singer"] for r in rows})}')
    print('Техніки:', dict(sorted(Counter(r['technique'] for r in rows).items())))
    print('Manifest:', MANIFEST)


if __name__ == '__main__':
    main()
