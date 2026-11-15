import tempfile
from pathlib import Path
from unittest.mock import patch

from app import app
from services import pair_moves, tag_editor


def make_pair(folder, name='a'):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{name}.png').write_bytes(b'image')
    (folder / f'{name}.txt').write_text('long hair, blue eyes', encoding='utf-8')


def test_nested_stage_return_and_preview_conflicts():
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        root = project / 'dataset'
        temp = root / '_temp'
        make_pair(root / 'group' / 'nested')
        temp.mkdir()
        client = app.test_client()
        for src, dst, expected in [('group/nested/a.png', '_temp', '_temp/group/nested/a.png'),
                                   ('_temp/group/nested/a.png', '', 'group/nested/a.png')]:
            body = {'folder': str(project), 'srcs': [src], 'dst': dst}
            preview = client.post('/api/tags/move-files/preview', json=body).get_json()
            assert preview['ok'] and preview['paired_captions'] == 1
            result = client.post('/api/tags/move-files', json={**body, 'preview_token': preview['preview_token']}).get_json()
            assert result['ok'] and result['moved'][0]['rel'] == expected
            assert (root / expected).with_suffix('.txt').read_text() == 'long hair, blue eyes'
            assert not (root / src).exists()


def test_case_insensitive_orphan_caption_conflict_and_rollback():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        make_pair(root / 'nested')
        temp = root / '_temp' / 'nested'
        temp.mkdir(parents=True)
        (temp / 'A.TXT').write_text('keep this', encoding='utf-8')
        preview = pair_moves.preview(root, [root / 'nested/a.png'], root / '_temp')
        assert preview['conflicts'] == 1
        result, status = pair_moves.move(root, 'nested/a.png', '_temp')
        assert status == 200 and result['rel'] == '_temp/nested/a_1.png'
        assert (temp / 'A.TXT').read_text() == 'keep this'
        real_move = pair_moves.shutil.move
        def fail_caption(src, dst):
            if str(src).endswith('.txt'):
                raise PermissionError('caption locked')
            return real_move(src, dst)
        with patch('services.pair_moves.shutil.move', side_effect=fail_caption):
            result, status = pair_moves.move(root, '_temp/nested/a_1.png', '')
        assert status == 500 and 'caption locked' in result['error']
        assert (temp / 'a_1.png').exists() and (temp / 'a_1.txt').exists()
        assert not (root / 'nested/a_1.png').exists()


def test_partial_moves_and_missing_sidecar_do_not_overwrite_caption():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / 'dataset'
        make_pair(root)
        (root / '_temp').mkdir()
        client = app.test_client()
        response = client.post('/api/tags/move-files', json={'folder': str(root), 'srcs': ['a.png', 'missing.png'], 'dst': '_temp'}).get_json()
        assert not response['ok'] and len(response['moved']) == len(response['errors']) == 1
        (root / 'b.png').write_bytes(b'image')
        (root / '_temp/b.txt').write_text('unrelated')
        result, status = pair_moves.move(root, 'b.png', '_temp')
        assert status == 200 and result['rel'] == '_temp/b_1.png'
        assert result['warnings'] and (root / '_temp/b.txt').read_text() == 'unrelated'
        bad, status = pair_moves.move(root, '_temp/a.txt', '')
        assert status == 404 and not bad['ok']


def test_non_file_sidecar_is_rejected_before_the_image_moves():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / '_temp').mkdir()
        (root / 'a.png').write_bytes(b'image')
        (root / 'a.txt').mkdir()
        result, status = pair_moves.move(root, 'a.png', '_temp')
        assert status == 500 and 'regular file' in result['error']
        assert (root / 'a.png').is_file() and (root / 'a.txt').is_dir()
        assert not (root / '_temp/a.png').exists()


def test_preview_all_modes_writes_neither_captions_nor_backups():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / 'dataset'
        make_pair(root)
        make_pair(root / '_temp')
        caption = root / '_temp/a.txt'
        for mode, tags in [('insert', 'outdoors'), ('delete', 'long hair'), ('replace', 'long hair->short hair'), ('dedup', '')]:
            caption.write_text('long hair, blue eyes, long hair')
            before = caption.read_bytes()
            _, log, meta = tag_editor.handle({'folder': str(root), 'mode': mode, 'tags': tags, 'exts': '.png', 'backup': '1', 'dry_run': '1'}, {})
            assert meta['ok'] and meta['counts']['changed'] == 1, (mode, log, meta)
            assert caption.read_bytes() == before
            assert not caption.with_suffix('.txt.bak').exists()
            assert (root / 'a.txt').read_text() == 'long hair, blue eyes'


def test_staged_tag_frequency_counts_caption_files_and_browser_excludes_temp():
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        root = project / 'dataset'
        make_pair(root)
        make_pair(root / '_temp')
        (root / '_temp/a.txt').write_text('duplicate, duplicate, very long tag name')
        (root / '_temp/missing.png').write_bytes(b'image')
        make_pair(root / '_temp', 'empty')
        (root / '_temp/empty.txt').write_text('')
        scan = tag_editor.scan_tags(root / '_temp', ['.png'], recursive=True)
        assert scan['total_images'] == 3 and scan['tag_files'] == 2
        assert dict((item['tag'], item['count']) for item in scan['tags']) == {'duplicate': 1, 'very long tag name': 1}
        listed = tag_editor.list_dataset_images(project, ['.png'], True, 80, 40, include_temp=False)
        assert [image['rel'] for image in listed['images']] == ['a.png']
