"""Container protocol. All user code runs inside the container's OS isolation."""
import base64
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path('/workspace')
MAX_BYTES = 20 * 1024 * 1024
request = json.loads(sys.stdin.buffer.read(MAX_BYTES * 2))
for item in request.get('files', []):
    target = ROOT / item['path']
    if not target.resolve().is_relative_to(ROOT) or '..' in target.parts:
        raise ValueError('Invalid input path')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(item['data'], validate=True))

language = request['language']
command = {'python': ['python', '-I'], 'javascript': ['node'], 'shell': ['/bin/sh']}[language]
suffix = {'python': '.py', 'javascript': '.js', 'shell': '.sh'}[language]
script = pathlib.Path('/tmp/program' + suffix)
script.write_text(request['code'], encoding='utf-8')
with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
    try:
        result = subprocess.run(command + [str(script)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                timeout=request.get('timeoutMs', 60000) / 1000, start_new_session=True)
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
    out.seek(0)
    err.seek(0)
    stdout = out.read(100000).decode('utf-8', errors='replace')
    stderr = err.read(100000).decode('utf-8', errors='replace')

files = []
total = 0
for directory, dirs, names in os.walk(ROOT, followlinks=False):
    dirs[:] = [name for name in dirs if not pathlib.Path(directory, name).is_symlink()]
    for name in sorted(names):
        file = pathlib.Path(directory, name)
        if file.is_symlink() or not file.is_file():
            continue
        size = file.stat().st_size
        if total + size > MAX_BYTES or len(files) >= 100:
            raise ValueError('Output exceeds 100 files or 20 MiB; no workspace changes will be committed')
        data = file.read_bytes()
        total += len(data)
        files.append({'path': file.relative_to(ROOT).as_posix(), 'data': base64.b64encode(data).decode('ascii')})
print(json.dumps({'exitCode': exit_code, 'stdout': stdout, 'stderr': stderr, 'files': files}))
