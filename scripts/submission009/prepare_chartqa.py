"""Fetch the complete pinned Trident ChartQA files and verify HF Git/LFS hashes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
REV='15c41335843f8880e24778843b462234b6427e4f'


def main():
    root=Path('datasets/session009/trident-chartqa')
    root.mkdir(parents=True,exist_ok=True)
    tree=json.loads(Path('evidence/session009/trident-chartqa-filetree.json').read_text())
    def one(row):
        path=root/Path(row['path']).name
        url=f"https://huggingface.co/datasets/OpenBMB/Trident-Eval/resolve/{REV}/{row['path']}"
        with urllib.request.urlopen(url,timeout=120) as stream:
            data=stream.read(row['size']+1)
        assert len(data)==row['size']
        sha=hashlib.sha256(data).hexdigest()
        if 'lfs' in row:assert sha==row['lfs']['oid']
        else:assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['oid']
        with path.open('xb') as stream:stream.write(data)
        return {'path':path.name,'bytes':len(data),'sha256':sha,'source_url':url}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(one,tree))
    (root/'DOWNLOAD-MANIFEST.json').write_text(json.dumps({'dataset_revision':REV,'files':files,'partial_subset':False},indent=2))
    print(json.dumps(files),flush=True)


if __name__=='__main__':main()
