# YOLOv5 && yolov8
"""
Download utils
"""

import logging
import os
import subprocess
import urllib
from itertools import repeat
from pathlib import Path
from multiprocessing.pool import ThreadPool
import requests
import torch

import contextlib
from urllib import parse, request
from zipfile import BadZipFile, ZipFile, is_zipfile


from tqdm import tqdm

GITHUB_ASSET_NAMES = [f'yolov8{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '6', '-cls', '-seg')] + \
                     [f'yolov5{size}u.pt' for size in 'nsmlx'] + \
                     [f'yolov3{size}u.pt' for size in ('', '-spp', '-tiny')]
GITHUB_ASSET_STEMS = [Path(k).stem for k in GITHUB_ASSET_NAMES]

def is_url(url, check_online=True):
    # Check if online file exists
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check_online else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False

def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX')):
    """
    Unzip a *.zip file to path/, excluding files containing strings in exclude list
    Replaces: ZipFile(file).extractall(path=path)
    """
    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent  # default path
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # list all archived filenames in the zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)
        return zipObj.namelist()[0]  # return unzip dir

def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes

def url_getsize(url='https://ultralytics.com/images/bus.jpg'):
    # Return downloadable file size in bytes
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get('content-length', -1))

def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        if file.exists():
            file.unlink()  # remove partial downloads
        LOGGER.info(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -# -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # curl download, retry and resume on fail
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
        LOGGER.info('')


def attempt_download(file, repo='ultralytics/yolov5', release='v7.0'):
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v6.2', etc.
    from utils.general import LOGGER

    def github_assets(repository, version='latest'):
        # Return GitHub repo tag (i.e. 'v7.0') and assets (i.e. ['yolov5s.pt', 'yolov5m.pt', ...])
        if version != 'latest':
            version = f'tags/{version}'  # i.e. tags/v6.2
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()  # github api
        return response['tag_name'], [x['name'] for x in response['assets']]  # tag, assets

    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f'Found {url} locally at {file}')  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1E5)
            return file

        # GitHub assets
        assets = [f'yolov5{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '6', '-cls', '-seg')]  # default
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        if name in assets:
            url3 = 'https://drive.google.com/drive/folders/1EFQTEUeXWSFww0luse2jB9M1QNZQGwNl'  # backup gdrive mirror
            safe_download(
                file,
                url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                #url2=f'https://storage.googleapis.com/{repo}/{tag}/{name}',  # backup url (optional)
                min_bytes=1E5,
                error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/{tag} or {url3}')

    return str(file)
def attempt_download_asset(file, repo='ultralytics/assets', release='v0.0.0'):
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v6.2', etc.
    from yolo.utils import SETTINGS,LOGGER, checks, is_online # scoped for circular import
    from urllib import parse
    def github_assets(repository, version='latest'):
        # Return GitHub repo tag and assets (i.e. ['yolov8n.pt', 'yolov8s.pt', ...])
        if version != 'latest':
            version = f'tags/{version}'  # i.e. tags/v6.2
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()  # github api
        return response['tag_name'], [x['name'] for x in response['assets']]  # tag, assets

    # YOLOv3/5u updates
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ''))
    if file.exists():
        return str(file)
    elif (SETTINGS['weights_dir'] / file).exists():
        return str(SETTINGS['weights_dir'] / file)
    else:
        # URL specified
        name = Path(parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f'Found {url} locally at {file}')  # file already exists
            else:
                safe_download(url=url, file=file, min_bytes=1E5)
            return file

        # GitHub assets
        assets = GITHUB_ASSET_NAMES
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output(['git', 'tag']).decode().split()[-1]
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        if name in assets:
            safe_download(url=f'https://github.com/{repo}/releases/download/{tag}/{name}', file=file, min_bytes=1E5)

        return str(file)


def download(url, dir=Path.cwd(), unzip=True, delete=False, curl=False, threads=1, retry=3):
    # Multithreaded file download and unzip function, used in data.yaml for autodownload
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        with ThreadPool(threads) as pool:
            pool.map(
                lambda x: safe_download(
                    url=x[0], dir=x[1], unzip=unzip, delete=delete, curl=curl, retry=retry, progress=threads <= 1),
                zip(url, repeat(dir)))
            pool.close()
            pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=curl, retry=retry)