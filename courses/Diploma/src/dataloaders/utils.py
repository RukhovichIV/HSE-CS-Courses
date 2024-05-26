import os
import pathlib
import sys
from urllib import request


def download_file(url_from: str, local_path: str) -> None:
    """
    Download a file to the given path.

    :param url_from: URL to download
    :param local_path: Where to download the content.
    """

    def _progress(count, block_size, total_size):
        progress_pct = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write(
            "\rDownloading {} to {} {:.1f}%".format(url_from, local_path, progress_pct)
        )
        sys.stdout.flush()

    if not os.path.isfile(local_path):
        opener = request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        request.install_opener(opener)
        pathlib.Path(os.path.dirname(local_path)).mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url_from, local_path, _progress)
        sys.stdout.write("\n")
        sys.stdout.flush()
        file_info = os.stat(f)
        print(
            f"Successfully downloaded {os.path.basename(local_path)} {file_info.st_size} bytes."
        )
    else:
        file_info = os.stat(local_path)
        print(f"File already exists: {local_path} {file_info.st_size} bytes.")
