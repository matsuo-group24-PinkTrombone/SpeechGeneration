import subprocess
import pathlib
import os
url = "http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip"
output_dir = pathlib.Path(__file__).parents[1].joinpath("data/jsut")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_dir = output_dir.joinpath("jsut.zip")
res = subprocess.call(f'curl "{url}" -o {output_dir}', shell=True)
