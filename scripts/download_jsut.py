import os
import pathlib
import subprocess

url = "http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip"
output_dir = pathlib.Path(__file__).parents[1].joinpath("data")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_zip_dir = output_dir.joinpath("jsut.zip")
res = subprocess.call(f'curl "{url}" -o "{output_zip_dir}"', shell=True)
res = subprocess.call(f'unzip "{output_zip_dir}" -d "{output_dir}"', shell=True)
