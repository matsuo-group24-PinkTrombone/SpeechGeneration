import glob

import pytest
from pydub import AudioSegment

target_sample_files = "data/sample_generated_sounds/*.wav"

test_files = glob.glob(target_sample_files)


@pytest.mark.parametrize("test_file", test_files)
def test_make_random_dataset(test_file: str):
    sound = AudioSegment.from_file(test_file)
    assert 2.0 <= len(sound) / 1000 <= 3.0
