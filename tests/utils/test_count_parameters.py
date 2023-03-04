from src.utils.count_parameters import count_parameters as f

test_file = "logs/size small/runs/2023-02-26_03-21-51/parameters.ckpt"
def test_count_parameters():
    num_p = f(test_file)
    assert num_p is not None

test_count_parameters()