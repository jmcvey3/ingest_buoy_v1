import xarray as xr
from pathlib import Path
from tsdat import PipelineConfig, assert_close

# DEVELOPER: Update paths to your configuration(s), test input(s), and expected test
# results files.
def test_metocean_pipeline():
    config_path = Path("pipelines/metocean/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/metocean/test/data/input/morro_data.csv"
    expected_file = (
        "pipelines/metocean/test/data/expected/abc.example.a1.20220424.000000.nc"
    )

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)
