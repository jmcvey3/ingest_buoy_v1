import xarray as xr
from pathlib import Path
from tsdat import PipelineConfig, assert_close


def test_lidar_pipeline():
    config_path = Path("pipelines/lidar/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/lidar/test/data/input/lidar.z06.00.20201201.000000.sta.7z"
    expected_file = "pipelines/lidar/test/data/expected/morro.buoy_z06-lidar-10m.a1.20201201.001000.nc"

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)
