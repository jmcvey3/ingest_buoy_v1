import xarray as xr
from pathlib import Path
from tsdat import PipelineConfig, assert_close


def test_imu_pipeline():
    config_path = Path("pipelines/imu/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/imu/test/data/input/buoy.z06.00.20201201.000000.imu.bin"
    expected_file = (
        "pipelines/imu/test/data/expected/morro.buoy_z06-imu.a1.20201201.000011.nc"
    )

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)
