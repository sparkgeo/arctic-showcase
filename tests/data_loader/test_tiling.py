from training.data_loader.tiling import chip_bounds, chip_starts


def test_chip_bounds_covers_the_full_grid_without_gaps() -> None:
    assert chip_starts(300, chip_size=256) == [0, 44]

    bounds = list(chip_bounds(sar_h=300, sar_w=512, chip_size=256))

    assert bounds == [
        (0, 256, 0, 256),
        (0, 256, 256, 512),
        (44, 300, 0, 256),
        (44, 300, 256, 512),
    ]
