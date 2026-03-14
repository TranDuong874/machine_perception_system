from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from server.services.camera_calibration_service import CameraCalibrationService


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = REPO_ROOT / "dataset" / "dataset-corridor1_512_16" / "dso" / "camchain.yaml"


class CameraCalibrationServiceTest(unittest.TestCase):
    def test_rectify_uses_dataset_calibration(self) -> None:
        service = CameraCalibrationService(
            enabled=True,
            calibration_file=CALIBRATION_FILE,
            camera_key="cam0",
            rectification_balance=0.0,
        )
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        image[:, :, 0] = np.tile(np.arange(512, dtype=np.uint8), (512, 1))
        image[:, :, 1] = np.tile(np.arange(512, dtype=np.uint8).reshape(512, 1), (1, 512))

        rectified = service.rectify(image)

        self.assertTrue(rectified.rectified)
        self.assertEqual(rectified.image_bgr.shape, image.shape)
        self.assertEqual(rectified.valid_mask.shape, image.shape[:2])
        self.assertEqual(rectified.intrinsics.shape, (3, 3))
        self.assertIn("camchain.yaml:cam0", rectified.calibration_id)
        self.assertGreater(np.abs(rectified.image_bgr.astype(np.int32) - image.astype(np.int32)).sum(), 0)

    def test_disabled_service_returns_identity_frame(self) -> None:
        service = CameraCalibrationService(
            enabled=False,
            calibration_file=None,
            camera_key="cam0",
            rectification_balance=0.0,
        )
        image = np.full((32, 48, 3), 127, dtype=np.uint8)

        rectified = service.rectify(image)

        self.assertFalse(rectified.rectified)
        self.assertEqual(rectified.image_bgr.shape, image.shape)
        self.assertTrue(np.array_equal(rectified.image_bgr, image))
        self.assertTrue(rectified.valid_mask.all())


if __name__ == "__main__":
    unittest.main()
