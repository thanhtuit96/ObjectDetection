from .ssd_detector import SSDDetector
from .mssd_detector import MSSDDetector
from .gssd_detector import GridSSDDetector

_DETECTION_META_ARCHITECTURES = {
    "SSDDetector": SSDDetector,
    "MSSDDetector": MSSDDetector,
    "GridSSDDetector": GridSSDDetector
}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
