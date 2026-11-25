
import numpy as np
import pydicom

from PIL import Image
from pydicom.pixels import (
    apply_modality_lut,
    apply_voi_lut
)


def dicom_to_image(ds):
    """Convertit les pixels DICOM en image Pillow."""

    pixels = ds.pixel_array
    photometrie = ds.PhotometricInterpretation

    # Cas 1 : image couleur RGB
    if photometrie == "RGB":
        if pixels.dtype != np.uint8 or pixels.ndim != 3:
            raise ValueError("Image RGB non prise en charge")

        return Image.fromarray(pixels).convert("RGB")

    # Cas 2 : image en niveaux de gris
    if photometrie not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError("Photométrie non prise en charge")

    # Application des transformations DICOM
    pixels = apply_modality_lut(pixels, ds)
    pixels = apply_voi_lut(pixels, ds, index=0)

    # Conversion des valeurs sur 8 bits
    pixels = pixels.astype(np.float32)

    minimum = pixels.min()
    maximum = pixels.max()

    if maximum > minimum:
        pixels = (pixels - minimum) / (maximum - minimum)
        pixels = pixels * 255
    else:
        pixels = np.zeros_like(pixels)

    # Inversion des intensités si nécessaire
    if photometrie == "MONOCHROME1":
        pixels = 255 - pixels

    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    return Image.fromarray(pixels).convert("RGB")


def lire_metadonnees(ds):
    """Extrait les métadonnées utiles au POC."""

    return {
        "modalite": str(ds.get("Modality", "")) or None,
        "region_anatomique": str(ds.get("BodyPartExamined", "")) or None,
        "description": str(ds.get("StudyDescription", "")) or None,
    }
