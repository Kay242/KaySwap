import copy
import logging
import os
import pickle
import struct
from ast import literal_eval
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple, Any
from zlib import crc32, decompress

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla
from numpy import ndarray

from lib.align import AlignedFace
from lib.png_meta import FaceswapMetaKeys, DeepFaceLabMetaKeys, FaceswapMasks, FACESWAP_MASKS, META_KEYS_TO_ENCODE

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class SegIEPolys:
    def __init__(self):
        self.polys = []

    def identical(self, b):
        polys_len = len(self.polys)
        o_polys_len = len(b.polys)
        if polys_len != o_polys_len:
            return False

        return all([a_poly.identical(b_poly) for a_poly, b_poly in zip(self.polys, b.polys)])

    def add_poly(self, ie_poly_type):
        poly = SegIEPoly(ie_poly_type)
        self.polys.append(poly)
        return poly

    def remove_poly(self, poly):
        if poly in self.polys:
            self.polys.remove(poly)

    def has_polys(self):
        return len(self.polys) != 0

    def get_poly(self, id):
        return self.polys[id]

    def get_polys(self):
        return self.polys

    def get_pts_count(self):
        return sum([poly.get_pts_count() for poly in self.polys])

    def sort(self):
        poly_by_type = {SegIEPolyType.EXCLUDE: [], SegIEPolyType.INCLUDE: []}

        for poly in self.polys:
            poly_by_type[poly.type].append(poly)

        self.polys = poly_by_type[SegIEPolyType.INCLUDE] + poly_by_type[SegIEPolyType.EXCLUDE]

    def __iter__(self):
        for poly in self.polys:
            yield poly

    def overlay_mask(self, mask):
        h, w, c = mask.shape
        white = (1,) * c
        black = (0,) * c
        for poly in self.polys:
            pts = poly.get_pts().astype(np.int32)
            if len(pts) != 0:
                cv2.fillPoly(mask, [pts], white if poly.type == SegIEPolyType.INCLUDE else black)

    def dump(self):
        return {'polys': [poly.dump() for poly in self.polys]}

    def mult_points(self, val):
        for poly in self.polys:
            poly.mult_points(val)

    @staticmethod
    def load(data=None):
        ie_polys = SegIEPolys()
        if data is not None:
            if isinstance(data, list):
                # Backward comp
                ie_polys.polys = [SegIEPoly(type=type, pts=pts) for (type, pts) in data]
            elif isinstance(data, dict):
                ie_polys.polys = [SegIEPoly(**poly_cfg) for poly_cfg in data['polys']]

        ie_polys.sort()

        return ie_polys


class DflPng:
    def __init__(self, file_path: Path):
        self.filename = file_path
        self.metadata: dict = {}
        self._decoded_image = None
        self._shape = None
        self._face_type = "whole_face"

    @staticmethod
    def load(filename: Path):
        """
        This function should be called first and is the primary method of instantiating a DflPng class. This will load
        the provided image, retrieve all the data and metadata and arrange it in a usable method.

        Parameters
        ----------
        filename : Path
            Filepath to image as a Path object.
        Returns
        -------
        dflpng : DflPng
            Instantiated DflPng object
        """
        dflpng: DflPng = DflPng(filename)
        dflpng._decoded_image, dflpng.metadata = read_image(str(filename), raise_error=True, with_metadata=True)

        if dflpng.metadata:
            if DeepFaceLabMetaKeys.extracted_landmarks not in dflpng.metadata:
                fs_landmarks_meta = dflpng.metadata[FaceswapMetaKeys.alignments][FaceswapMetaKeys.landmarks]
                extracted_landmarks = np.asarray(fs_landmarks_meta.copy())
                source_image_dimensions: tuple = dflpng.metadata[FaceswapMetaKeys.source][FaceswapMetaKeys.frame_dims]
                extracted_landmarks[:, 0] *= dflpng.decoded_image.shape[1] / source_image_dimensions[1]
                extracted_landmarks[:, 1] *= dflpng.decoded_image.shape[0] / source_image_dimensions[0]
                dflpng.metadata[DeepFaceLabMetaKeys.extracted_landmarks] = extracted_landmarks

            for loaded_key in dflpng.metadata.keys():
                # Some objects needed to be saved as bytes objects and must be loaded via pickle
                if loaded_key in META_KEYS_TO_ENCODE:
                    loaded_value = dflpng.metadata[loaded_key]
                    if type(loaded_value) is bytes:
                        dflpng.metadata[loaded_key] = pickle.loads(loaded_value)
        return dflpng

    @property
    def source_shape(self) -> Tuple[int, int]:
        """

        Returns
        -------
        tuple
            tuple of the shape of the source image as (height, width)
        """
        source_height, source_width = self.metadata[FaceswapMetaKeys.source][FaceswapMetaKeys.frame_dims]
        return source_height, source_width

    @property
    def face_centered_image(self):
        #al_face = AlignedFace(self.source_landmarks,
        #                      self.decoded_image,
        #                      size=self.shape[0],
        #                      centering="face",
        #                      is_aligned=True)
        #return al_face.face
        return self.decoded_image

    @property
    def center_face_masked(self, masker: FaceswapMasks = FaceswapMasks.bisenet_fp_face, debug=True):
        stored_mask, _ = self.faceswap_mask(masker, return_mask_parent_dict=True)
        extracted_image_shape = self.shape[:2]
        face = self.face_centered_image

        extracted_frame_mask = cv2.resize(stored_mask, extracted_image_shape, interpolation=cv2.INTER_AREA)
        masked_face = np.concatenate((face, np.expand_dims(extracted_frame_mask, axis=-1)), axis=-1)

        if debug:
            f, sub_plot = plt.subplots(1, 2)
            sub_plot[0].imshow(face)
            sub_plot[1].imshow(masked_face)
            plt.show()

        return masked_face

    @staticmethod
    def zoomed_affinity(affine_matrix,
                        mask_storage_size,
                        face_pixel_edge_size,
                        inverse=False) -> np.ndarray:
        if inverse:
            zoom = mask_storage_size / face_pixel_edge_size
        else:
            zoom = face_pixel_edge_size / mask_storage_size
        zoom_mat = np.array([[zoom, 0, 0.], [0, zoom, 0.]])

        zoomed_affine_matrix = np.dot(zoom_mat, np.concatenate((affine_matrix, np.array([[0., 0., 1.]]))))

        return zoomed_affine_matrix

    def _retrieve_pickled_transformation(self,
                                         source_image: np.ndarray,
                                         face_pixel_edge_size: int = 128,
                                         masker: FaceswapMasks = FaceswapMasks.bisenet_fp_face):
        """
        FOR DEBUGGING ONLY.
        Parameters
        ----------
        source_image
        face_pixel_edge_size
        masker

        Returns
        -------

        """
        stored_mask, mask_meta_dict = self.faceswap_mask(masker, return_mask_parent_dict=True)
        mask_stored_size = mask_meta_dict["stored_size"]
        stored_affine_matrix = np.asarray(mask_meta_dict["affine_matrix"])

        source_height, source_width = self.source_shape
        extracted_image_shape = self.shape[:2]

        resized_face_mask = cv2.resize(stored_mask, extracted_image_shape, interpolation=cv2.INTER_AREA)

        adjusted_affine_matrix = self.zoomed_affinity(stored_affine_matrix,
                                                      mask_stored_size,
                                                      extracted_image_shape[0],
                                                      inverse=False)

        self.set_affine_matrix(adjusted_affine_matrix)
        self.save()

        # cropped to extracted face zone
        transformed_image = cv2.warpAffine(source_image[..., 2::-1],
                                           adjusted_affine_matrix,
                                           extracted_image_shape,
                                           flags=cv2.INTER_AREA,
                                           borderMode=cv2.BORDER_CONSTANT)
        masked_extracted_face = np.concatenate((transformed_image, resized_face_mask[..., None]), axis=-1)

        dst_frame = np.zeros((source_height, source_width, 3), dtype="uint8")
        adjusted_affine_matrix2 = self.zoomed_affinity(stored_affine_matrix,
                                                       mask_stored_size,
                                                       face_pixel_edge_size,
                                                       inverse=False)
        # cv2.wrapAffine needs shape given as (width, height) to return an array with shape (height, width). Basically
        # it needs the transposed shape.
        mask: np.ndarray = cv2.warpAffine(stored_mask,
                                          adjusted_affine_matrix2,
                                          (source_width, source_height),
                                          dst_frame,
                                          flags=cv2.WARP_INVERSE_MAP,
                                          borderMode=cv2.BORDER_CONSTANT)

        masked_face_original_source_frame = np.concatenate((source_image, np.expand_dims(mask, axis=-1)), axis=-1)
        masked_face_empty_source_frame = np.concatenate((dst_frame, np.expand_dims(mask, axis=-1)), axis=-1)

        f, sub_plot = plt.subplots(2, 2)
        sub_plot[0, 0].imshow(transformed_image)
        sub_plot[0, 1].imshow(masked_extracted_face)
        sub_plot[1, 0].imshow(masked_face_original_source_frame)
        sub_plot[1, 1].imshow(masked_face_empty_source_frame)
        plt.show()

    def save(self) -> None:
        # Initially assign object by reference in case nothing needs to be done
        metadata_to_save = self.metadata.copy()

        for loaded_key in self.metadata.keys():
            if loaded_key in META_KEYS_TO_ENCODE:
                loaded_value = self.metadata[loaded_key]
                # At this point the data SHOULD NOT be in bytes form.
                if type(loaded_value) is not bytes:
                    # xseg polys need to be saved a bytes object to avoid issues with ast.literal_eval
                    # later when the meta_data is re-read by the png_read_meta function.
                    if loaded_key in [DeepFaceLabMetaKeys.seg_ie_polys, DeepFaceLabMetaKeys.xseg_mask]:
                        # regular copy top level dict entries so changes to top level entries are not reflected in
                        # the original object (self.metadata). However this strictly applies to the top level,
                        # for example, if one top level element in a dict is a list value and an element in that list
                        # is changed in the copy it will still be reflected in the original. Whereas if you replace
                        # that list entirely with another list or something else it won't be reflected in the
                        # original. Since we are only interested in replacing the seg_polys value with a bytes
                        # representation without affecting the original. A deepcopy is needed (but just for that
                        # element in the dict).
                        metadata_to_save[loaded_key] = copy.deepcopy(loaded_value)
                        metadata_to_save[loaded_key] = pickle.dumps(loaded_value)
                    else:
                        metadata_to_save[loaded_key] = pickle.dumps(loaded_value)

        update_existing_metadata(str(self.filename), metadata_to_save)

    @property
    def has_metadata(self):
        return len(self.metadata.keys()) != 0

    @property
    def decoded_image(self) -> np.ndarray:
        if self._decoded_image is None:
            self._decoded_image = cv2.imread(str(self.filename), flags=cv2.IMREAD_UNCHANGED)
        return self._decoded_image

    @property
    def shape(self) -> tuple:
        if self._shape is None:
            self._shape = self.decoded_image.shape
        return self._shape

    @property
    def face_type(self) -> str:
        return self._face_type

    def set_face_type(self, face_type) -> None:
        self._face_type = face_type

    @property
    def extracted_landmarks(self) -> np.ndarray:
        extracted_landmarks = self.metadata.get(DeepFaceLabMetaKeys.extracted_landmarks, None)
        return extracted_landmarks

    def set_extracted_landmarks(self, extracted_landmarks: np.ndarray) -> None:
        """
        Set the 68 point-based landmarks for the extracted image.
        @param extracted_landmarks: numpy array of the landmark coordinates in the extracted image.
        @return: None.
        """
        self.metadata[DeepFaceLabMetaKeys.extracted_landmarks] = extracted_landmarks

    @property
    def source_filename(self) -> str:
        faceswap_source_dict = self.metadata.get(FaceswapMetaKeys.source, None)
        source_file_name = faceswap_source_dict.get(FaceswapMetaKeys.source_filename, None)
        return source_file_name

    def set_source_filename(self, source_filename: str) -> None:
        faceswap_source_dict = self.metadata[FaceswapMetaKeys.source]
        faceswap_source_dict[FaceswapMetaKeys.source_filename] = source_filename

    @property
    def source_rect(self) -> np.ndarray:
        return self.metadata.get(DeepFaceLabMetaKeys.source_rect, None)

    def set_source_rect(self, source_rect: np.ndarray) -> None:
        self.metadata[DeepFaceLabMetaKeys.source_rect] = source_rect

    @property
    def source_landmarks(self) -> np.ndarray:
        faceswap_dict: dict = self.metadata.get(FaceswapMetaKeys.alignments, None)
        source_landmarks = faceswap_dict.get(FaceswapMetaKeys.landmarks, None)
        return np.array(source_landmarks) if source_landmarks else None

    def set_source_landmarks(self, source_landmarks: np.ndarray) -> None:
        """

        @param source_landmarks: A numpy array of the 68 point-based landmarks for the source image.
        @return: None.
        """
        faceswap_dict: dict = self.metadata[FaceswapMetaKeys.alignments]
        faceswap_dict[FaceswapMetaKeys.landmarks] = source_landmarks

    @property
    def affine_matrix(self) -> np.ndarray:
        mat = self.metadata.get(DeepFaceLabMetaKeys.image_to_face_mat, None)
        return mat

    def set_affine_matrix(self, affine_matrix: np.ndarray) -> None:
        self.metadata[DeepFaceLabMetaKeys.image_to_face_mat] = affine_matrix

    @property
    def has_xpolys(self) -> bool:
        return self.metadata.get(DeepFaceLabMetaKeys.seg_ie_polys) is not None

    @property
    def xpolys(self) -> SegIEPolys:
        """

        @return: A SegIEPolys class.
        """
        dict_val = self.metadata.get(DeepFaceLabMetaKeys.seg_ie_polys, None)
        if dict_val is not None:
            d = SegIEPolys.load(dict_val)
        else:
            d = SegIEPolys()

        return d

    def set_seg_ie_polys(self, seg_ie_polys: SegIEPolys) -> None:
        """
        Add polys to image metadata.

        @param seg_ie_polys: Object of type SegIEPolys.
        @return: None
        """
        if seg_ie_polys is not None:
            if not isinstance(seg_ie_polys, SegIEPolys):
                raise ValueError('seg_ie_polys should be instance of SegIEPolys')

            if seg_ie_polys.has_polys():
                seg_ie_polys = seg_ie_polys.dump()
            else:
                seg_ie_polys = None

        self.metadata[DeepFaceLabMetaKeys.seg_ie_polys] = seg_ie_polys

    @property
    def has_xmask(self):
        return self.metadata.get(DeepFaceLabMetaKeys.xseg_mask, None) is not None

    @property
    def compressed_xmask(self) -> Optional[ndarray]:
        """

        @return: numpy array of an Xseg compressed mask or None if no mask is present.
        """
        mask_buf = self.metadata.get(DeepFaceLabMetaKeys.xseg_mask, None)
        if mask_buf is None:
            return None

        return mask_buf

    def faceswap_mask(self, mask_name: FaceswapMasks, return_mask_parent_dict=False) -> Tuple[Any, Any]:
        """

        Parameters
        ----------
        mask_name
        return_mask_parent_dict

        Returns
        -------

        """
        try:
            masks_in_metadata = self.metadata[FaceswapMetaKeys.alignments][FaceswapMetaKeys.alignments_mask]
        except KeyError as e:
            raise Exception("Missing faceswap alignments/masks in metadata. Double check that the image {0} has been"
                            " correctly extracted from faceswap.".format(self.filename.name)) from e

        if mask_name not in FACESWAP_MASKS:
            raise KeyError("Provided mask name, {0}, is either incorrect or not currently supported.".format(mask_name))
        try:
            mask_parent_dict = masks_in_metadata[mask_name]
        except KeyError:
            raise KeyError("Provided mask name, {0}, is missing from metadata".format(mask_name))
        mask_compressed_array = mask_parent_dict["mask"]
        stored_size = mask_parent_dict["stored_size"]
        dims = (stored_size, stored_size, 1)

        decompressed_mask = np.frombuffer(decompress(mask_compressed_array), dtype="uint8").reshape(dims)
        if return_mask_parent_dict:
            return decompressed_mask, mask_parent_dict
        else:
            return decompressed_mask

    @property
    def xmask(self) -> Optional[ndarray]:
        """

        Returns
        -------
        ndarray
            Returns an array of the decompressed Xseg mask or None if no mask is available.
        """
        mask_buf = self.metadata.get(DeepFaceLabMetaKeys.xseg_mask, None)
        if mask_buf is None:
            return None

        img = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)

        if len(img.shape) == 2:
            img = img[..., None]

        return np.asarray(img.astype(np.float32) / 255.0)

    def set_xmask(self, mask_a: np.ndarray) -> None:
        if mask_a is None:
            self.metadata[DeepFaceLabMetaKeys.xseg_mask] = None
            return

        mask_a = normalize_channels(mask_a, 1)
        img_data = np.clip(mask_a * 255, 0, 255).astype(np.uint8)

        ret, buf = cv2.imencode('.png', img_data)

        if not ret:
            raise Exception("set_xseg_mask: unable to generate image data for set_xseg_mask")

        self.metadata[DeepFaceLabMetaKeys.xseg_mask] = buf

    def convert_xmask_to_fs_mask(self, mask_storage_size=128):
        """

        @param mask_storage_size: The size that the mask array is stored as, default 128.
        @return: A faceswap readable and understandable mask in BGR format.
        """
        xseg_mask = self.xmask
        if not self.xmask.any():
            return None

        mask = cv2.resize(xseg_mask, self.shape[:2], interpolation=cv2.INTER_AREA)
        normalized_mask = normalize_channels(mask, 1)
        fs_mask_from_xseg = (1 - normalized_mask) * 0.5 + normalized_mask
        fs_mask_from_xseg_clipped = np.clip(fs_mask_from_xseg * 255, 0, 255).astype(np.uint8)
        fs_stored_mask = cv2.resize(fs_mask_from_xseg_clipped, (mask_storage_size, mask_storage_size),
                                    interpolation=cv2.INTER_AREA)

        return fs_stored_mask

    def xmask_image_overlay(self):
        """

        @return: Extracted face overlain by Xseg generated mask in BGR channel format.
        """
        xseg_mask = self.xmask
        if not self.xmask.any():
            return None

        mask = cv2.resize(xseg_mask, self.shape[:2], interpolation=cv2.INTER_AREA)
        normalized_mask = normalize_channels(mask, 1)
        xseg_image = self.decoded_image.astype(np.float32) / 255.0
        xseg_overlay_mask = xseg_image * (1 - normalized_mask) * 0.5 + xseg_image * normalized_mask
        clipped_xseg_overlay_mask = np.clip(xseg_overlay_mask * 255, 0, 255).astype(np.uint8)

        return clipped_xseg_overlay_mask

    def debug_fs_bisenet_face_overlay(self):
        """
        Returns the extracted face image array overlain by the bisenet-fp_face mask for debugging purposes.
        @return:
        """

        bisenet_face_mask = self.faceswap_mask("bisenet-fp_face", return_mask_parent_dict=False)

        mask = cv2.resize(bisenet_face_mask, self.shape[:2], interpolation=cv2.INTER_AREA)
        normalized_mask = normalize_channels(mask, 1) / 255.
        image = self.decoded_image.astype(np.float32) / 255.
        bisenet_mask_overlay = image * (1 - normalized_mask) * 0.5 + image * normalized_mask
        clipped_bisenet_mask_overlay = np.clip(bisenet_mask_overlay * 255, 0, 255).astype(np.uint8)

        fs_mask_from_xseg = (1 - normalized_mask) * 0.5 + normalized_mask
        fs_mask_from_xseg_clipped = np.clip(fs_mask_from_xseg * 255, 0, 255).astype(np.uint8)
        fs_stored_mask = cv2.resize(fs_mask_from_xseg_clipped, (128, 128), interpolation=cv2.INTER_AREA)

        return mask, clipped_bisenet_mask_overlay, fs_stored_mask

    @property
    def get_eyebrows_expand_mod(self):
        return 1.0


def read_image(filename, raise_error=False, with_metadata=False):
    """ Read an image file from a file location.

    Extends the functionality of :func:`cv2.imread()` by ensuring that an image was actually
    loaded. Errors can be logged and ignored so that the process can continue on an image load
    failure.

    Parameters
    ----------
    filename: str
        Full path to the image to be loaded.
    raise_error: bool, optional
        If ``True`` then any failures (including the returned image being ``None``) will be
        raised. If ``False`` then an error message will be logged, but the error will not be
        raised. Default: ``False``
    with_metadata: bool, optional
        Only returns a value if the images loaded are extracted Faceswap faces. If ``True`` then
        returns the Faceswap metadata stored with in a Face images .png_meta exif header.
        Default: ``False``

    Returns
    -------
    numpy.ndarray or tuple
        If :attr:`with_metadata` is ``False`` then returns a `numpy.ndarray` of the image in `BGR`
        channel order. If :attr:`with_metadata` is ``True`` then returns a `tuple` of
        (`numpy.ndarray`" of the image in `BGR`, `dict` of face's Faceswap metadata)
    Example
    -------
    >>> image_file = "/path/to/image.png_meta"
    >>> try:
    >>>    image = read_image(image_file, raise_error=True, with_metadata=False)
    >>> except:
    >>>     raise ValueError("There was an error")
    """
    success = True
    image = None
    filename = str(filename)
    try:
        if not with_metadata:
            retval = cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)
            if retval is None:
                raise ValueError("Image is None")
        else:
            with open(filename, "rb") as infile:
                raw_file = infile.read()
                metadata = png_read_meta(raw_file)
            image = cv2.imdecode(np.frombuffer(raw_file, dtype="uint8"), cv2.IMREAD_UNCHANGED)
            retval = (image, metadata)
    except TypeError as err:
        success = False
        msg = "Error while reading image (TypeError): '{}'".format(filename)
        msg += ". Original error message: {}".format(str(err))
        logger.error(msg)
        if raise_error:
            raise Exception(msg)
    except ValueError as err:
        success = False
        msg = ("Error while reading image. This can be caused by special characters in the "
               "filename or a corrupt image file: '{}'".format(filename))
        msg += ". Original error message: {}".format(str(err))
        logger.error(msg)
        if raise_error:
            raise Exception(msg)
    except Exception as err:  # pylint:disable=broad-except
        success = False
        msg = "Failed to load image '{}'. Original Error: {}".format(filename, str(err))
        logger.error(msg)
        if raise_error:
            raise Exception(msg)
    return retval


def png_read_meta(itxt_chunk):
    """ Read the Faceswap information stored in a png_meta's iTXt field.

    Parameters
    ----------
    itxt_chunk: bytes
        The bytes encoded png_meta file to read header data from

    Returns
    -------
    dict
        The Faceswap information stored in the png_meta header

    Notes
    -----
    This is a very stripped down, non-robust and non-secure header reader to fit a very specific
    task. OpenCV will not write any iTXt headers to the png_meta file, so we make the assumption that
    the only iTXt header that exists is the one that Faceswap created for storing alignments.
    """
    retval = None
    pointer = 0

    while True:
        pointer = itxt_chunk.find(b"iTXt", pointer) - 4
        if pointer < 0:
            print("No metadata in png_meta")
            break
        length = struct.unpack(">I", itxt_chunk[pointer:pointer + 4])[0]
        pointer += 8
        keyword, value = itxt_chunk[pointer:pointer + length].split(b"\0", 1)
        if keyword == b"faceswap":
            retval = literal_eval(value[4:].decode("utf-8"))
            break
        print("Skipping iTXt chunk: '%s'", keyword.decode("latin-1", "ignore"))
        pointer += length + 4
    return retval


def pack_to_itxt(metadata):
    """ Pack the given metadata dictionary to a PNG iTXt header field.

    Parameters
    ----------
    metadata: dict or bytes
        The dictionary to write to the header. Can be pre-encoded as utf-8.

    Returns
    -------
    bytes
        A byte encoded PNG iTXt field, including chunk header and CRC
    """
    if not isinstance(metadata, bytes):
        metadata = str(metadata).encode("utf-8", "strict")
    key = "faceswap".encode("latin-1", "strict")

    chunk = key + b"\0\0\0\0\0" + metadata
    crc = struct.pack(">I", crc32(chunk, crc32(b"iTXt")) & 0xFFFFFFFF)
    length = struct.pack(">I", len(chunk))
    retval = length + b"iTXt" + chunk + crc
    return retval


def update_existing_metadata(filename, metadata):
    """ Update the png header metadata for an existing .png extracted face file on the filesystem.

    Parameters
    ----------
    filename: str
        The full path to the face to be updated
    metadata: dict or bytes
        The dictionary to write to the header. Can be pre-encoded as utf-8.
    """

    tmp_filename = filename + "~"
    with open(filename, "rb") as png, open(tmp_filename, "wb") as tmp:
        chunk = png.read(8)
        if chunk != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Invalid header found in png: {filename}")
        tmp.write(chunk)

        while True:
            chunk = png.read(8)
            length, field = struct.unpack(">I4s", chunk)

            if field == b"IDAT":  # Write out all remaining data
                tmp.write(chunk + png.read())
                break

            if field != b"iTXt":  # Write non iTXt chunk straight out
                tmp.write(chunk + png.read(length + 4))  # Header + CRC
                continue

            keyword, value = png.read(length).split(b"\0", 1)
            if keyword != b"faceswap":
                # Write existing non fs-iTXt data + CRC
                tmp.write(keyword + b"\0" + value + png.read(4))
                continue

            tmp.write(pack_to_itxt(metadata))
            png.seek(4, 1)  # Skip old CRC

    os.replace(tmp_filename, filename)


class SegIEPolyType(IntEnum):
    EXCLUDE = 0
    INCLUDE = 1


class SegIEPoly:
    def __init__(self, type=None, pts=None, **kwargs):
        self.type = type

        if pts is None:
            pts = np.empty((0, 2), dtype=np.float32)
        else:
            pts = np.float32(pts)
        self.pts = pts
        self.n_max = self.n = len(pts)

    def dump(self):
        return {'type': int(self.type), 'pts': self.get_pts(), }

    def identical(self, b):
        if self.n != b.n:
            return False
        return (self.pts[0:self.n] == b.pts[0:b.n]).all()

    def get_type(self):
        return self.type

    def add_pt(self, x, y):
        self.pts = np.append(self.pts[0:self.n], [(float(x), float(y))], axis=0).astype(np.float32)
        self.n_max = self.n = self.n + 1

    def undo(self):
        self.n = max(0, self.n - 1)
        return self.n

    def redo(self):
        self.n = min(len(self.pts), self.n + 1)
        return self.n

    def redo_clip(self):
        self.pts = self.pts[0:self.n]
        self.n_max = self.n

    def insert_pt(self, n, pt):
        if n < 0 or n > self.n:
            raise ValueError("insert_pt out of range")
        self.pts = np.concatenate((self.pts[0:n], pt[None, ...].astype(np.float32), self.pts[n:]), axis=0)
        self.n_max = self.n = self.n + 1

    def remove_pt(self, n):
        if n < 0 or n >= self.n:
            raise ValueError("remove_pt out of range")
        self.pts = np.concatenate((self.pts[0:n], self.pts[n + 1:]), axis=0)
        self.n_max = self.n = self.n - 1

    def get_last_point(self):
        return self.pts[self.n - 1].copy()

    def get_pts(self):
        return self.pts[0:self.n].copy()

    def get_pts_count(self):
        return self.n

    def set_point(self, id, pt):
        self.pts[id] = pt

    def set_points(self, pts):
        self.pts = np.array(pts)
        self.n_max = self.n = len(pts)

    def mult_points(self, val):
        self.pts *= val


def dist_to_edges(pts, pt, is_closed=False):
    """
    returns array of dist from pt to edge and projection pt to edges
    """
    if is_closed:
        a = pts
        b = np.concatenate((pts[1:, :], pts[0:1, :]), axis=0)
    else:
        a = pts[:-1, :]
        b = pts[1:, :]

    pa = pt - a
    ba = b - a

    div = np.einsum('ij,ij->i', ba, ba)
    div[div == 0] = 1
    h = np.clip(np.einsum('ij,ij->i', pa, ba) / div, 0, 1)

    x = npla.norm(pa - ba * h[..., None], axis=1)

    return x, a + ba * h[..., None]


def struct_unpack(data, counter, fmt):
    fmt_size = struct.calcsize(fmt)
    return (counter + fmt_size,) + struct.unpack(fmt, data[counter:counter + fmt_size])


def normalize_channels(img, target_channels):
    img_shape_len = len(img.shape)
    if img_shape_len == 2:
        h, w = img.shape
        c = 0
    elif img_shape_len == 3:
        h, w, c = img.shape
    else:
        raise ValueError("normalize: incorrect image dimensions.")

    if c == 0 and target_channels > 0:
        img = img[..., np.newaxis]
        c = 1

    if c == 1 and target_channels > 1:
        img = np.repeat(img, target_channels, -1)
        c = target_channels

    if c > target_channels:
        img = img[..., 0:target_channels]
        c = target_channels

    return img


class DFLJPG(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = b""
        self.length = 0
        self.chunks = []
        self.dfl_dict = None
        self.shape = None
        self.img = None

    @staticmethod
    def load_raw(filename, loader_func=None):
        try:
            if loader_func is not None:
                data = loader_func(filename)
            else:
                with open(filename, "rb") as f:
                    data = f.read()
        except:
            raise FileNotFoundError(filename)

        try:
            inst = DFLJPG(filename)
            inst.data = data
            inst.length = len(data)
            inst_length = inst.length
            chunks = []
            data_counter = 0
            while data_counter < inst_length:
                chunk_m_l, chunk_m_h = struct.unpack("BB", data[data_counter:data_counter + 2])
                data_counter += 2

                if chunk_m_l != 0xFF:
                    raise ValueError(f"No Valid JPG info in {filename}")

                chunk_name = None
                chunk_size = None
                chunk_data = None
                chunk_ex_data = None
                is_unk_chunk = False

                if chunk_m_h & 0xF0 == 0xD0:
                    n = chunk_m_h & 0x0F

                    if n >= 0 and n <= 7:
                        chunk_name = "RST%d" % (n)
                        chunk_size = 0
                    elif n == 0x8:
                        chunk_name = "SOI"
                        chunk_size = 0
                        if len(chunks) != 0:
                            raise Exception("")
                    elif n == 0x9:
                        chunk_name = "EOI"
                        chunk_size = 0
                    elif n == 0xA:
                        chunk_name = "SOS"
                    elif n == 0xB:
                        chunk_name = "DQT"
                    elif n == 0xD:
                        chunk_name = "DRI"
                        chunk_size = 2
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xC0:
                    n = chunk_m_h & 0x0F
                    if n == 0:
                        chunk_name = "SOF0"
                    elif n == 2:
                        chunk_name = "SOF2"
                    elif n == 4:
                        chunk_name = "DHT"
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xE0:
                    n = chunk_m_h & 0x0F
                    chunk_name = "APP%d" % (n)
                else:
                    is_unk_chunk = True

                # if is_unk_chunk:
                #    #raise ValueError(f"Unknown chunk {chunk_m_h} in {filename}")
                #    io.log_info(f"Unknown chunk {chunk_m_h} in {filename}")

                if chunk_size == None:  # variable size
                    chunk_size, = struct.unpack(">H", data[data_counter:data_counter + 2])
                    chunk_size -= 2
                    data_counter += 2

                if chunk_size > 0:
                    chunk_data = data[data_counter:data_counter + chunk_size]
                    data_counter += chunk_size

                if chunk_name == "SOS":
                    c = data_counter
                    while c < inst_length and (data[c] != 0xFF or data[c + 1] != 0xD9):
                        c += 1

                    chunk_ex_data = data[data_counter:c]
                    data_counter = c

                chunks.append({'name': chunk_name, 'm_h': chunk_m_h, 'data': chunk_data, 'ex_data': chunk_ex_data, })
            inst.chunks = chunks

            return inst
        except Exception as e:
            raise Exception(f"Corrupted JPG file {filename} {e}")

    @staticmethod
    def load(filename, loader_func=None):
        try:
            inst = DFLJPG.load_raw(filename, loader_func=loader_func)
            inst.dfl_dict = {}

            for chunk in inst.chunks:
                if chunk['name'] == 'APP0':
                    d, c = chunk['data'], 0
                    c, id, _ = struct_unpack(d, c, "=4sB")

                    if id == b"JFIF":
                        c, ver_major, ver_minor, units, Xdensity, Ydensity, Xthumbnail, Ythumbnail = struct_unpack(d, c,
                                                                                                                   "=BBBHHBB")
                    else:
                        raise Exception("Unknown jpeg ID: %s" % (id))
                elif chunk['name'] == 'SOF0' or chunk['name'] == 'SOF2':
                    d, c = chunk['data'], 0
                    c, precision, height, width = struct_unpack(d, c, ">BHH")
                    inst.shape = (height, width, 3)

                elif chunk['name'] == 'APP15':
                    if type(chunk['data']) == bytes:
                        inst.dfl_dict = pickle.loads(chunk['data'])

            return inst
        except Exception as e:
            print()
            return None

    def has_data(self):
        return len(self.dfl_dict.keys()) != 0

    def save(self):
        try:
            with open(self.filename, "wb") as f:
                f.write(self.dump())
        except:
            raise Exception(f'cannot save {self.filename}')

    def dump(self):
        data = b""

        dict_data = self.dfl_dict

        # Remove None keys
        for key in list(dict_data.keys()):
            if dict_data[key] is None:
                dict_data.pop(key)

        for chunk in self.chunks:
            if chunk['name'] == 'APP15':
                self.chunks.remove(chunk)
                break

        last_app_chunk = 0
        for i, chunk in enumerate(self.chunks):
            if chunk['m_h'] & 0xF0 == 0xE0:
                last_app_chunk = i

        dflchunk = {'name': 'APP15', 'm_h': 0xEF, 'data': pickle.dumps(dict_data), 'ex_data': None, }
        self.chunks.insert(last_app_chunk + 1, dflchunk)

        for chunk in self.chunks:
            data += struct.pack("BB", 0xFF, chunk['m_h'])
            chunk_data = chunk['data']
            if chunk_data is not None:
                data += struct.pack(">H", len(chunk_data) + 2)
                data += chunk_data

            chunk_ex_data = chunk['ex_data']
            if chunk_ex_data is not None:
                data += chunk_ex_data

        return data

    def get_img(self):
        if self.img is None:
            self.img = read_image(self.filename)
        return self.img

    def get_shape(self):
        if self.shape is None:
            img = self.get_img()
            if img is not None:
                self.shape = img.shape
        return self.shape

    def get_dict(self):
        return self.dfl_dict

    def set_dict(self, dict_data=None):
        self.dfl_dict = dict_data

    def get_face_type(self):
        return self.dfl_dict.get('face_type')

    def set_face_type(self, face_type):
        self.dfl_dict['face_type'] = face_type

    def get_landmarks(self):
        return np.array(self.dfl_dict['landmarks'])

    def set_landmarks(self, landmarks):
        self.dfl_dict['landmarks'] = landmarks

    def get_eyebrows_expand_mod(self):
        return self.dfl_dict.get('eyebrows_expand_mod', 1.0)

    def set_eyebrows_expand_mod(self, eyebrows_expand_mod):
        self.dfl_dict['eyebrows_expand_mod'] = eyebrows_expand_mod

    def get_source_filename(self):
        return self.dfl_dict.get('source_filename', None)

    def set_source_filename(self, source_filename):
        self.dfl_dict['source_filename'] = source_filename

    def get_source_rect(self):
        return self.dfl_dict.get('source_rect', None)

    def set_source_rect(self, source_rect):
        self.dfl_dict['source_rect'] = source_rect

    def get_source_landmarks(self):
        return np.array(self.dfl_dict.get('source_landmarks', None))

    def set_source_landmarks(self, source_landmarks):
        self.dfl_dict['source_landmarks'] = source_landmarks

    def get_image_to_face_mat(self):
        mat = self.dfl_dict.get('image_to_face_mat', None)
        if mat is not None:
            return np.array(mat)
        return None

    def set_image_to_face_mat(self, image_to_face_mat):
        self.dfl_dict['image_to_face_mat'] = image_to_face_mat

    def has_seg_ie_polys(self):
        return self.dfl_dict.get('seg_ie_polys', None) is not None

    def get_seg_ie_polys(self):
        d = self.dfl_dict.get('seg_ie_polys', None)
        if d is not None:
            d = SegIEPolys.load(d)
        else:
            d = SegIEPolys()

        return d

    def set_seg_ie_polys(self, seg_ie_polys):
        if seg_ie_polys is not None:
            if not isinstance(seg_ie_polys, SegIEPolys):
                raise ValueError('seg_ie_polys should be instance of SegIEPolys')

            if seg_ie_polys.has_polys():
                seg_ie_polys = seg_ie_polys.dump()
            else:
                seg_ie_polys = None

        self.dfl_dict['seg_ie_polys'] = seg_ie_polys

    def has_xseg_mask(self):
        return self.dfl_dict.get('xseg_mask', None) is not None

    def get_xseg_mask_compressed(self):
        mask_buf = self.dfl_dict.get('xseg_mask', None)
        if mask_buf is None:
            return None

        return mask_buf

    def get_xseg_mask(self):
        mask_buf = self.dfl_dict.get('xseg_mask', None)
        if mask_buf is None:
            return None

        img = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = img[..., None]

        return img.astype(np.float32) / 255.0

    def set_xseg_mask(self, mask_a):
        if mask_a is None:
            self.dfl_dict['xseg_mask'] = None
            return

        mask_a = normalize_channels(mask_a, 1)
        img_data = np.clip(mask_a * 255, 0, 255).astype(np.uint8)

        data_max_len = 50000

        ret, buf = cv2.imencode('.png', img_data)

        if not ret or len(buf) > data_max_len:
            for jpeg_quality in range(100, -1, -1):
                ret, buf = cv2.imencode('.jpg', img_data, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if ret and len(buf) <= data_max_len:
                    break

        if not ret:
            raise Exception("set_xseg_mask: unable to generate image data for set_xseg_mask")

        self.dfl_dict['xseg_mask'] = buf
