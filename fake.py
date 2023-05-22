from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import imageio
from lib.image_alt import SegIEPolys, normalize_channels
from PIL import Image, PngImagePlugin
import PNG
import json


def imread(filename):
    read_image: imageio.core.util.Array = imageio.imread(filename)
    np_image: np.ndarray = np.asarray(read_image)

    # Original code is based on reading JPG data using cv2.
    # This needs to change for using FaceSwap (to read PNG files).
    # After testing by reading a JPG file using cv2 and imageio it was found that
    # imageio.imread returns the data a little differently.
    # To ensure the data is delivered the same way to the code (since original implementation is based on cv2),
    # the 1st and 3rd columns need to be swapped, as is done below.
    # Code for doing so is based on https://stackoverflow.com/a/33362288
    np_image[:, :, 0], np_image[:, :, 2] = \
        np_image[:, :, 2], np_image[:, :, 0].copy()

    return np_image


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class DflPng():
    def __init__(self, file_path: Path):
        self.filename = file_path
        self.data = b""
        self.length = 0
        self.chunks = []
        self.dfl_dict: Dict[str, Any] = {}
        for key, value in zip(PNG.IMAGE_INFO_DICT_KEYS, PNG.IMAGE_INFO_DICT_VALUES):
            self.dfl_dict[key] = value
        self.dfl_dict["source_filename"] = file_path.name
        self.shape = None
        self.img = None

    @staticmethod
    def load(filename: Path):
        png_file = open(filename, "rb")
        png_contents = png_file.read()

        inst = DflPng(filename)
        inst.data = png_contents
        inst.length = len(png_contents)

        inst.img = imm = Image.open(filename)
        inst.dfl_dict = imm.info
        inst.fix_dict_vals(imm.info)

        inst.shape = (imm.height, imm.width, 3)
        png_file.close()

        return inst

    def fix_dict_vals(self, info_dict: Dict[Any, Any]) -> None:
        for current_key, current_value in info_dict.items():
            if current_key in PNG.META_KEYS_OF_INT_VALUES:
                self.dfl_dict[current_key] = int(current_value)

            elif current_key in PNG.META_KEYS_OF_NPARRAY_FLOAT32_VALUES:
                self.dfl_dict[current_key] = np.asarray(json.loads(current_value)).astype(np.float32)

            elif current_key in PNG.META_KEYS_OF_NPARRAY_UNIT8_VALUES:
                mask_buf = np.asarray(json.loads(current_value)).astype(np.uint8)
                self.dfl_dict[current_key] = mask_buf

            elif current_key == PNG.META_XSEG_POLYS_KEY:
                self.dfl_dict[current_key] = json.loads(current_value)

            elif current_key in PNG.META_KEYS_OF_NPARRAY_FLOAT64_VALUES:
                self.dfl_dict[current_key] = np.asarray(json.loads(current_value)).astype(np.float64)

            # for regular string dict values (like source file name) just assign as usual
            else:
                self.dfl_dict[current_key] = current_value

    def save(self) -> None:
        info = PngImagePlugin.PngInfo()
        for current_key, current_value in self.dfl_dict.items():
            if current_key in PNG.META_KEYS_OF_NPARRAY_ANYTYPE_VALUES:
                adjusted_value = json.dumps(current_value, indent=2, cls=NumpyEncoder).encode('latin-1')
            elif current_key == PNG.META_XSEG_POLYS_KEY:
                adjusted_value = json.dumps(current_value, indent=2, cls=NumpyEncoder).encode('latin-1')
            else:
                adjusted_value = str(current_value)

            info.add_text(current_key, adjusted_value)

        imm = Image.open(self.filename)
        imm.save(self.filename, "PNG", pnginfo=info)

    def has_data(self):
        return len(self.dfl_dict.keys()) != 0

    def get_img(self):
        if self.img is None:
            self.img = imread(self.filename)
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

    def set_face_type(self, face_type):
        self.dfl_dict['face_type'] = face_type

    def get_landmarks(self):
        value = json.loads(self.dfl_dict['landmarks'])
        return np.asarray(value).astype(np.float64)

    def set_landmarks(self, landmarks):
        self.dfl_dict['landmarks'] = landmarks

    def get_eyebrows_expand_mod(self):
        return self.dfl_dict.get('eyebrows_expand_mod', 1.0)

    def get_source_filename(self):
        return self.dfl_dict.get('source_filename', None)

    def set_source_filename(self, source_filename):
        self.dfl_dict['source_filename'] = source_filename

    def get_source_rect(self):
        return self.dfl_dict.get('source_rect', None)

    def set_source_rect(self, source_rect):
        self.dfl_dict['source_rect'] = source_rect

    def get_source_landmarks(self):
        return self.dfl_dict.get('source_landmarks', None)

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
        return self.dfl_dict.get('seg_ie_polys') is not None

    def get_seg_ie_polys(self):
        dict_val = self.dfl_dict.get("seg_ie_polys")
        if dict_val is not None:
            d = SegIEPolys.load(dict_val)
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

        ret, buf = cv2.imencode('.png', img_data)

        # if not ret or len(buf) > data_max_len:
        #     for jpeg_quality in range(100, -1, -1):
        #         ret, buf = cv2.imencode('.png', img_data, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        #         if ret and len(buf) <= data_max_len:
        #             break

        if not ret:
            raise Exception("set_xseg_mask: unable to generate image data for set_xseg_mask")

        self.dfl_dict['xseg_mask'] = buf


import pickle
originals_dir = Path("/home/kay/Desktop/dfl_model/data_src/Py_Aligned/xseg_old")
extracted = Path("/home/kay/Desktop/dfl_model/data_src/Py_Aligned")
image_paths_OLD = list(x for x in originals_dir.iterdir() if x.is_file())
image_paths = list(x for x in extracted.iterdir() if x.is_file())


def dump_old_meta():
    pickled_array = []
    for image_path in image_paths_OLD:
        readim = DflPng.load(image_path)
        xseg_poly = readim.get_seg_ie_polys()
        xseg_mask = readim.get_xseg_mask()
        pickled_array.append([xseg_poly, xseg_mask])

    with open(r"out.pkl", 'wb') as pickle_file:
        pickle.dump(pickled_array, pickle_file)


def load_new_meta():
    from lib import image_alt
    with open("out.pkl", 'rb') as pickle_file:
        pp = pickle.load(pickle_file)
    i: int = 0
    polys: SegIEPolys
    mask: np.ndarray
    for image_path in image_paths:
        readim = image_alt.DflPng.load(image_path)
        polys, mask = pp[i]
        if polys.has_polys():
            readim.set_seg_ie_polys(polys)
        readim.set_xmask(mask)
        readim.save()
        i += 1
    print()

dump_old_meta()
load_new_meta()


from lib import image_alt
#from lib import image as og_fs
og = image_alt.read_image(r"D:\AI\z1_SideB\maddii_edee\3_Training_Src\00299_0.png", with_metadata=True)

img = image_alt.DflPng.load(Path(r"C:\Users\akana\Desktop\dfl_model\data_src\Py_Aligned\002.png"))
comp = img.xmask()
img.faceswap_mask("bisenet-fp_face")
print()
