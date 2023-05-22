from typing import List, Any

META_FACE_TYPE_KEY: str = "face_type"

META_SOURCE_IMAGE_HEIGHT_KEY: str = "source_height"
META_SOURCE_IMAGE_WIDTH_KEY: str = "source_width"
META_BOUNDING_BOX_X_KEY: str = "bounding_box_left_most_x"
META_BOUNDING_BOX_Y_KEY: str = "bounding_box_top_most_y"
META_BOUNDING_BOX_WIDTH_KEY: str = "bounding_box_width"
META_BOUNDING_BOX_HEIGHT_KEY: str = "bounding_box_height"
META_EXTRACTED_IMAGE_LANDMARKS_KEY: str = "landmarks"
META_SOURCE_FILENAME_KEY: str = "source_filename"
META_SOURCE_RECT_KEY: str = "source_rect"  # TODO: Currently left empty. Implement it for PNG, (low priority)
META_SOURCE_IMAGE_LANDMARKS_KEY: str = "source_landmarks"
META_IMAGE2FACE_MAT_KEY: str = "image_to_face_mat"
META_XSEG_POLYS_KEY: str = "seg_ie_polys"
META_XSEG_MASK_KEY: str = "xseg_mask"
META_FACESWAP_INFO_KEY: str = "faceswap"

META_FACE_TYPE_DEFAULT_VALUE: str = "whole_face"

META_KEYS_OF_STRING_VALUES: List[str] = [META_FACE_TYPE_KEY,
                                         META_SOURCE_FILENAME_KEY,
                                         META_FACESWAP_INFO_KEY]
META_KEYS_OF_INT_VALUES: List[str] = [META_SOURCE_IMAGE_HEIGHT_KEY, META_SOURCE_IMAGE_WIDTH_KEY,
                                      META_BOUNDING_BOX_X_KEY, META_BOUNDING_BOX_Y_KEY,
                                      META_BOUNDING_BOX_WIDTH_KEY, META_BOUNDING_BOX_HEIGHT_KEY]
META_KEYS_OF_NPARRAY_UNIT8_VALUES: List[str] = [META_XSEG_MASK_KEY,
                                                META_SOURCE_RECT_KEY]
META_KEYS_OF_NPARRAY_FLOAT32_VALUES: List[str] = [META_SOURCE_IMAGE_LANDMARKS_KEY,
                                                  META_EXTRACTED_IMAGE_LANDMARKS_KEY]
META_KEYS_OF_NPARRAY_FLOAT64_VALUES: List[str] = [META_IMAGE2FACE_MAT_KEY]
META_KEYS_OF_NPARRAY_ANYTYPE_VALUES: List[str] = [META_XSEG_MASK_KEY, META_SOURCE_RECT_KEY, META_IMAGE2FACE_MAT_KEY,
                                                  META_SOURCE_IMAGE_LANDMARKS_KEY, META_EXTRACTED_IMAGE_LANDMARKS_KEY]

IMAGE_INFO_DICT_KEYS: List[str] = [META_FACE_TYPE_KEY, META_SOURCE_IMAGE_HEIGHT_KEY, META_SOURCE_IMAGE_WIDTH_KEY,
                                   META_BOUNDING_BOX_X_KEY, META_BOUNDING_BOX_Y_KEY, META_BOUNDING_BOX_WIDTH_KEY,
                                   META_BOUNDING_BOX_HEIGHT_KEY, META_EXTRACTED_IMAGE_LANDMARKS_KEY,
                                   META_SOURCE_FILENAME_KEY, META_SOURCE_RECT_KEY, META_SOURCE_IMAGE_LANDMARKS_KEY,
                                   META_IMAGE2FACE_MAT_KEY, META_XSEG_POLYS_KEY, META_XSEG_MASK_KEY,
                                   META_FACESWAP_INFO_KEY]
IMAGE_INFO_DICT_VALUES: List[Any] = [META_FACE_TYPE_DEFAULT_VALUE, "", "", "", "",
                                     "", "", "", "", "", "", "", None, None, ""]



