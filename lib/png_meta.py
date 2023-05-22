import dataclasses
from dataclasses import dataclass


@dataclass
class DeepFaceLabMetaKeys:
    face_type: str = "face_type"
    source_height: str = "source_height"
    source_width: str = "source_width"
    bounding_box_left_most_x: str = "bounding_box_left_most_x"
    bounding_box_top_most_y: str = "bounding_box_top_most_y"
    bounding_box_width: str = "bounding_box_width"
    bounding_box_height: str = "bounding_box_height"
    extracted_landmarks: str = "extracted_landmarks"
    extracted_filename: str = "extracted_filename"
    source_rect: str = "source_rect"
    image_to_face_mat: str = "image_to_face_mat"
    seg_ie_polys: str = "seg_ie_polys"
    xseg_mask: str = "xseg_mask"
    # Below keys are extra info created from faceswap extracted images for use in xseg training
    extracted_image_name: str = "extracted_image_name"
    resized_mask: str = "resized_mask"
    transformed_image: str = "transformed_image"
    extracted_face_mask_overlay: str = "extracted_face_mask_overlay"


@dataclass
class FaceswapMetaKeys:
    source: str = "source"
    itxt_chunk: str = "faceswap"
    alignments: str = "alignments"
    frame_dims: str = "source_frame_dims"
    landmarks: str = "landmarks_xy"
    alignments_mask: str = "mask"
    source_filename: str = "source_filename"
    bounding_box_x: str = 'x'
    bounding_box_y: str = 'y'
    bounding_box_width: str = 'w'
    bounding_box_height: str = 'h'


@dataclass
class FaceswapMasks:
    components: str = "components"
    extended: str = "extended"
    bisenet_fp_face: str = "bisenet-fp_face"
    vgg_clear: str = "vgg_clear"
    vgg_obstructed: str = "vgg_obstructed"
    unet_dfl: str = "unet_dfl"


META_KEYS_TO_ENCODE: list = [DeepFaceLabMetaKeys.xseg_mask, DeepFaceLabMetaKeys.seg_ie_polys,
                             DeepFaceLabMetaKeys.extracted_landmarks, DeepFaceLabMetaKeys.image_to_face_mat]

FACESWAP_MASKS = []
for field in dataclasses.fields(FaceswapMasks):
    FACESWAP_MASKS.append(getattr(FaceswapMasks, field.name))
