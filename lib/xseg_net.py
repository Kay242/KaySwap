from pathlib import Path
import numpy as np
from lib.deepfacelab.interact import interact as io
from lib.deepfacelab.leras.models.XSeg import XSeg
from lib.deepfacelab.leras.neuralnet import NeuralNet


class XSegNet(object):
    VERSION = 1

    def __init__(self, name,
                 resolution=512,
                 load_weights=True,
                 weights_file_root=None,
                 training=False,
                 place_model_on_cpu=False,
                 run_on_cpu=False,
                 optimizer=None,
                 data_format="NHWC",
                 raise_on_no_model_files=False):

        self.resolution = resolution
        self.weights_file_root = Path(weights_file_root) if weights_file_root is not None else Path(__file__).parent

        NeuralNet.initialize(data_format=data_format)
        tf = NeuralNet.tf

        model_name = f'{name}_{resolution}'
        self.model_filename_list = []

        # Initializing model classes
        with tf.device('/CPU:0' if place_model_on_cpu else NeuralNet.tf_default_device_name):
            self.model = XSeg(3, 32, 1, name=name)
            self.model_weights = self.model.get_weights()
            if training:
                if optimizer is None:
                    raise ValueError("Optimizer should be provided for training mode.")
                self.opt = optimizer
                self.opt.initialize_variables(self.model_weights, vars_on_cpu=place_model_on_cpu)
                self.model_filename_list += [[self.opt, f'{model_name}_opt.npy']]

        self.model_filename_list += [[self.model, f'{model_name}.npy']]

        if not training:
            def net_run(input_np):
                self.input_t = input_np
                with tf.device('/CPU:0' if run_on_cpu else NeuralNet.tf_default_device_name):
                    _, pred = self.model(self.input_t)
                    return pred

            self.net_run = net_run

        self.initialized = True
        # Loading/initializing all models/optimizers weights
        for model, filename in self.model_filename_list:
            do_init = not load_weights

            if not do_init:
                model_file_path = self.weights_file_root / filename
                do_init = not model.load_weights(model_file_path)
                if do_init:
                    if raise_on_no_model_files:
                        raise Exception(f'{model_file_path} does not exists.')
                    if not training:
                        self.initialized = False
                        break

            if do_init:
                model.init_weights()

    def get_resolution(self):
        return self.resolution

    def flow(self, x, pretrain=False):
        return self.model(x, pretrain=pretrain)

    def get_weights(self):
        return self.model_weights

    def save_weights(self):
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Saving", leave=False):
            model.save_weights(self.weights_file_root / filename)

    def extract(self, input_image):
        if not self.initialized:
            return 0.5 * np.ones((self.resolution, self.resolution, 1), NeuralNet.floatx.as_numpy_dtype)

        input_shape_len = len(input_image.shape)

        if input_shape_len == 3:
            input_image = input_image[None, ...]

        result = np.clip(self.net_run(input_image), 0, 1.0)
        result[result < 0.1] = 0  # get rid of noise

        if input_shape_len == 3:
            result = result[0]

        return result
