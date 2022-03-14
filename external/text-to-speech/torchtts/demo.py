"""
 Copyright (c) 2020-2022 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import os.path as osp

import numpy as np
from openvino.runtime import PartialShape

from text_preprocessing import text_to_sequence, cmudict, intersperse


def check_input_name(model, input_tensor_name):
    try:
        model.input(input_tensor_name)
        return True
    except RuntimeError:
        return False


class IEModel:
    def __init__(self, ie, device):
        self.ie = ie
        self.device = device

    def load_network(self, model_xml):
        model_bin_name = ".".join(osp.basename(model_xml).split('.')[:-1]) + ".bin"
        model_bin = osp.join(osp.dirname(model_xml), model_bin_name)
        log.info('Reading AcousticGAN model {}'.format(model_xml))
        model = self.ie.read_model(model=model_xml, weights=model_bin)
        return model

    def create_infer_request(self, model, path=None):
        compiled_model = self.ie.compile_model(model, device_name=self.device)
        if path is not None:
            log.info('The AcousticGAN model {} is loaded to {}'.format(path, self.device))
        return compiled_model.create_infer_request()


class Encoder(IEModel):
    def __init__(self, model_encoder, ie, device='CPU'):
        super().__init__(ie, device)
        self.encoder = self.load_network(model_encoder)
        self.request = self.create_infer_request(self.encoder, model_encoder)
        self.enc_input_data_name = "seq"
        self.enc_input_mask_name = "seq_len"

    def preprocess(self, seq):
        seq = np.array(seq)[None, :]
        seq_len = np.array([seq.shape[1]])

        model_shape = self.encoder.input(self.enc_input_data_name).shape[1]
        if model_shape != seq.shape[1]:
            new_shape = {self.enc_input_data_name: PartialShape(seq.shape), self.enc_input_mask_name: PartialShape([1])}
            self.encoder.reshape(new_shape)
            self.request = self.create_infer_request(self.encoder)

        return {self.enc_input_data_name: seq, self.enc_input_mask_name: seq_len}

    def infer(self, data):
        self.request.infer(data)


class Decoder(IEModel):
    def __init__(self, model_decoder, ie, device='CPU'):
        super().__init__(ie, device)
        self.decoder = self.load_network(model_decoder)
        self.request = self.create_infer_request(self.encoder, model_decoder)

        self.dec_input_data_name = "z"
        self.dec_input_mask_name = "z_mask"

    def preprocess(self, z, z_mask):
        model_shape = list(self.decoder.input(self.dec_input_data_name).shape)
        if model_shape[-1] != z.shape[-1]:
            self.decoder.reshape({self.dec_input_data_name: PartialShape(z.shape),
                                  self.dec_input_mask_name: PartialShape(z_mask.shape)})
            self.request = self.create_infer_request(self.decoder)

        return {self.dec_input_data_name: z, self.dec_input_mask_name: z_mask}

    def infer(self, data):
        self.request.infer(data)


class AcousticGANIE:
    def __init__(self, model_encoder, model_decoder, ie, device='CPU', verbose=False):
        self.verbose = verbose
        self.device = device
        self.ie = ie

        self.cmudict = cmudict.CMUDict(osp.join(osp.dirname(osp.realpath(__file__)), 'data/cmu_dictionary'))

        self.encoder = Encoder(model_encoder, ie, device)
        self.decoder = Decoder(model_decoder, ie, device)

    def seq_to_indexes(self, text):
        res = text_to_sequence(text, dictionary=self.cmudict)
        if self.verbose:
            log.debug(res)
        return res

    @staticmethod
    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = np.arange(max_length, dtype=length.dtype)
        return np.expand_dims(x, 0) < np.expand_dims(length, 1)

    @staticmethod
    def generate_path(duration, mask):
        """
        duration: [b, t_text]
        mask: [b, t_text, t_mel]
        """

        b, t_x, t_y = mask.shape  # batch size, text size, mel size
        cum_duration = np.cumsum(duration, 1)

        cum_duration_flat = cum_duration.flatten()  # view(b * t_x)
        path = AcousticGANIE.sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)

        path = path.reshape(b, t_x, t_y)
        path = path - np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]

        path = path * mask

        return path

    def gen_decoder_in(self, alpha=1.0, offset=0.3):
        x_mask = self.encoder.request.get_tensor("x_mask").data[:]
        x_res = self.encoder.request.get_tensor("x_res").data[:]
        logw = self.encoder.request.get_tensor("logw").data[:]

        w = (np.exp(logw) + offset) * x_mask
        w_ceil = np.ceil(w) * alpha

        mel_lengths = np.clip(np.sum(w_ceil, axis=(1, 2)), a_min=1, a_max=None).astype(dtype=np.long)

        z_mask = np.expand_dims(self.sequence_mask(mel_lengths), 1).astype(x_mask.dtype)
        attn_mask = np.expand_dims(x_mask, -1) * np.expand_dims(z_mask, 2)

        attn = self.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1))
        attn = np.expand_dims(attn, 1)
        attn = attn.squeeze(1).transpose(0, 2, 1)
        z = np.matmul(attn, x_res.transpose(0, 2, 1)).transpose(0, 2, 1)  # [b, t', t], [b, t, d] -> [b, d, t']

        return z, z_mask

    def forward(self, text, alpha=1.0, **kwargs):
        seq = self.seq_to_indexes(text)

        encoder_in = self.encoder.preprocess(seq)
        self.encoder.infer(encoder_in)
        decoder_in = self.decoder.preprocess(self.gen_decoder_in(alpha))
        self.decoder_request.infer(decoder_in)

        res = self.decoder.request.get_tensor("mel").data[:]
        res = res * 6.0 - 6.0
        return res
