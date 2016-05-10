"""Top-level objects for accelerated imaging, that glue the lower-level
objects together."""

from __future__ import print_function, division
import numpy as np
from katsdpsigproc import accel
from . import grid, weight, image, clean, types


class ImagingTemplate(object):
    """Template holding all the other templates for imaging."""

    def __init__(self, command_queue, array_parameters, image_parameters,
                 weight_parameters, grid_parameters, clean_parameters):
        self.command_queue = command_queue
        self.array_parameters = array_parameters
        self.image_parameters = image_parameters
        self.weight_parameters = weight_parameters
        self.grid_parameters = grid_parameters
        self.clean_parameters = clean_parameters
        context = command_queue.context
        image_shape = (len(image_parameters.polarizations),
                       image_parameters.pixels,
                       image_parameters.pixels)
        # Currently none of the kernels accessing the layer need any padding.
        # It would be nice if there was a cleaner way to handle this; possibly
        # by deferring creation of the FFT plan until instantiation.
        padded_layer_shape = layer_shape = image_shape[1:]
        self.weights = weight.WeightsTemplate(
            context, weight_parameters.weight_type, len(image_parameters.polarizations))
        self.gridder = grid.GridderTemplate(context, image_parameters, grid_parameters)
        self.degridder = grid.DegridderTemplate(context, image_parameters, grid_parameters)
        self.grid_image = image.GridImageTemplate(
            command_queue, layer_shape, padded_layer_shape, image_parameters.real_dtype)
        self.clean = clean.CleanTemplate(
            context, clean_parameters, image_parameters.real_dtype, image_shape[0])
        self.scale = image.ScaleTemplate(
            context, image_parameters.real_dtype, image_shape[0])
        self.taper1d = accel.SVMArray(
            context, (image_parameters.pixels,), image_parameters.real_dtype)
        self.gridder.convolve_kernel.taper(image_parameters.pixels, self.taper1d)
        self.untaper1d = accel.SVMArray(
            context, (image_parameters.pixels,), image_parameters.real_dtype)
        self.degridder.convolve_kernel.taper(image_parameters.pixels, self.untaper1d)

    def instantiate(self, *args, **kwargs):
        return Imaging(self, *args, **kwargs)


class Imaging(accel.OperationSequence):
    def __init__(self, template, max_vis, allocator=None):
        self.template = template
        lm_scale = float(template.image_parameters.pixel_size)
        lm_bias = -0.5 * template.image_parameters.pixels * lm_scale
        image_shape = (len(template.image_parameters.polarizations),
                       template.image_parameters.pixels,
                       template.image_parameters.pixels)
        self._gridder = template.gridder.instantiate(
            template.command_queue, template.array_parameters, max_vis, allocator)
        self._degridder = template.degridder.instantiate(
            template.command_queue, template.array_parameters, max_vis, allocator)
        grid_shape = self._gridder.slots['grid'].shape
        degrid_shape = self._degridder.slots['grid'].shape
        self._weights = template.weights.instantiate(
            template.command_queue, grid_shape, max_vis, allocator)
        self._weights.robustness = template.weight_parameters.robustness
        self._grid_to_image = template.grid_image.instantiate_grid_to_image(
            grid_shape, lm_scale, lm_bias, allocator)
        self._image_to_grid = template.grid_image.instantiate_image_to_grid(
            degrid_shape, lm_scale, lm_bias, allocator)
        self._clean = template.clean.instantiate(
            template.command_queue, template.image_parameters, allocator)
        self._scale = template.scale.instantiate(
            template.command_queue, image_shape, allocator)
        self._grid_to_image.bind(kernel1d=template.taper1d)
        self._image_to_grid.bind(kernel1d=template.untaper1d)
        operations = [
            ('weights', self._weights),
            ('gridder', self._gridder),
            ('degridder', self._degridder),
            ('grid_to_image', self._grid_to_image),
            ('image_to_grid', self._image_to_grid),
            ('clean', self._clean),
            ('scale', self._scale)
        ]
        compounds = {
            'uv': ['weights:uv', 'gridder:uv', 'degridder:uv'],
            'weights': ['weights:weights'],
            'weights_grid': ['weights:grid', 'gridder:weights_grid'],
            'w_plane': ['gridder:w_plane', 'degridder:w_plane'],
            'vis': ['gridder:vis', 'degridder:vis'],
            'grid': ['gridder:grid', 'grid_to_image:grid'],
            'degrid': ['degridder:grid', 'image_to_grid:grid'],
            'layer': ['grid_to_image:layer', 'image_to_grid:layer'],
            'dirty': ['grid_to_image:image', 'clean:dirty', 'scale:data'],
            'model': ['clean:model', 'image_to_grid:image'],
            'psf': ['clean:psf'],
            'tile_max': ['clean:tile_max'],
            'tile_pos': ['clean:tile_pos'],
            'peak_value': ['clean:peak_value'],
            'peak_pos': ['clean:peak_pos'],
            'peak_pixel': ['clean:peak_pixel']
        }
        # TODO: could alias weights with something, since it's only needed
        # early on while setting up weights_grid
        super(Imaging, self).__init__(
            template.command_queue, operations, compounds, allocator=allocator)

    def __call__(self):
        raise NotImplementedError()

    def clear_weights(self):
        self.ensure_all_bound()
        self._weights.clear()

    def grid_weights(self, uv, weights):
        self.ensure_all_bound()
        self._weights.grid(uv, weights)

    def finalize_weights(self):
        self.ensure_all_bound()
        self._weights.finalize()

    def clear_grid(self):
        self.ensure_all_bound()
        self.buffer('grid').zero(self.command_queue)

    def clear_dirty(self):
        self.ensure_all_bound()
        self.buffer('dirty').zero(self.command_queue)

    def clear_model(self):
        self.ensure_all_bound()
        self.buffer('model').zero(self.command_queue)

    def grid(self, *args, **kwargs):
        self.ensure_all_bound()
        self._gridder.grid(*args, **kwargs)

    def degrid(self, *args, **kwargs):
        self.ensure_all_bound()
        self._degridder.degrid(*args, **kwargs)

    def grid_to_image(self, w):
        self.ensure_all_bound()
        self._grid_to_image.set_w(w)
        self._grid_to_image()

    def model_to_grid(self, w):
        self.ensure_all_bound()
        self._image_to_grid.set_w(w)
        self._image_to_grid()

    def scale_dirty(self, scale_factor):
        self.ensure_all_bound()
        self._scale.set_scale_factor(scale_factor)
        self._scale()

    def clean_reset(self):
        self.ensure_all_bound()
        self._clean.reset()

    def clean_cycle(self):
        self.ensure_all_bound()
        self._clean()


class ImagingHost(object):
    """Host-only equivalent to :class:`Imaging`."""

    def __init__(self, image_parameters, weight_parameters, grid_parameters, clean_parameters):
        lm_scale = float(image_parameters.pixel_size)
        lm_bias = -0.5 * image_parameters.pixels * lm_scale
        psf_shape = (len(image_parameters.polarizations),
                     clean_parameters.psf_patch, clean_parameters.psf_patch)
        self._gridder = grid.GridderHost(image_parameters, grid_parameters)
        self._degridder = grid.DegridderHost(image_parameters, grid_parameters)
        self._grid = self._gridder.values
        self._degrid = self._degridder.values
        self._weights_grid = self._gridder.weights_grid
        self._weights = weight.WeightsHost(weight_parameters.weight_type, self._weights_grid)
        self._weights.robustness = weight_parameters.robustness
        self._layer = np.empty(self._grid.shape, image_parameters.complex_dtype)
        self._dirty = np.empty(self._grid.shape, image_parameters.real_dtype)
        self._model = np.empty(self._grid.shape, image_parameters.real_dtype)
        self._psf = np.empty(psf_shape, image_parameters.real_dtype)
        self._grid_to_image = image.GridToImageHost(
            self._grid, self._layer, self._dirty,
            self._gridder.kernel.taper(image_parameters.pixels), lm_scale, lm_bias)
        self._image_to_grid = image.ImageToGridHost(
            self._degrid, self._layer, self._model,
            self._degridder.kernel.taper(image_parameters.pixels), lm_scale, lm_bias)
        self._clean = clean.CleanHost(image_parameters, clean_parameters,
                                      self._dirty, self._psf, self._model)
        self._buffer = {
            'psf': self._psf,
            'dirty': self._dirty,
            'model': self._model,
            'grid': self._grid,
            'degrid': self._degrid,
            'weights_grid': self._weights_grid
        }

    def buffer(self, name):
        return self._buffer[name]

    def clear_weights(self):
        self._weights_grid.fill(0)

    def grid_weights(self, uv, weights):
        self._weights.grid(uv, weights)

    def finalize_weights(self):
        self._weights.finalize()

    def clear_grid(self):
        self._grid.fill(0)

    def clear_dirty(self):
        self._dirty.fill(0)

    def clear_model(self):
        self._model.fill(0)

    def grid(self, *args, **kwargs):
        self._gridder.grid(*args, **kwargs)

    def degrid(self, *args, **kwargs):
        self._degridder.degrid(*args, **kwargs)

    def grid_to_image(self, w):
        self._grid_to_image.set_w(w)
        self._grid_to_image()

    def model_to_grid(self, w):
        self._image_to_grid.set_w(w)
        self._image_to_grid()

    def scale_dirty(self, scale_factor):
        self._dirty *= scale_factor[:, np.newaxis, np.newaxis]

    def clean_reset(self):
        self._clean.reset()

    def clean_cycle(self):
        self._clean()