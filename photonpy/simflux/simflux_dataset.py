# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:00:53 2021

@author: jelmer
"""

from photonpy import Dataset
import numpy as np


class SFDataset(Dataset):
    """
    Dataset that holds an array of intensity values per spot
    """

    def __init__(self, length, dims, imgshape, numPatterns=None, config=None, **kwargs):

        if config is None:
            config = {}

        # This looks kinda convoluted but the goal is that Dataset.merge and __getitem__ are able to create SFDatasets
        if numPatterns is not None:
            config['numPatterns'] = numPatterns

        super().__init__(length, dims, imgshape, config=config, **kwargs)

    def createDTypes(self, dims, imgdims, includeGaussSigma, extraFields=None):
        numPatterns = self.config['numPatterns']

        extraFields = [
            ('IBg', np.float32, (numPatterns, 2)),
            ('IBg_crlb', np.float32, (numPatterns, 2))
        ]

        return super().createDTypes(dims, imgdims, includeGaussSigma, extraFields=extraFields)

    @property
    def IBg(self):
        return self.data.IBg

