from functools import partial
from typing import Any, Dict, List, Optional

from .contrastive import Contrastive
from dig.sslgraph.method.contrastive.views_fn import (NodeAttrMask, EdgePerturbation, Sequential,
                                                      UniformSample)
from gtaxo_graphgym.transform.perturbations.spectral import BandpassFiltering, WaveletBankFiltering

# TODO: Make aug_dict accessible across all NCE methods?
aug_dict = {
    "dropN": UniformSample,
    "permE": EdgePerturbation,
    "dropE": partial(EdgePerturbation, drop=True),
    "maskN": NodeAttrMask,
    "BP": BandpassFiltering,
    'BPhi': partial(BandpassFiltering, band='hi'),
    'BPmid': partial(BandpassFiltering, band='mid'),
    'BPlo': partial(BandpassFiltering, band='lo'),
    "WP": WaveletBankFiltering,
    'WBhi': partial(WaveletBankFiltering, bands=[True, False, False], norm="sym"),
    'WBmid': partial(WaveletBankFiltering, bands=[False, True, False], norm="sym"),
    'WBlo': partial(WaveletBankFiltering, bands=[False, False, True], norm="sym"),
}


def setup_views_fn(aug: Optional[List[Dict[str, Any]]]):
    """ Set up views function given the aug specification.

    Example aug:
    [
        {
            "name": "dropN",
            "params": {
                "ratio": 0.2,
            },
        },
        {
            "name": "WBlo",
            "params": {
                "bands": [False, False, True],
                "norm": "sym",
            },
        },
    ]

    """
    return (
        Sequential([aug_dict[i["name"]](**i["params"]) for i in aug])
        if aug is not None
        else lambda x: x
    )


class GRACE(Contrastive):
    r"""
    Contrastive learning method proposed in the paper `Deep Graph Contrastive Representation 
    Learning <https://arxiv.org/abs/2006.04131>`_. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_grace.ipynb>`_ for
    an example of usage.
    
    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`GRACE`.
        
    Args:
        dim (int): The embedding dimension.
        dropE_rate_1, dropE_rate_2 (float): The ratio of the edge dropping augmentation for 
            view 1. A number between [0,1).
        maskN_rate_1, maskN_rate_2 (float): The ratio of the node masking augmentation for
            view 2. A number between [0,1).
        **kwargs (optinal): Additional arguments of :class:`dig.sslgraph.method.Contrastive`.
    """
    
    def __init__(
        self,
        dim: int,
        aug_1: Optional[List[Dict[str, Any]]] = None,
        aug_2: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):

        views_fn = list(map(setup_views_fn, [aug_1, aug_2]))
        super(GRACE, self).__init__(objective='NCE',
                                    views_fn=views_fn,
                                    graph_level=False,
                                    node_level=True,
                                    z_n_dim=dim,
                                    proj_n='MLP',
                                    **kwargs)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GRACE removes projection heads after pre-training
        for enc, proj in super().train(encoders, data_loader, 
                                       optimizer, epochs, per_epoch_out):
            yield enc
