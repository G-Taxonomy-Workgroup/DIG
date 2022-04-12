from .contrastive import Contrastive
from dig.sslgraph.method.contrastive.views_fn import NodeAttrMask, EdgePerturbation, \
    UniformSample, RWSample, RandomView
from gtaxo_graphgym.transform.perturbations.spectral import BandpassFiltering, WaveletBankFiltering, FiedlerFragmentation


class GraphCL(Contrastive):
    r"""    
    Contrastive learning method proposed in the paper `Graph Contrastive Learning with 
    Augmentations <https://arxiv.org/abs/2010.13902>`_. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_graphcl.ipynb>`_ for
    an example of usage.

    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`GraphCL`.
    
    Args:
        dim (int): The embedding dimension.
        aug1 (sting, optinal): Types of augmentation for the first view from (:obj:`"dropN"`, 
            :obj:`"permE"`, :obj:`"subgraph"`, :obj:`"maskN"`, :obj:`"random2"`, :obj:`"random3"`, 
            :obj:`"random4"`). (default: :obj:`None`)
        aug2 (sting, optinal): Types of augmentation for the second view from (:obj:`"dropN"`, 
            :obj:`"permE"`, :obj:`"subgraph"`, :obj:`"maskN"`, :obj:`"random2"`, :obj:`"random3"`, 
            :obj:`"random4"`). (default: :obj:`None`)
        aug_ratio (float, optional): The ratio of augmentations. A number between [0,1).
        **kwargs (optinal): Additional arguments of :class:`dig.sslgraph.method.Contrastive`.
    """

    def __init__(self, dim, aug_1=None, aug_2=None, aug_ratio=0.2, **kwargs):

        views_fn = []
        aug_dict = {
            'dropN': UniformSample(ratio=aug_ratio),
            'permE': EdgePerturbation(ratio=aug_ratio),
            'subgraph': RWSample(ratio=aug_ratio),
            'maskN': NodeAttrMask(mask_ratio=aug_ratio),
            'random2': RandomView([UniformSample(ratio=aug_ratio),
                                   RWSample(ratio=aug_ratio)]),
            'random3': RandomView([UniformSample(ratio=aug_ratio),
                                   RWSample(ratio=aug_ratio),
                                   EdgePerturbation(ratio=aug_ratio),
                                   NodeAttrMask(mask_ratio=aug_ratio)]),
            'random4': RandomView([UniformSample(ratio=aug_ratio),
                                   RWSample(ratio=aug_ratio),
                                   EdgePerturbation(ratio=aug_ratio)]),
            'BPhi': BandpassFiltering(band='hi'),
            'BPmid': BandpassFiltering(band='mid'),
            'BPlo': BandpassFiltering(band='lo'),
            'WBhi': WaveletBankFiltering(bands=[True, False, False]),
            'WBmid': WaveletBankFiltering(bands=[False, True, False]),
            'WBlo': WaveletBankFiltering(bands=[False, False, True]),
            'Fiedler': FiedlerFragmentation()
        }

        for aug in [aug_1, aug_2]:
            if aug is None:
                views_fn.append(lambda x: x)
            elif aug not in aug_dict:
                raise ValueError(
                    f"Unknown augmentation {aug!r}, available options are: "
                    f"{sorted(aug_dict)} or None",
                )
            else:
                views_fn.append(aug_dict[aug])

        super(GraphCL, self).__init__(objective='NCE',
                                      views_fn=views_fn,
                                      z_dim=dim,
                                      proj='MLP',
                                      node_level=False,
                                      **kwargs)

    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GraphCL removes projection heads after pre-training
        for enc, proj in super(GraphCL, self).train(encoders, data_loader,
                                                    optimizer, epochs, per_epoch_out):
            yield enc
