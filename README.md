# ReLICv2

This implementation provides the linear evaluation code and checkpoints for ReLICv2, a self-supervised method introduced in the paper: [
Pushing the limits of self-supervised ResNets: Can we outperform supervised learning without labels on ImageNet?](https://arxiv.org/abs/2201.05119)

The module `eval_experiment.py` trains a linear classifier on ImageNet and evaluates
the performance of the frozen encoder/representation learnt by ReLICv2 on the
ImageNet test set.

The code provided in repository relies heavily on the BYOL code
(https://github.com/deepmind/deepmind-research/tree/master/byol).

## Setup

To set up a Python virtual environment with the required dependencies, run:

```shell
python3 -m venv relicv2_env
source relicv2_env/bin/activate
pip install --upgrade pip
pip install -r relicv2_env/requirements.txt
```

The code uses `tensorflow_datasets` to load the ImageNet dataset. Manual
download may be required; see
https://www.tensorflow.org/datasets/catalog/imagenet2012 for details.

## Full pipeline for linear evaluation on ImageNet

The various parts of the pipeline can be run using:

```shell
python -m relicv2.main_loop \
  --worker_mode=<'train' or 'eval'> \
  --checkpoint_root=</path/to/the/checkpointing/folder> \
```

Use `--worker_mode=train` for a training job, which will load the encoder
weights from an existing checkpoint (form a pretrain experiment) located at
`<checkpoint_root>/pretrain.pkl`, and train a linear classifier on top of this
encoder. The main loop for linear evaluation runs for 100 epochs.

The training job will regularly save checkpoints under
`<checkpoint_root>/linear-eval.pkl`. You can run a second worker (using
`--worker_mode=eval`) with the same `checkpoint_root` setting to regularly load
the checkpoint and evaluate the performance of the classifier (trained by the
linear-eval `train` worker) on the test set.

Note that the config/eval.py is set-up for using the ResNet50 1x architecture.
If you want to run the code for different architectures, please change the
encoder_class in config/eval.py.

## ReLICv2 Checkpoints

We provide the following ReLICv2 checkpoints for different ResNet architectures.

-   [ResNet-50 1x](https://drive.google.com/file/d/1PqLiSdGA8zCRVxvdb_6yQYBiTrtWqETn/view?usp=share_link)
-   [ResNet-50 2x](https://drive.google.com/file/d/1R26CfMZHRPeoHqgY7uEGzgze00fkoudx/view?usp=share_link)
-   [ResNet-50 4x](https://drive.google.com/file/d/1ZE-Q6zuDfXyYqHjoLdGMNa7dk8npdjJa/view?usp=share_link)
-   [ResNet-101](https://drive.google.com/file/d/1SXX55JsV8F168hkk2pjnsqMkZjq5jO1I/view?usp=share_link)
-   [ResNet-152](https://drive.google.com/file/d/1haWbrCB7IF7O7yENvgAYnsDI7AsF1wBF/view?usp=share_link)
-   [ResNet-200](https://drive.google.com/file/d/1DaTyJj_5HDaOBOVH9PVD7U-Eqh_qeruM/view?usp=share_link)
-   [ResNet-200 2x](https://drive.google.com/file/d/1uPkFECQV-EbCrn22juz6gaqmP87KaLAD/view?usp=share_link)

## Citing this work

If you use this code please cite:

```
@article{tomasev2022pushing,
  title={Pushing the limits of self-supervised ResNets: Can we outperform supervised learning without labels on ImageNet?},
  author={Tomasev, Nenad and Bica, Ioana and McWilliams, Brian and Buesing, Lars and Pascanu, Razvan and Blundell, Charles and Mitrovic, Jovana},
  journal={arXiv preprint arXiv:2201.05119},
  year={2022}
}
```

## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
