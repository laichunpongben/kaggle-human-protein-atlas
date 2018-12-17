# kaggle-human-protein-atlas
Sharing code for Kaggle Human Protein Atlas competition

## Data
Color filters for the fluorescent images

  * Green (Alexa Flour 488): The protein of interest
  * Blue (DAPI): Nucleus
  * Red: Microtubules
  * Yellow: Endoplasmic reticulum

Training sample size: 124288

## Labels
  0.   Nucleoplasm
  1.   Nuclear membrane
  2.   Nucleoli
  3.   Nucleoli fibrillar center
  4.   Nuclear speckles
  5.   Nuclear bodies
  6.   Endoplasmic reticulum
  7.   Golgi apparatus
  8.   Peroxisomes
  9.   Endosomes
  10.  Lysosomes
  11.  Intermediate filaments
  12.  Actin filaments
  13.  Focal adhesion sites
  14.  Microtubules
  15.  Microtubule ends
  16.  Cytokinetic bridge
  17.  Mitotic spindle
  18.  Microtubule organizing center
  19.  Centrosome
  20.  Lipid droplets
  21.  Plasma membrane
  22.  Cell junctions
  23.  Mitochondria
  24.  Aggresome
  25.  Cytosol
  26.  Cytoplasmic bodies
  27.  Rods & rings

## Quickstart
Resnet

`python3 -m code.resnet_fastai --imagesize=256 --loss=focal --gpuid=0`

`python3 -m code.resnet_fastai --model=stage-2-resnet50-224-drop0.5-ep5_15 --gpuid=0`

Mask RCNN

`python3 -m code.mask_rcnn train --dataset=data/official --subset=train --weights=coco`

`python3 -m code.mask_rcnn detect --dataset=data/official --subset=test --weights=logs/nucleus20181212T0318/mask_rcnn_nucleus_0025.h5`
