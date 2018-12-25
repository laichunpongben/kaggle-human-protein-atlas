# kaggle-human-protein-atlas
Sharing code for Kaggle Human Protein Atlas competition

## Data
Color filters for the fluorescent images

  * Green (Alexa Flour 488): The protein of interest
  * Blue (DAPI): Nucleus
  * Red: Microtubules
  * Yellow: Endoplasmic reticulum

Train sample size: 31072

Test sample size:  11702

## Labels / Cell component / # images in train set
  0.   Nucleoplasm			12885
  1.   Nuclear membrane			1254
  2.   Nucleoli				3621
  3.   Nucleoli fibrillar center	1561
  4.   Nuclear speckles			1858
  5.   Nuclear bodies			2513
  6.   Endoplasmic reticulum		1008
  7.   Golgi apparatus			2822
  8.   Peroxisomes			53
  9.   Endosomes			45
  10.  Lysosomes			28
  11.  Intermediate filaments		1093
  12.  Actin filaments			688
  13.  Focal adhesion sites		537
  14.  Microtubules			1066
  15.  Microtubule ends			21
  16.  Cytokinetic bridge		530
  17.  Mitotic spindle			210
  18.  Microtubule organizing center	902
  19.  Centrosome			1482
  20.  Lipid droplets			172
  21.  Plasma membrane			3777
  22.  Cell junctions			802
  23.  Mitochondria			2965
  24.  Aggresome			322
  25.  Cytosol				8228
  26.  Cytoplasmic bodies		828
  27.  Rods & rings			11

## Quickstart
Resnet FastAI

`python3 -m code.resnet_fastai --imagesize=256 --loss=focal --gpuid=0`

`python3 -m code.resnet_fastai --model=stage-2-resnet50-224-drop0.5-ep5_15 --gpuid=0`

  * -a, --arch         : Neural network architecture
  * -b, --batchsize    : batch size
  * -d, --encoderdepth : encoder depth of the network
  * -D, --dataset      : Dataset
  * -e, --epochnum1    : epoch number for stage 1
  * -E, --epochnum2    : epoch number for stage 2
  * -f, --fold         : K fold cross validation
  * -i, --gpuid        : GPU device id
  * -l, --loss         : loss function
  * -m, --model        : trained model to load
  * -p, --dropout      : dropout ratio
  * -r, --learningrate : learning rate
  * -s, --size         : image size
  * -S, --sampler      : sampler
  * -t, --thres        : threshold
  * -v, --verbose      : set verbosity 0-3, 0 to turn off output (not yet implemented)


Mask RCNN

`python3 -m code.mask_rcnn train --dataset=data/official --subset=train --weights=coco`

`python3 -m code.mask_rcnn detect --dataset=data/official --subset=test --weights=logs/nucleus20181212T0318/mask_rcnn_nucleus_0025.h5`
