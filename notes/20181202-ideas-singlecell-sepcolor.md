Segmenting single cell image instances
==

- segment into single cells (randomize single cell images from different sources?)
- separate color channels


### Motivations & Advantages:
- single cell: 
    - train by subcellular localization and texture, instead of cell shapes, densities, positions and orientations etc (Li et al., 2018)
    - data augmentation
- separate color channels: 
    - mark bounding boxes of cells with red channel 
      (usually the whole cell is filled with microtubules outside the nucleus)
    - depending on the amount and density of target protein, color may affect
    - can validate results by checking if predicted class fall into correct compartment, e.g. nucleus for nucleolus


### Proposed procedures

1. segment into single cells
    - get bounding boxes, e.g. 
        - retrain from Pascal VOC dataset
        - use imageJ
        - parse with faster r-cnn `keras_frcnn` (`git clone https://github.com/yhenon/keras-frcnn/`)
    
2. Randomize and split single cell images into training batches

3. Train
    - test different models (check references)

4. Check if falling into relevant compartments


### References

- Li, et al., 2018; Deep CNNs for HEp-2 Cells Classification : A Cross-specimen Analysis
    - https://arxiv.org/abs/1604.05816 
    - differentiate between Speckled, Nucleolar, Centromere, Nuclear Membrane, and Golgi
    - dataset: I3A contest (13596 cell images)
    - accuracy: 70% (Golgi) - 86.87% (Nuclear membrane)
    - segment into single cells
    - leave-one-specimen-out (LOSO)
    - mean class accuracy (MCA) evaluation for each split
    
- Li, et al., 2017; A Deep Residual Inception Network for HEp-2 Cell Classification
    - https://link.springer.com/chapter/10.1007/978-3-319-67558-9_2
    - Auxiliary classifiers were only used as regularizers to improve network convergence during training in original Inception.
    - Caffe toolbox

- R-CNN, Fast R-CNN, Faster R-CNN for object detection
    - https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/    
    - https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/
    - https://www.analyticsvidhya.com/blog/2018/10/a-step-by-step-introduction-to-the-basic-object-detection-algorithms-part-1/
    
- Automated Analysis and Reannotation of Subcellular Locations in Confocal Images from the Human Protein Atlas
    - https://www.ncbi.nlm.nih.gov/pubmed/23226299
    
    
- A deep convolutional neural network for classification of red blood cells in sickle cell anemia
    - https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005746
    - segment into single cell images
    - CNN
