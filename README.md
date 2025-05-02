Radiomic-based thyroid disease classification techniques exploration and end-to-end model design
Abstract
Thyroid diseases affect over 300 million people annually, requiring accurate differentiation of pathologies. Imaging methods like ultrasound, SPECT, and CT aid diagnosis, but subtle cases can be missed due to human limitations and subjectivity. Automated solutions are crucial to enhance diagnostic accuracy and efficiency.

The method extracting features from Region of Interest(ROI) combined with Machine Learning has a significance improvement on the efficiency for physician classifying thyroid problems. The main process of our research can be broke down into several parts: image preprocessing including clipping, normalization and resampling, feature extraction and selection, classification and then an end-to-end presentation.

Instruction On using the code
Add an input folder under Max
Add an output folder under Max
Add images folder under input
Add labels folder under input
Add images folder under output
Add labels folder under output
Put all .nii.gz pics in images and labels separately
Run Feature Extractor
Then you can get features.xlsx in the output folder
Based on the output folder, you should manually add class label into the spreadsheet and delete all configuration columns. Then name it as features_with_labels.xlsx and put it under input folder.
Run Machine Learning
Then you can get the result.
Reference
Sabouri, M., & Ahamed, S. (n.d.). Thyroidiomics: An Automated Pipeline for Segmentation and  Classification of Thyroid Pathologies from Scintigraphy Images.
