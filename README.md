<h1>Radiomic-based thyroid disease classification techniques exploration and end-to-end model design</h1>
<h2>Abstract</h2>
Thyroid diseases affect over 300 million people annually, requiring accurate differentiation of pathologies. Imaging methods like ultrasound, SPECT, and CT aid diagnosis, but subtle cases can be missed due to human limitations and subjectivity. Automated solutions are crucial to enhance diagnostic accuracy and efficiency.

The method extracting features from Region of Interest(ROI) combined with Machine Learning has a significance improvement on the efficiency for physician classifying thyroid problems. The main process of our research can be broke down into several parts: image preprocessing including clipping, normalization and resampling, feature extraction and selection, classification and then an end-to-end presentation.

<h2>Workflow</h2>
Create input folder and output folder
Add images folder in input folder
Add labels folder in input folder
Add images folder in output folder
Add labels folder in output folder
Put all .nii.gz pics in images and labels separately
Run Feature Extractor
A resulting features.xlsx is created in the output folder
Based on the output folder, you should manually add class labels into the spreadsheet and delete all configuration columns. Then name it as features_with_labels.xlsx and put it under input folder.
Run Machine Learning on said file




**Reference** 



Sabouri, M., & Ahamed, S. (n.d.). Thyroidiomics: An Automated Pipeline for Segmentation and  Classification of Thyroid Pathologies from Scintigraphy Images.
