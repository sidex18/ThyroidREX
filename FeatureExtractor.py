
# Importing the required libraries
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import glob


class RadiomicsFeatureExtractor:
    def __init__(self, params):
        """
        Initialize the RadiomicsFeatureExtractor with specified parameters.
        """
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    def extract_features(self, stpaths, gtpaths, output_excel_file):
        """
        Extract radiomics features from given paths for images and labels.
        """
        sample_names = []
        feature_data = []

        # Process each image-label pair
        for image_path, label_path in zip(stpaths, gtpaths):
            try:
                label_image = sitk.ReadImage(label_path)
                image = sitk.ReadImage(image_path)
                features = self.extractor.execute(image, label_image)

                sample_name = os.path.basename(image_path).split('.')[0]
                sample_names.append(sample_name)
                feature_data.append(features)
            except Exception as e:
                print(f"Error processing {image_path} or {label_path}: {e}")

        # Create and save the DataFrame
        # if feature_data:
        df = pd.DataFrame(feature_data)
        df.insert(0, 'PatientID', sample_names)
        
        # Ensure the output directory exists
        # os.makedirs(os.path.dirname(output_excel_file), exist_ok=True)
        df.to_excel(output_excel_file, index=False)
        print("Radiomics features saved to", output_excel_file)



# Main program 


params = {
        "binWidth": 0.1,                 # 0.1
        "resampledPixelSpacing": (1, 1), # Resampling
        "interpolator": sitk.sitkBSpline,# Correct the interpolator setting
        "force2D": True,                 # Force the extractor to treat the images as 2D
        "normalize": True,               # Normalize the image before calculation
    }


featExtractor = RadiomicsFeatureExtractor(params)

#Loading and reading images and label files 

imagesFolder = "rex_dataset/images"

labelsFolder = "rex_dataset/labels"

stPaths = sorted(glob.glob(os.path.join(imagesFolder, "*.nii.gz")))
gtPaths = sorted(glob.glob(os.path.join(labelsFolder, "*.nii.gz")))


#Feature Extraction 
featExtractor.extract_features(stPaths,gtPaths,"features.xlsx")
