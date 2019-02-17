# facial-expressions-recognition-features-extractor
An implementation of facial images features extraction procedure
Developed to parse CK+ dataset: http://www.consortium.ri.cmu.edu/ckagree/
The feature extraction procedure goes through the dataset and extracts 2 types of image features: facial elements displacement features (based on face detection using Haar Cascades) and LBP (textural) features. 
Then both sets of features are fused in one set of features with Autoencoder
The resulting features are then fed into the self organizning map to classify a number of facial expressions presented in the dataset
