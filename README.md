# dress-features-recognition

### Task
- Interactively returns the product features of the uploaded image of women's dress. (using Streamlit)

### Approach
- A multitask classification is performed by training a neural network.
- A model with multiple outputs is trained to learn the product features (color, product
- type, material) from the image dataset by using the product features as labels.
- Streamlit has been used to make interactive user interface.

### Data

- The database was acquired from Kaggle.
- Database name: All Products From Myntra.com 2019
- Databse url: https://www.kaggle.com/datasets/promptcloud/all-products-from-myntracom-2019

### Model:
- EfficientNet-V2-B1 has been used as the base model.
- The top layers after the base model diverge into three branches for three
  different output, one each for predicting color, product type, and material.
- Transfer learning with further fine-tuning was used to train the model.

### This project is still a work in progress. The preformance can be improved by doing some more experiments.

'model_training.ipynb' contains code with detailed explanation of the whole process.
'interactive_dress_features_recognition.py' contains the code for interactive user interface which recoginizes the product features
