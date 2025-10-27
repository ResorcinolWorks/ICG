# Image Caption Generator

This project generates captions for images using a deep learning model. It combines a Convolutional Neural Network (CNN) to extract features from images and a Long Short-Term Memory (LSTM) network to generate descriptive text sequences.

## Model Architecture

The model uses a CNN-LSTM architecture:
-   **CNN (DenseNet201)**: A pre-trained DenseNet201 model is used as the image feature extractor. It processes an input image and outputs a feature vector that represents the image's content.
-   **LSTM**: The feature vector from the CNN is fed into an LSTM network, which is a type of Recurrent Neural Network (RNN). The LSTM generates a caption word by word, based on the image features and the words it has already generated.

## Dataset

This project is trained on the Flickr8K dataset, which contains 8,000 images, each with five different captions.

## How to Run

To run this project, you will need to have the Flickr8K dataset.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ResorcinolWorks/ICG.git
    cd ICG
    ```

2.  **Install dependencies:**
    This project requires Python and several libraries. You can install them using pip:
    ```bash
    pip install numpy pandas tensorflow matplotlib seaborn
    ```

3.  **Download the dataset:**
    Download the Flickr8K dataset and place the `Images` folder and `captions.txt` file in a directory accessible by the notebook. You may need to update the paths in the `CNN-LSTM project.ipynb` notebook.

4.  **Run the Jupyter Notebook:**
    Open and run the `CNN-LSTM project.ipynb` notebook in a Jupyter environment.

## Future Work
- Train the model on a larger dataset like Flickr30k.
- Implement an attention mechanism to improve caption quality.
- Evaluate the model using metrics like BLEU score.
