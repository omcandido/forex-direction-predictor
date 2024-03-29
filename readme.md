# Stock direction predictor
Using deep learning to predict the movement (up, down or none) of the forex market. The [demo](demo.ipynb) notebook demonstrates how to predict price fluctuations of the EURUSD pair using a 1D-CNN, achieving a 65% accuracy on the test set.

## Dependencies
To install the dependencies you will need python>=3.11.
If you are on windows you can do:
`pip install -r requirements-windows.txt` (using windows).

Otherwise you can manually install the following packages:

`torch torchvision torchaudio numpy pandas matplotlib tqdm scikit-learn pytest ipywidgets jupyterlab plotly`