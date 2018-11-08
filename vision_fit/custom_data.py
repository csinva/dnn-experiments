import numpy as np

def get_binary_bars(numInputs, numDatapoints, probabilityOn):
    """
    Generate random dataset of images containing lines. Each image has a mean value of 0.
    Inputs:
        numInputs [int] number of pixels for each image, must have integer sqrt()
        numDatapoints [int] number of images to generate
        probabilityOn [float] probability of a line (row or column of 1 pixels) appearing in the image,
            must be between 0.0 (all zeros) and 1.0 (all ones)
    Outputs:
        outImages [np.ndarray] batch of images, each of size
            (numDatapoints, numInputs)
    """
    if probabilityOn < 0.0 or probabilityOn > 1.0:
        assert False, "probabilityOn must be between 0.0 and 1.0"

    # Each image is a square, rasterized into a vector
    outImages = np.zeros((numInputs, numDatapoints))
    labs = np.zeros(numDatapoints, dtype=np.int)
    numEdgePixels = int(np.sqrt(numInputs))
    for batchIdx in range(numDatapoints):
        outImage = np.zeros((numEdgePixels, numEdgePixels))
        # Construct a set of random rows & columns that will have lines with probablityOn chance
        rowIdx = [0]; colIdx = [0];
        #while not np.any(rowIdx) and not np.any(colIdx): # uncomment to remove blank inputs
        row_sel = np.random.uniform(low=0, high=1, size=numEdgePixels) < probabilityOn
        col_sel = np.random.uniform(low=0, high=1, size=numEdgePixels) < probabilityOn
        rowIdx = np.where(row_sel)
        colIdx = np.where(col_sel)
        if np.any(rowIdx):
            outImage[rowIdx, :] = 1
        if np.any(colIdx):
            outImage[:, colIdx] = 1
        outImages[:, batchIdx] = outImage.reshape((numInputs))
        labs[batchIdx] = int(np.sum(row_sel) + np.sum(col_sel))
    return outImages.T, labs