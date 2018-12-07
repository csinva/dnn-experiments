if __name__ == '__main__':
    import argparse
    import sigpy.plot as pl
    import numpy as np
    import os
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    
    with open("results/bit_manual.txt", "rb") as fp:   # Unpickling
        bit_manual = pickle.load(fp)
    with open("results/rmse_manual.txt", "rb") as fp:   # Unpickling
        rmse_manual = pickle.load(fp)
    with open("results/bit_ffmpeg.txt", "rb") as fp:   # Unpickling
        bit_ffmpeg = pickle.load(fp)
    with open("results/rmse_ffmpeg.txt", "rb") as fp:   # Unpickling
        rmse_ffmpeg = pickle.load(fp)
            
    fig = plt.figure(figsize=(15,8))
    plt.title("Rate Distortion Performance")
    plt.plot(rmse_manual,bit_manual,color = 'r', linestyle='-',linewidth = 5)
    plt.plot(rmse_ffmpeg,bit_ffmpeg,color = 'b', linestyle='-',linewidth = 5)
    plt.legend(["Manual Implementation","ffmpeg Implementation"],frameon=False)
    plt.xlabel("Distortion")
    plt.ylabel("Rate")
    plt.ylim([0,2])
    fig.savefig('results/plot_comparason.png')
    plt.show()