#! /usr/bin/env python
if __name__ == '__main__':
    import argparse
    import sigpy.plot as pl
    import numpy as np
    import os
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type = str)

    args = parser.parse_args()
    # Define mean square error function
    def rmse(im_g,im_r):
        return np.sqrt(np.sum(((im_g-im_r)**2))/np.size(im_g))
    
    # Compress images
    for i in range(31):
        os.system("ffmpeg -i %s -q:v %s image_compression/image_output%s.jpg" % (args.i,str(i+1),str(i+1)))
    
    rmses = []
    bits = []
    groundtruth = cv2.imread(args.i,cv2.IMREAD_GRAYSCALE)
    for i in range(31):
        dir1 = "image_compression/image_output%s.jpg" %str(i+1)
        result = cv2.imread(dir1,cv2.IMREAD_GRAYSCALE)
#         plt.figure()
#         plt.imshow(result,cmap='gray')
#         plt.show()
        rmse_1 = rmse(result,groundtruth)
        bit = (os.path.getsize(dir1)*8)/(512*512)
        rmses.append(rmse_1)
        bits.append(bit)

    with open("results/bit_ffmpeg.txt", "wb") as fp:   #Pickling
        pickle.dump(bits, fp)
    with open("results/rmse_ffmpeg.txt", "wb") as fp:   #Pickling
        pickle.dump(rmses, fp)
    fig = plt.figure()
    plt.plot(rmses,bits);plt.title("Rate Distortion Performance")
    plt.xlabel("Distortion")
    plt.ylabel("Rate")
    fig.savefig('results/plot_ffmpeg.png')
    plt.show()