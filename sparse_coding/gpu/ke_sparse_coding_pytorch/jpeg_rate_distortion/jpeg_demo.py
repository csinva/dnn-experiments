#! /usr/bin/env python
if __name__ == '__main__':

    import argparse
    import src.utils
    import matplotlib.pyplot as plt
#     from optparse import OptionParser
    # %matplotlib notebook
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type = str)
    args = parser.parse_args()
    bit_rate = []
    re_rate = []
    lamda = [0.2,0.5,1,2,3,4,5,8,10,15,20,30,40,50]
    for lam in lamda:
        print(lam)
        _,_,bit,re = src.utils.jpeg_rate_distortion(args.i,lam)
        bit_rate.append(bit)
        re_rate.append(re)
    plt.plot(re_rate,bit_rate);plt.title("Rate Distortion Performance")
    plt.xlabel("Distortion")
    plt.ylabel("Rate")
    plt.show()