import os
import sys

bin_path = "./build/"

# step1 : extract features
print("FeatureExtraction")
os.system(bin_path + "FeatureExtraction " + sys.argv[1])

# step2 : compute matcher
print("ComputeMatcher")
os.system(bin_path + "ComputeMatches " + sys.argv[1])

# step3 : reconstruction
print("Reconstruction")
os.system(bin_path + "Reconstruction " + sys.argv[1])
