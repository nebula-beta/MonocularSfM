import os

# image path
images_path = ""
# database
database_path = ""
# 输出结果所在的路径
output_path = "./"

# 相机内参
fx = 0
fy = 0
cx = 0
cy = 0

# 畸变参数
k1 = 0
k2 = 0
p1 = 0
p2 = 0

bin_path = "./bin/"

os.system(bin_path + "FeatureExtraction " + images_path + " " + database_path)
os.system(bin_path + "ComputeBruteMatches " + database_path)
# os.system("./ComputeSequentialMatches " + database_path)
if k1 == 0:
    os.system(bin_path + "Reconstruction " + database_path + " " + str(fx) + " " + str(fy) + " " + str(cx) + " " + str(cy) + " " + output_path)
else:
    os.system(bin_path + "Reconstruction " + database_path + " " + str(fx) + " " + str(fy) + " " + str(cx) + " " + str(cy) + " " +
                                                          str(k1) + " " + str(k2) + " " + str(p1) + " " + str(p2) + " " + output_path)
