import os, shutil
from PIL import Image


# src_dir = os.path.abspath(r"D:\\lhq\\catdog\\train\\")
# dst1_dir = os.path.abspath(r"D:\\lhq\\catdog\\train\\dog")
# dst2_dir = os.path.abspath(r"D:\\lhq\\catdog\\train\\cat")
# for root,dirs,files in os.walk(src_dir):
#         for file in files:
#             if file[0:3] == 'dog':
#                 src_file = os.path.join(root, file)
#                 shutil.copy(src_file, dst1_dir)
#             else:
#                 src_file = os.path.join(root, file)
#                 shutil.copy(src_file, dst2_dir)

path='D:\\lhq\\catdog\\test\\pic'
for file in os.listdir(path):
    data=Image.open('D:\\lhq\\catdog\\test\\pic\\'+file)
    data = data.convert('RGB')
    data.save('D:\\lhq\\catdog\\test\\pic\\'+file)


