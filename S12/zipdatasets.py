import zipfile
import os
import shutil


with zipfile.ZipFile('tiny-imagenet-200.zip','r') as my_zip:
     my_zip.extractall()
	 
	 
print('------------- INFO : Preparing Train and Test Loader Folder ---------------- ')


os.makedirs('TinyImageNet')
os.makedirs('TinyImageNet/train')
os.makedirs('TinyImageNet/test')
os.makedirs('TinyImageNet/val')


for classes in os.listdir('tiny-imagenet-200/train'):
  os.makedirs('TinyImageNet/train/'+classes)
  for images in os.listdir('tiny-imagenet-200/train/'+classes+'/images'):
    shutil.move('tiny-imagenet-200/train/'+classes+'/images/'+images,'TinyImageNet/train/'+classes)


print('Train Data Transfer successfully')

val_class = []

for line in open('tiny-imagenet-200/val/val_annotations.txt','r'):
  img_name , class_id = line.split('\t')[:2]
  val_class.append(class_id)


val_class = list(set(val_class))


for i in val_class:
  os.makedirs(f'TinyImageNet/val/{i}')


for line in open('tiny-imagenet-200/val/val_annotations.txt','r'):
  img_name , class_id = line.split('\t')[:2]
  shutil.move(f'tiny-imagenet-200/val/images/{img_name}',f'TinyImageNet/val/{class_id}')


print('Val Data Transfer successfully')
print('------------- INFO : Train and Test Loader Folder TinyImageNet Done ---------------- ')