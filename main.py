import os
import cv2
import numpy as np


class nameImage:
	def __init__(self, img_path,img_name):
		self.img = cv2.imread(img_path+img_name)
		self.name = img_name
	def __str__(self):
		return self.__name



def folderSetup():
	path=os.path.abspath(os.getcwd())
	img_path= path +'/img/'
	try:
		os.path.exists(img_path)
	except:
		print('Please create an /img subfolder for the original pictures.')

	rot_path=path+'/img_rotated/'
	flip_path=path+'/img_flipped/'
	noise_path= path+'/img_noise/'
	crop_path=path+'/img_cropped/'

	if not os.path.exists(rot_path):
    		os.makedirs(rot_path)
	if not os.path.exists(flip_path):
		os.makedirs(flip_path)
	if not os.path.exists(noise_path):
    		os.makedirs(noise_path)
	if not os.path.exists(crop_path):
    		os.makedirs(crop_path)
	paths=[img_path, rot_path,flip_path, noise_path, crop_path]
	return paths

def rotateImage(img,rot_path):
	print("Rotating image...")
	img_name= str(img.name)
	rot_img = cv2.rotate(img.img, cv2.ROTATE_90_CLOCKWISE)
	file= rot_path + 'rot90_' + img_name
	cv2.imwrite(file, rot_img)
	rot_img = cv2.rotate(img.img, cv2.ROTATE_180)
	file= rot_path + 'rot180_' + img_name
	cv2.imwrite(file, rot_img)
	rot_img = cv2.rotate(img.img, cv2.ROTATE_90_COUNTERCLOCKWISE)
	file= rot_path + 'rot270_' + img_name
	cv2.imwrite(file, rot_img)
	print("Done, saved to:")
	print(rot_path)

def flipImage(img, flip_path):
	print("Flipping image..")
	flipV = cv2.flip(img.img, 0)
	flipH = cv2.flip(img.img, 1)
	flipB = cv2.flip(img.img, -1)
	cv2.imwrite(flip_path+'flipV_'+str(img.name), flipV)
	cv2.imwrite(flip_path+'flipH_'+str(img.name), flipH)
	cv2.imwrite(flip_path+'flipB_'+str(img.name), flipB)
	print("Done, saved to:")
	print(flip_path)

def noiseImage(img, noise_path):
	print('Applying noise..')
	name=str(img.name)
	img=img.img
	gauss = np.random.normal(0,1,img.size)
	gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
	# Add the Gaussian noise to the image
	img_gauss = cv2.add(img,gauss)
	img_speckle = img + img * gauss
	cv2.imwrite(noise_path+'gauss_'+name, img_gauss)
	cv2.imwrite(noise_path+'speckle_'+name, img_speckle)
	print('Done, saved to:')
	print(noise_path)


def cropImage(img, crop_path):
	print('Cropping image..')
	height = img.img.shape[0]
	width = img.img.shape[1]
	percent=0.05
	h_lim=int(percent*height)
	w_lim=int(percent*width)
	for i in range(2):
		h_start = np.random.randint(0, h_lim)
		h_end= np.random.randint(height-h_lim, height)
		w_start = np.random.randint(0, w_lim)
		w_end = np.random.randint(width-w_lim, width)
		crop = img.img[h_start : h_end, w_start : w_end]
		cv2.imwrite(crop_path+"crop_"+str(i)+"_"+str(img.name),crop)

	print("Dome, saved to: ")
	print(crop_path)

def main():
	[img_path,rot_path,flip_path,noise_path,crop_path]=folderSetup()

	for image in os.listdir(img_path):
		img = nameImage(img_path,image)
		print("Image: " + str(img.name))
		rotateImage(img,rot_path)
		flipImage(img,flip_path)
		noiseImage(img,noise_path)
		cropImage(img,crop_path)
	print("Flipping, adding noise, and cropping the rotated images..")
	for image in os.listdir(rot_path):
		img= nameImage(rot_path,image)
		flipImage(img,flip_path)
		noiseImage(img,noise_path)
		cropImage(img,crop_path)
	print("Applying noise and cropping flipped images..")
	for image in os.listdir(flip_path):
		img= nameImage(flip_path,image)
		noiseImage(img,noise_path)
		cropImage(img,crop_path)
	print("Cropping images with noise..")
	for image in os.listdir(noise_path):
		img=nameImage(noise_path,image)
		cropImage(img,crop_path)

if __name__== "__main__":
	main()
