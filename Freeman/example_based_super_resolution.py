import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors
import random
import h5py

class imset:
    def __init__(self, path, create = True, scale = 0.25):
        self.path = path
        self.pathSource = path + '/Source'
        self.items_name = os.listdir(self.pathSource)
        self.pathFiles = [path+'/'+f for f in self.items_name]
        self.scale = scale
        self.images = self.get_images(path+'/Source')
        if create:
            self.lowResolution, self.lowResolutionNormalized = self.make_low_resolution()
            self.interpolated, self.interpolatedNormalized = self.make_interpolation()
            self.HR_post = self.make_HR_postprocess()
        else:
            self.lowResolution = self.get_images(path+'/LR')
            self.lowResolutionNormalized = self.get_images(path+'/LRN')
            self.interpolated = self.get_images(path+'/IN')
            self.interpolatedNormalized = self.get_images(path+'/INN')
            self.HR_post = self.get_images(path+'/HR')

    def get_images(self, path):
        if not os.path.exists(path):
            print('Verifique que el directorio {} exista'.format(path))
            return []
        else: 
            items_name = os.listdir(path)
            pathFiles = [path+'/'+f for f in items_name]
            return [cv2.imread(image) for image in pathFiles]

    def make_low_resolution(self):
        """
        Función para reducir las dimensiones de una imagen a partir de un factor de escalado.
        """
        lowResolution = [cv2.resize(img, None, fx = self.scale, fy = self.scale) for img in self.images]
        lowResolutionNormalized = [self.postprocess(img) for img in lowResolution]
        self.save_dataset(lowResolution, self.path+'/LR')
        self.save_dataset(lowResolutionNormalized, self.path+'/LRN')
        return lowResolution, lowResolutionNormalized

    def make_interpolation(self):
        interpolated = [cv2.resize(img, None, fx = 1/self.scale, fy = 1/self.scale, interpolation = cv2.INTER_CUBIC)
                        for img in self.lowResolution]
        interpolatedNormalized = [self.postprocess(img) for img in interpolated]
        self.save_dataset(interpolated, self.path+'/IN')
        self.save_dataset(interpolatedNormalized, self.path+'/INN')
        return interpolated, interpolatedNormalized

    def make_HR_postprocess(self):
        HR_post = [self.postprocess(img) for img in self.images]
        self.save_dataset(HR_post, self.path+'/HR')
        return HR_post

    def postprocess(self,image):
        kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
        kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)
        #filter the source image
        image_filter = cv2.filter2D(image,-1,kernel)
        mean = np.max(image_filter)
        return image_filter/mean

    def save_dataset(self, data, outpath):
        outpath_files = [outpath+'/'+name for name in self.items_name]
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        count = 0
        for img in data:
            cv2.imwrite(outpath_files[count],img)
            count += 1

class training_data:
    def __init__(self, imHRN, imLRN, sizes):
        self.imHRN = imHRN
        self.imLRN = imLRN
        self.N = sizes[0]
        self.M = sizes[1]
        self.patchesHR, self.vectorsID = self.imgs_to_patches()
        
    def save_data(self):
        f = h5py.File('training_data.h5', 'w')
        f.create_dataset(name = 'patches', data = self.patchesHR)
        f.create_dataset(name = 'vectors', data = self.vectorsID)
        f.close()

    def imgs_to_patches(self):
        n_images = len(self.imHRN)
        patchesHR = []
        vectorsID = []
        for i in range(n_images):
            print('Descomponiendo en parches imagen {}. Espere por favor... '.format(i+1))
            if i == 0:
                patchesHR, vectorsID = self.make_patches(i)
            else:
                tempHR, tempID = self.make_patches(i)
                patchesHR = np.row_stack((patchesHR, tempHR))
                vectorsID = np.row_stack((vectorsID, tempID))
        return patchesHR, vectorsID
 
    def make_patches(self, k):
        imgLR = self.imLRN[k]
        imgHR = self.imHRN[k]
        sz = imgHR.shape
        cnt = 0
        for i in range(2,sz[0]-1,4):
            for j in range(2,sz[1]-1,4):
                px = (i,j)
                cnt += 1
                patchHR = self.get_patch(imgHR, px, self.N)
                patchLR = self.get_patch(imgLR, px, self.M)
                
                tempLR = patchLR.flatten()
                tempHR = patchHR.flatten()

                f_row = patchHR[0,:,:]
                f_col = patchHR[:,0,:]
                f_row = f_row.flatten()
                f_col = f_col.flatten()
                tempSupp = np.concatenate((f_row, f_col))

                tempID = np.concatenate((tempLR, tempSupp))

                if cnt == 1:
                    dataID = tempID
                    dataHR = tempHR                
                else:
                    dataID = np.row_stack((dataID, tempID))
                    dataHR = np.row_stack((dataHR, tempHR))
        return dataHR, dataID

    def get_patch(self, image, center, size):
        patch = np.zeros((size, size,3))
        offs = int((size - 1 )/2)
        img_border = self.make_borders(image, offs)
        patch = img_border[offs+center[0]-offs:offs+center[0]+offs+1, offs+center[1]-offs:offs+center[1]+offs+1, :]
        return patch
    
    def make_borders(self, image, offs):
        image_border = cv2.copyMakeBorder(image, offs, offs, offs, offs, cv2.BORDER_CONSTANT, value = (0,0,0))
        return image_border


class superresolution:
    def __init__(self, path, scale, alpha):
        self.path = path
        self.scale = scale
        self.image = self.get_image()
        self.image_input = self.preprocess()
        self.patchesHR, self.vectorsID = self.get_training_data()
        self.N = 5
        self.M = 7
        self.alpha = alpha
        self.scale_up = []
        sz = self.image_input.shape
        offs = int((self.N - 1 )/2)
        print(sz)
        self.high_frequencies = np.zeros((sz[0],sz[1],3))
        self.superresolution = self.algorithm(sz)


    def preprocess(self):
        kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
        kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)
        #filter the source image
        image_filter = cv2.filter2D(self.image,-1,kernel)
        return cv2.resize(image_filter, None, fx = self.scale, fy = self.scale, interpolation = cv2.INTER_CUBIC)

    def get_training_data(self):
        f = h5py.File('training_data.h5', 'r')
        patchesHR = f.get('patches')
        vectorsID = f.get('vectors')
        return patchesHR, vectorsID

    def get_image(self):
        return cv2.imread(self.path)

    def meanAbs(self, image):
        return abs(np.mean(image)) + 0.0000000000000000000000001

    def predict_high_frequencies(self, sz):
        cnt = 0
        nbrs = NearestNeighbors(n_neighbors = 1, algorithm = 'ball_tree').fit(self.vectorsID)
        print(self.high_frequencies.shape)
        for i in range(2,sz[0]-1,4):
            for j in range(2,sz[1]-1,4):
                px = (i,j)
                cnt += 1
                patchHF = self.get_patch(self.high_frequencies, px, self.N)
                patchLR = self.get_patch(self.image_input, px, self.M)
                
                mean = self.meanAbs(patchLR)
                
                tempLR = patchLR.flatten()

                f_row = patchHF[0,:,:]
                f_col = patchHF[:,0,:]

                f_row = f_row.flatten()
                f_col = f_col.flatten()

                tempSupp = np.concatenate((f_row, f_col)) * self.alpha

                vector_search = np.concatenate((tempLR, tempSupp))/mean

                _, index = nbrs.kneighbors([vector_search])
                index = index[0]

                self.put_patch(px, 5, index, mean)

    def put_patch(self, center, size, index, mean):
        patchHR = self.patchesHR[index]
        patchHR = np.reshape(patchHR, (size, size, 3))
        patchHR = patchHR*mean
        #print('XXXXX:', patchHR.shape)
        offs = int((size - 1 )/2)
        img_temp = self.high_frequencies
        img_temp = self.make_borders(img_temp, offs)
        img_temp[offs+center[0]-offs:offs+center[0]+offs+1, offs+center[1]-offs:offs+center[1]+offs+1, :] = patchHR
        self.high_frequencies = img_temp[offs:-offs, offs:-offs, :]

    def get_patch(self, image, center, size):
        patch = np.zeros((size, size,3))
        offs = int((size - 1 )/2)
        img_border = self.make_borders(image, offs)
        patch = img_border[offs+center[0]-offs:offs+center[0]+offs+1, offs+center[1]-offs:offs+center[1]+offs+1, :]
        return patch

    def algorithm(self, sz):
        self.scale_up = cv2.resize(self.image, None, fx = self.scale, fy = self.scale, interpolation = cv2.INTER_CUBIC)
        self.predict_high_frequencies(sz)
        return self.high_frequencies + self.scale_up

    def make_borders(self, image, offs):
        image_border = cv2.copyMakeBorder(image, offs, offs, offs, offs, cv2.BORDER_CONSTANT, value = (0,0,0))
        return image_border