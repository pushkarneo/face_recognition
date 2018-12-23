import numpy as np
from lfw_load import load_data,readImage,normalise
import sys
import os
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D,Conv2D,Dense,MaxPooling2D,BatchNormalization,Input,ReLU,Flatten,Activation,Dropout
from keras import backend as K
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import argparse

class CNN:
	def __init__(self, input_shape = (125,125,6), batch_size = 128,drop_rate = 0.01, epochs = 1000):
		self.batch_size = batch_size
		self.drop_rate = drop_rate
		self.epochs = epochs
                self.input_shape = input_shape
		self.model = ()

	def build_model(self, use_vgg_weights = True):
		
		if use_vgg_weights:

			inpt = Input(shape = self.input_shape)
			x = Conv2D(3,(1,1),activation='relu')(inpt)
                        vggmodel = applications.VGG19(weights="imagenet", include_top = False,input_shape = (125,125,3))
			
			for layer in vggmodel.layers:
			    layer.trainable=False

			x = vggmodel(x)
			x = GlobalAveragePooling2D()(x)
			x = Dense(1024,activation='relu')(x)
			x = Dense(1024,activation='relu')(x)
			x = Dense(1,activation='sigmoid')(x)

			self.model = Model(inputs = (inpt), outputs = (x))

			
		else:
			model = Sequential()

			model.add(Conv2D(32,(5,5),input_shape=self.input_shape))
			model.add(ReLU())
			model.add(BatchNormalization())
			model.add(Dropout(self.drop_rate))

			model.add(Conv2D(64,(3,3)))
			model.add(ReLU())
			model.add(BatchNormalization())
			model.add(MaxPooling2D(pool_size=(2,2)))
			model.add(Dropout(self.drop_rate))

			model.add(Conv2D(128,(3,3)))
			model.add(ReLU())
			model.add(BatchNormalization())
			model.add(MaxPooling2D(pool_size=(2,2)))
			model.add(Dropout(self.drop_rate))

			model.add(Conv2D(256,(1,1)))
			model.add(ReLU())
			model.add(BatchNormalization())
			model.add(MaxPooling2D(pool_size=(2,2)))
			model.add(Dropout(self.drop_rate))

			model.add(Conv2D(512,(1,1)))
			model.add(ReLU())
			model.add(BatchNormalization())
			model.add(MaxPooling2D(pool_size=(2,2)))
			model.add(Dropout(self.drop_rate))

			model.add(Flatten())

			model.add(Dense(100))
			model.add(BatchNormalization())
			model.add(ReLU())
			model.add(Dropout(self.drop_rate))

			model.add(Dense(1))
			model.add(Activation("sigmoid"))
			self.model = model

		self.model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['binary_accuracy'])
		return self.model

	def train(self, datagen, x_train, y_train, x_test, y_test,weights_dir):
                for epoch in range(1,self.epochs):
		    self.model.fit_generator( datagen.flow(x_train, y_train, batch_size = self.batch_size), steps_per_epoch = len(x_train)/self.batch_size, epochs = 1, validation_data = (x_test, y_test))
                    if epoch % 50 == 0:
                        if not os.path.isdir(weights_dir):
                            os.mkdir(weights_dir)
                        self.model.save_weights(weights_dir + "/weights_checkpoint_{}_epoch.hdf5".format(epoch))
    
        def predict(self,im):
                return self.model.predict(im)
        
        def load_weights(self,weights):
            self.model.load_weights(weights)
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-image1', default='./example/image1.jpeg')
    ap.add_argument('-infer_similarity', type=bool, default=False)
    ap.add_argument('-image2',default='./example/image2.jpeg')
    ap.add_argument("-train",type=bool,default=True)
    ap.add_argument("-weights",default='./weights/weights_checkpoint_90_epoch.hdf5')
    args = vars(ap.parse_args())

    storage_data_type = np.float32
    storage_size = (125,125)
    
    batch_size = 128
    drop_rate = 0.01
    epochs = 1000
    weights_dir = "weights"

    cnn = CNN(batch_size = batch_size, drop_rate = drop_rate, epochs = epochs)
    model = cnn.build_model(use_vgg_weights = False)
    print(model.summary())
    
    if args['infer_similarity']:
        img1 = readImage(args['image1'],storage_data_type,storage_size)
        img2 = readImage(args['image2'],storage_data_type,storage_size)
        im = np.concatenate((img1,img2),axis=2)
        im = normalise(im)
        im = np.expand_dims(im,axis=0)
        cnn.load_weights(args['weights'])
        similarity_score = cnn.predict(im)
        print("Similarity Score : " + str(similarity_score))
        if similarity_score > 0.5:
            print("The two images are of same person")
        else:
            print("The two images are not of same person")
    else:
        (x_train,y_train) = load_data("pairsDevTrain.txt", storage_data_type, storage_size)
        (x_test,y_test) = load_data("pairsDevTest.txt", storage_data_type, storage_size)

        x_train,y_train = shuffle(x_train,y_train,random_state=0)
        x_test,y_test = shuffle(x_test,y_test,random_state=0)

        datagen = ImageDataGenerator(rotation_range=10,width_shift_range=0.2,height_shift_range=0.2)

        datagen.fit(x_train)
        cnn.train(datagen, x_train, y_train, x_test, y_test,weights_dir)






