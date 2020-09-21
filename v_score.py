import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from tensorflow.keras.layers import Input, Dense
from keras.models import load_model, Model
from keras.optimizers import Adam
import os
from keras.layers.advanced_activations import LeakyReLU, ReLU
from SpectralNormalizationKeras import ConvSN2D,DenseSN
from keras.layers.convolutional import UpSampling2D, Convolution2D, Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import concatenate
from keras.layers import Input, Reshape, Embedding, Dropout, Activation, Flatten, Dense

from optparse import OptionParser


from batch_generator import BatchGenerator as BatchGenerator


from sklearn.manifold import TSNE
import time
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score 

if __name__ == '__main__':

    argParser = OptionParser()
                  
    argParser.add_option("-o",  default="ganda",
                  action="store", type="string", dest="gan",
                  help="Either 'ganda', or 'bagan', or 'acgan'.")
    (options, args) = argParser.parse_args()
    
    resolution = 28
    channels = 1
    latent_size = 100
    adam_lr=0.00005
    adam_beta_1 = 0.5
    gan = options.gan

    def build_reconstructor_bagan(latent_size, min_latent_res=8):
        image = Input(shape=( resolution, resolution ,channels ))
        features = _build_common_encoder_bagan(image, min_latent_res)

        # Reconstructor specific
        latent = Dense(latent_size, activation='linear')(features)
        reconstructor = Model(inputs=image, outputs=latent)
        return reconstructor

    def _build_common_encoder_bagan(image, min_latent_res=8):
        # build a relatively standard conv net, with LeakyReLUs as suggested in ACGAN
        cnn = Sequential()

        cnn.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2),
                       input_shape=(channels, resolution, resolution), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        while cnn.output_shape[-2] > min_latent_res:
            cnn.add(Conv2D(256, (3, 3), padding='same', strides=(2, 2), use_bias=True))
            cnn.add(LeakyReLU())
            cnn.add(Dropout(0.3))

            cnn.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), use_bias=True))
            cnn.add(LeakyReLU())
            cnn.add(Dropout(0.3))

        cnn.add(Flatten())

        features = cnn(image)
        return features

    def build_reconstructor_ganda(latent_size, min_latent_res=8):
        image = Input(shape=(resolution, resolution,channels))
        features = _build_common_encoder_ganda(image, min_latent_res)

        # Reconstructor specific

        ### encoder condition~~~~

        
        label = Input(shape=(10,)) #mnist 10 classes
        features_label= concatenate([label,features],axis=1)###~~~~~~axis=1
        
        latent = Dense(latent_size, activation='linear')(features_label)
        reconstructor = Model(inputs=[image,label], outputs=latent)
        return reconstructor

    def _build_common_encoder_ganda(image, min_latent_res=8):

        # # build a relatively standard conv net, with LeakyReLUs as suggested in ACGAN
        # cnn = Sequential()
        # cnn.add(ConvSN2D(32, kernel_size=3, strides=2,kernel_initializer='glorot_uniform', padding='same',bias=True))
        # cnn.add(LeakyReLU())
        # # cnn.add(Dropout(0.3))
        # cnn.add(ConvSN2D(64, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same',bias=True))
        # cnn.add(LeakyReLU())
        # # cnn.add(Dropout(0.3))

        # cnn.add(ConvSN2D(128, kernel_size=3, strides=2,kernel_initializer='glorot_uniform', padding='same',bias=True))
        # cnn.add(LeakyReLU())
        # # cnn.add(Dropout(0.3))

        # cnn.add(ConvSN2D(256, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same',bias=True))
        # cnn.add(LeakyReLU())
        # # cnn.add(Dropout(0.3))


        # cnn.add(ConvSN2D(256, kernel_size=3, strides=2,kernel_initializer='glorot_uniform', padding='same',bias=True))
        # cnn.add(LeakyReLU())
        # # cnn.add(Dropout(0.3))

        # cnn.add(ConvSN2D(256, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same',bias=True))
        # cnn.add(LeakyReLU())
        # # cnn.add(Dropout(0.3))

        # cnn.add(ConvSN2D(256, kernel_size=3, strides=2,kernel_initializer='glorot_uniform', padding='same',bias=True))
        # cnn.add(LeakyReLU())
        # # cnn.add(Dropout(0.3))

        # cnn.add(ConvSN2D(256, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same',bias=True))
        # cnn.add(LeakyReLU())
        # # cnn.add(Dropout(0.3))
        # # 96-----------
        # cnn.add(ConvSN2D(256, kernel_size=3, strides=2,kernel_initializer='glorot_uniform', padding='same',bias=True))
        # cnn.add(LeakyReLU())
        # # cnn.add(Dropout(0.3))

        # cnn.add(ConvSN2D(256, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same',bias=True))
        # cnn.add(LeakyReLU())
        # # cnn.add(Dropout(0.3))
        # # 128------------
        # build a relatively standard conv net, with LeakyReLUs as suggested in ACGAN
        cnn = Sequential()
        cnn.add(ConvSN2D(32, kernel_size=3, strides=2,kernel_initializer='glorot_uniform', padding='same'))
        cnn.add(LeakyReLU())
        # cnn.add(Dropout(0.3))
        cnn.add(ConvSN2D(64, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
        cnn.add(LeakyReLU())
        # cnn.add(Dropout(0.3))

        cnn.add(ConvSN2D(128, kernel_size=3, strides=2,kernel_initializer='glorot_uniform', padding='same'))
        cnn.add(LeakyReLU())
        # cnn.add(Dropout(0.3))

        cnn.add(ConvSN2D(256, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
        cnn.add(LeakyReLU())
        # cnn.add(Dropout(0.3))


        # deafault ----------- 32
        feature_resolution = resolution/4
        while feature_resolution > 8 :
          cnn.add(ConvSN2D(256, kernel_size=3, strides=2,kernel_initializer='glorot_uniform', padding='same'))
          cnn.add(LeakyReLU())
          cnn.add(ConvSN2D(256, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
          cnn.add(LeakyReLU())
          feature_resolution  = feature_resolution/2


        cnn.add(Flatten())

        features = cnn(image)
        return features


    nclasses=10
    batch_size=16
    dataset_name='MNIST'

    if(gan == "ganda") : #
        
        print("loading encoder...")
        
        
        reconstructor = build_reconstructor_ganda(latent_size)

        reconstructor.compile(
                optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                loss='mean_squared_error'
        )
            
        reconstructor.load_weights('encoder.h5')

        print("loading data...")
        bg_test = BatchGenerator(BatchGenerator.TEST, batch_size,
                             class_to_prune=None, unbalance=None, dataset=dataset_name)

        
        
        latent_full=[] #init latent space array
        label_full=[] #init label array

        print("get latent...")
        for c in range(nclasses):
            imgs = bg_test.dataset_x[bg_test.per_class_ids[c]]
            labels = bg_test.dataset_y[bg_test.per_class_ids[c]]###~~~cond
            bs=len(labels)
            labels_onehot = np.zeros([bs,nclasses])
            for i in range(bs):
                labels_onehot[i][int(labels[i])]=1.0
            latent = reconstructor.predict([imgs,labels_onehot]) 

            latent_full.extend(latent)
            label_full.extend(labels)
        
        latent_full=np.array(latent_full)
        label_full=np.array(label_full)
        
        

    elif(gan == "bagan") :#bagan

        print("loading encoder")
        reconstructor = build_reconstructor_bagan(latent_size)

        reconstructor.compile(
                optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                loss='mean_squared_error'
        )
        reconstructor.load_weights('encoder_bagan.h5') 

        print("loading data...")
        bg_test = BatchGenerator(BatchGenerator.TEST, batch_size,
                             class_to_prune=None, unbalance=None, dataset=dataset_name)

        latent_full=[] #init latent space array
        label_full=[] #init label array
        
        print("get latent...")
        for c in range(nclasses):
            imgs = bg_test.dataset_x[bg_test.per_class_ids[c]]
            labels = bg_test.dataset_y[bg_test.per_class_ids[c]]
            latent = reconstructor.predict(imgs)

            latent_full.extend(latent)
            label_full.extend(labels)

        latent_full=np.array(latent_full)
        label_full=np.array(label_full)

    


    print("dimension reduction...")
    time_start = time.time()
    tsne = TSNE(n_components=2).fit_transform(latent_full)
    #tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(latent_full)
    print( 't-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start ))
        
        
    latent_full=tsne

    #plot
    colors = ['red','yellow','orange','turquoise','green','powderblue','blue','purple','navy','indigo']
    scat=plt.scatter(latent_full[:,0],latent_full[:,1],s=2,c=label_full,cmap=matplotlib.colors.ListedColormap(colors))
    cb = plt.colorbar(scat, spacing='proportional')
    cb.set_label('classes')
    plt.title('tsne')
    plt.show()
        

    #k-clustering
    print("k-means clustering...")
    num_clusters = nclasses
    km = KMeans(n_clusters=num_clusters).fit(latent_full)
    y_kmeans = km.predict(latent_full)

    scat = plt.scatter(latent_full[:,0],latent_full[:,1], s=2, c=y_kmeans,cmap=matplotlib.colors.ListedColormap(colors))
    cb = plt.colorbar(scat, spacing='proportional')
    cb.set_label('classes')
    plt.title('Kmeans')
    plt.show()

    #v-score
    #print(v_measure_score(y_kmeans,label_full))
    print('v-score : '+ str(v_measure_score(y_kmeans,label_full)))

        
            

    





