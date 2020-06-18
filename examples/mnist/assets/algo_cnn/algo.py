import substratools as tools
import os
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

class Algo(tools.algo.Algo):

    def _normalize_X(self, X):

        if K.image_data_format() == 'channels_first':
            X = X.reshape(X.shape[0], 1, img_rows, img_cols)
       
            input_shape = (1, img_rows, img_cols)
        else:
            X = X.reshape(X.shape[0], img_rows, img_cols, 1)

            input_shape = (img_rows, img_cols, 1)

        X = X.astype('float32')

        X /= 255

        print('X shape:', X.shape)
        print(X.shape[0], ' samples')
        
        return X


    def train(self, X, y, models, rank):
        X = self._normalize_X(X)
        num_classes = 10
        y = keras.utils.to_categorical(y, num_classes)
        
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
        
        epochs = 1
        batch_size = 128
        # convert class vectors to binary class matrices 
        # TODO in metrics
        # y_test = keras.utils.to_categorical(y_test, num_classes)

        model.fit(X, y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                # validation_data=(x_test, y_test)
                )
        
        return model

    def predict(self, X, model):
    	X = self._normalize_X(X)
    	y_pred = model.predict_classes(X, verbose=0)
    	return y_pred
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            return keras.models.load_model(f)

    def save_model(self, model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            model.save(f)

if __name__ == '__main__':
    tools.algo.execute(Algo())
