import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetworkModel:
    def __init__(self, input_shape, optimizer, loss, activation, layers_vector, output_layer, batch_size = 5):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.loss = loss
        self.activation = activation
        self.layers_vector = layers_vector
        self.output_layer = output_layer
        self.model = self._build_model()
        self.batch_size = batch_size

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        
        for units in self.layers_vector:
            model.add(tf.keras.layers.Dense(units=units, activation=self.activation))
        
        model.add(tf.keras.layers.Dense(units=self.output_layer))
        return model

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def train_model(self, train_x, train_y, validation_split=0.2, epochs=100, verbose= 1 ,patience=5, checkpoint_path='model_checkpoint.keras'):
        self.compile_model()
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
        
        self.losses = self.model.fit(train_x, train_y, validation_split=validation_split,
                                     batch_size=self.batch_size, epochs=epochs,
                                     verbose = verbose,
                                     callbacks=[early_stopping, model_checkpoint])
        return self.losses

    def last_val_loss_value(self):
        return self.losses.history['val_loss'][-1]

    def plot_losses(self):
        loss_df = pd.DataFrame(self.losses.history)
        fig = loss_df.loc[:, ['loss', 'val_loss']].plot()
        plt.show()

    def predict(self, test_x,verbose = 1):
        results = self.model.predict(test_x,
                                     verbose = verbose)
        df = pd.DataFrame(results)
        df_mean = np.mean(df, axis=1)
        return df_mean

    def summary(self):
        return self.model.summary()
    
    def load_model(self, checkpoint_path='model_checkpoint.keras'):
        """
        Load the best model saved by the ModelCheckpoint callback.
        
        Args:
        checkpoint_path (str): Path to the saved model checkpoint file.
        
        Returns:
        Loaded model.
        """
        try:
            self.model = tf.keras.models.load_model(checkpoint_path)
            # print(f"Model loaded from {checkpoint_path}.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        return self.model
        

    def save_model(self,case, subset, args):
        """
        Save a Keras model to the specified file path.
        
        Args:
        model (tf.keras.Model): The Keras model to be saved.
        file_path (str): Path where the model will be saved. Default is 'saved_model.keras'.
        
        Returns:
        None
        """
        try:
            self.model.save(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\Regression\\Case_{case}\\GridSearch\\Neural Network\\{subset}_{args}.keras")
            # print(f"Model successfully saved to {file_path}.")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
        
class NeuralNetworkModel2Classes:
    def __init__(self, input_shape, optimizer, loss, activation, layers_vector, num_classes, batch_size: 5):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.loss = loss
        self.activation = activation
        self.layers_vector = layers_vector
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        
        for units in self.layers_vector:
            model.add(tf.keras.layers.Dense(units=units, activation=self.activation))
        
        model.add(tf.keras.layers.Dense(units=self.num_classes, activation="sigmoid"))
        return model

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def train_model(self, train_x, train_y, verbose = 0, validation_split=0.2, epochs=100, patience=5, checkpoint_path='model_checkpoint.keras'):
        self.compile_model()
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
        
        self.losses = self.model.fit(train_x, train_y, validation_split=validation_split,
                                     batch_size=self.batch_size, epochs=epochs,
                                     verbose= verbose,
                                     callbacks=[early_stopping, model_checkpoint])
        return self.losses

    def last_val_loss_value(self):
        return self.losses.history['val_loss'][-1]

    def plot_losses(self):
        loss_df = pd.DataFrame(self.losses.history)
        fig = loss_df.loc[:, ['loss', 'val_loss']].plot()
        plt.show()

    def predict(self, test_x):
        results = self.model.predict(test_x)
        return (results >= 0.5).astype(np.int32)

    def summary(self):
        return self.model.summary()

    def load_model(self, checkpoint_path='model_checkpoint.keras'):
        """
        Load the best model saved by the ModelCheckpoint callback.
        
        Args:
        checkpoint_path (str): Path to the saved model checkpoint file.
        
        Returns:
        Loaded model.
        """
        try:
            self.model = tf.keras.models.load_model(checkpoint_path)
            # print(f"Model loaded from {checkpoint_path}.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        return self.model
        

    def save_model(self,case, subset, args):
        """
        Save a Keras model to the specified file path.
        
        Args:
        model (tf.keras.Model): The Keras model to be saved.
        file_path (str): Path where the model will be saved. Default is 'saved_model.keras'.
        
        Returns:
        None
        """
        try:
            self.model.save(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\Classification 2 classes\\Case_{case}\\GridSearch\\Neural Network\\{subset}_{args}.keras")
            # print(f"Model successfully saved to {file_path}.")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
        
class NeuralNetworkModel3Classes:
    
# loss is categoricalcrossentropy


    def __init__(self, input_shape, optimizer, loss, activation, layers_vector, num_classes,batch_size=5):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.loss = loss
        self.activation = activation
        self.layers_vector = layers_vector
        self.num_classes = num_classes
        self.model = self._build_model()
        self.batch_size = batch_size

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        
        for units in self.layers_vector:
            model.add(tf.keras.layers.Dense(units=units, activation=self.activation))
        
        model.add(tf.keras.layers.Dense(units=self.num_classes, activation="softmax"))
        return model

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def train_model(self, train_x, train_y, validation_split=0.2,  epochs=100, patience=5,verbose=1, checkpoint_path='model_checkpoint.keras'):
        self.compile_model()
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
        
        self.losses = self.model.fit(train_x, train_y, validation_split=validation_split,
                                     batch_size=self.batch_size, epochs=epochs,
                                     verbose = verbose,
                                     callbacks=[early_stopping, model_checkpoint])
        return self.losses

    def last_val_loss_value(self):
        return self.losses.history['val_loss'][-1]

    def plot_losses(self):
        loss_df = pd.DataFrame(self.losses.history)
        fig = loss_df.loc[:, ['loss', 'val_loss']].plot()
        plt.show()

    def predict(self, test_x):
        results = self.model.predict(test_x)
        return np.argmax(results, axis=1)  # multiclass

    def summary(self):
        return self.model.summary()
    
    def load_model(self, checkpoint_path='model_checkpoint.keras'):
        """
        Load the best model saved by the ModelCheckpoint callback.
        
        Args:
        checkpoint_path (str): Path to the saved model checkpoint file.
        
        Returns:
        Loaded model.
        """
        try:
            self.model = tf.keras.models.load_model(checkpoint_path)
            # print(f"Model loaded from {checkpoint_path}.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        return self.model
        

    def save_model(self,case, subset, args):
        """
        Save a Keras model to the specified file path.
        
        Args:
        model (tf.keras.Model): The Keras model to be saved.
        file_path (str): Path where the model will be saved. Default is 'saved_model.keras'.
        
        Returns:
        None
        """
        try:
            self.model.save(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\Classification 3 classes\\Case_{case}\\GridSearch\\Neural Network\\{subset}_{args}.keras")
            # print(f"Model successfully saved to {file_path}.")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
        