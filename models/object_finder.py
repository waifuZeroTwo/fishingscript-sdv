import cv2
import math
import time
import logging
from pathlib import Path
import numpy as np
from tensorflow.keras import layers, models, optimizers


logger = logging.getLogger(__name__)

class object_finder:
    def __init__(self, load_model_path=None, save_model_path=None, new_data=False, train=False):
        self.base_dir = Path(__file__).resolve().parent
        self.numpy_data_dir = self.base_dir / 'numpy_data'
        self.image_data_dir = self.base_dir / 'image_data'

        self.model_save_path = self._resolve_model_path(save_model_path, for_save=True)
        if self.model_save_path is not None:
            if self.model_save_path.exists() and self.model_save_path.is_dir():
                raise IsADirectoryError(
                    f"Model save path points to a directory, expected a file path: {self.model_save_path}"
                )
            if not self.model_save_path.parent.exists():
                raise FileNotFoundError(
                    f"Model save directory does not exist: {self.model_save_path.parent}"
                )

        if train:
            if new_data:
                self.new_labels()
            self.init_fish_data()
            self.train_fish()
            
        self.model = None
        model_load_path = self._resolve_model_path(load_model_path)
        if model_load_path is not None and not model_load_path.is_file():
            raise FileNotFoundError(
                f"Could not find model file to load at: {model_load_path}"
            )
        if model_load_path is not None:
            self.model = models.load_model(model_load_path, compile=False)
        
        self.fish_block_size = 27
        self.bar_block_size = 158
        self.bar_max_top = 385
        self.bar_offset = 5
        self.last_bar_data = None


    def _resolve_model_path(self, model_path, for_save=False):
        if model_path is None:
            return None

        model_path = Path(model_path)
        if not model_path.is_absolute():
            # Support paths provided relative to either the repo root (e.g. models/foo.h5)
            # or the models directory itself (e.g. foo.h5).
            candidate_paths = [
                self.base_dir / model_path,
                self.base_dir.parent / model_path,
            ]

            if for_save:
                model_path = candidate_paths[0]
            else:
                model_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])

        if model_path.suffix in ('.keras', '.h5'):
            return model_path

        if for_save:
            return model_path.with_suffix('.keras')

        keras_path = model_path.with_suffix('.keras')
        if keras_path.is_file():
            return keras_path

        h5_path = model_path.with_suffix('.h5')
        if h5_path.is_file():
            return h5_path

        return model_path
        
    def new_labels(self):
        fish_data_path = self.image_data_dir / 'fish_labels.txt'
        fish_labels_path = self.numpy_data_dir / 'fish_labels.npy'
        with fish_data_path.open('r') as labels_file:
            label_data = [list(map(int, line.rstrip('\n').split(',')[1:])) for line in labels_file]
        label_data = np.array(label_data)
        np.save(fish_labels_path, label_data)

        
    def init_fish_data(self):
        train_image_path = self.numpy_data_dir / 'train_imgs.npy'
        fish_labels_path = self.numpy_data_dir / 'fish_labels.npy'
        train_data = np.load(train_image_path)
        labels_data = np.load(fish_labels_path)
        predictions = np.array([cv2.imread(str(self.image_data_dir / '171.jpg'))])
        rows = train_data.shape[1]
        
        return train_data[:labels_data.shape[0]], labels_data, predictions, rows


    def train_fish(self):
        flag = 3
        X, y, predictions, rows = self.init_fish_data()
        X, y, predictions = self.reshape_data(X, y, predictions, rows, flag=flag, block_size=27)
        
        new_model = models.Sequential()
        sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
        new_model.add(layers.Dense(600, input_shape=(X.shape[1],), activation='sigmoid'))
        new_model.add(layers.Dense(1, activation='sigmoid'))
        new_model.compile(
            optimizer=sgd,
            loss='mean_squared_error',
            metrics=['accuracy']
        )
        new_model.fit(X, y, epochs=1, batch_size=1, verbose=0)
        if self.model_save_path is not None:
            new_model.save(self.model_save_path)

        
    def locate_fish(self, screen):
        screen = np.mean(screen, axis=2)
        new_x = list()
        for row in range(screen.shape[0] - self.fish_block_size):
            temp = screen[row:row+self.fish_block_size]
            new_x.append(temp.reshape(temp.shape[0] * temp.shape[1]))

        screen = np.array(new_x)
        num_rows = screen.shape[0]

        model_input_shape = getattr(self.model, 'input_shape', None)
        if isinstance(model_input_shape, list):
            model_input_shape = model_input_shape[0] if model_input_shape else None
        if model_input_shape is None and getattr(self.model, 'inputs', None):
            model_input_shape = self.model.inputs[0].shape

        input_rank = len(model_input_shape) if model_input_shape is not None else screen.ndim
        if input_rank == 2:
            model_input = screen
        elif input_rank == 3:
            model_input = screen.reshape((1, num_rows, screen.shape[1]))
        else:
            raise ValueError(
                f"Unsupported model input rank {input_rank} for fish locator. "
                f"Detected model input shape: {model_input_shape}"
            )

        logger.debug(
            'locate_fish model input shape=%s adapted_input_shape=%s num_rows=%s',
            model_input_shape,
            model_input.shape,
            num_rows,
        )
        fish_row = self.model.predict(model_input)
        logger.debug('locate_fish raw prediction shape=%s', np.shape(fish_row))

        fish_scores = np.squeeze(fish_row)
        logger.debug('locate_fish squeezed prediction shape=%s', np.shape(fish_scores))

        if np.ndim(fish_scores) == 0:
            raise ValueError(
                f"Model prediction collapsed to scalar after squeeze; raw prediction shape was {np.shape(fish_row)}"
            )

        if np.ndim(fish_scores) == 1:
            row_scores = fish_scores
        else:
            matching_axes = [axis for axis, size in enumerate(fish_scores.shape) if size == num_rows]
            if matching_axes:
                row_axis = matching_axes[0]
                row_scores = np.moveaxis(fish_scores, row_axis, 0).reshape(num_rows, -1).mean(axis=1)
            else:
                row_scores = fish_scores.reshape(-1)

        logger.debug('locate_fish normalized score vector shape=%s', np.shape(row_scores))
        return int(np.argmax(row_scores))


    def locate_bar(self, screen):
        edges = cv2.Canny(screen, 100, 200)       
        edges  = edges / 255
        col_start = 10
        col_end = 30
        row_start = 12
        mask = np.ones(col_end-col_start)
        row = np.where((edges[row_start:,col_start:col_end]==mask).all(axis=1))
        if row[0].size > 0:
            row = row[0][0]
        else:
            if self.last_bar_data is not None:
                return self.last_bar_data
            return (0, 0)
        
        if row > self.bar_max_top:
            if self.last_bar_data is not None and abs(self.last_bar_data[0] - (row-self.bar_block_size+self.bar_offset)) > 130:
                return self.last_bar_data

            self.last_bar_data = (row-self.bar_block_size+self.bar_offset), row+self.bar_offset
            return (row-self.bar_block_size+self.bar_offset), row+self.bar_offset
        
        #Found bar top
        if row != 0:
            if self.last_bar_data is not None and abs(self.last_bar_data[0] - (row+self.bar_offset)) > 130:
                return self.last_bar_data
            
            self.last_bar_data = (row+self.bar_offset, row+self.bar_block_size+self.bar_offset)
            return row+self.bar_offset, row+self.bar_block_size+self.bar_offset

            if self.last_bar_data is not None:
                return self.last_bar_data
        
        return self.last_bar_data

    
    def reshape_data(self, X, y, predictions, rows, flag=0, block_size=0):
        """
        Flag:
            0 (default) : shape as just matching 1 row (1 output layer)
            1 : Match every row of fish (27 output layers)
            2 : Reshape input to be n inputs of 27, 1 output to see if yes or no fish
            3 : do same as 2, but average the RGB values for each pixel
        """

        if flag == 0:
            X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            predictions = predictions.reshape((predictions.shape[0], predictions.shape[1] * predictions.shape[2] * predictions.shape[3]))
            #Reshape from from just a # to all 0's
            reshaped_target = np.zeros((y.shape[0], rows))

            #Change index of correct predictions to a 1
            reshaped_target[np.arange(reshaped_target.shape[0]), y]=1

            return X, reshaped_target, predictions

        
        elif flag == 1:
            X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))

            predictions = predictions.reshape((predictions.shape[0], predictions.shape[1] * predictions.shape[2] * predictions.shape[3]))
        
            #Reshape from from just a # to all 0's
            reshaped_target = np.zeros((y.shape[0], rows))

            #Change index of correct predictions to a 1
            for row in range(reshaped_target.shape[0]):
                reshaped_target[row,y[row][0]:y[row][1]]=1

            #pdb.set_trace()
            return X, reshaped_target, predictions

        
        elif flag == 2:
            in_range = block_size

            #Reshape inputs
            new_x = list()
            for inp in X:
                for row in range(inp.shape[0] - block_size):            
                    temp = inp[row:row+in_range]
                    new_x.append(temp.reshape(temp.shape[0] * temp.shape[1] * temp.shape[2]))
                          
            new_x = np.array(new_x)
            X = new_x

            #Reshape target
            reshaped_target = np.zeros((y[:,0].shape[0], rows - block_size))

            #Change index of correct predictions to a 1
            reshaped_target[np.arange(reshaped_target.shape[0]), y[:,0]]=1
            y = reshaped_target
            y = y.reshape(y.shape[0] * y.shape[1])

            #Reshape predictions
            new_preds = list()
            for inp in predictions:
                for row in range(inp.shape[0] - block_size):            
                    temp = inp[row:row+in_range]
                    new_preds.append(temp.reshape(temp.shape[0] * temp.shape[1] * temp.shape[2]))
            new_preds = np.array(new_preds)
            predictions = new_preds
            return X, y, predictions

        elif flag == 3:
            in_range = block_size
            X = np.mean(X, axis=3)
            #Reshape inputs
            new_x = list()
            for inp in X:
                for row in range(inp.shape[0] - block_size):            
                    temp = inp[row:row+in_range]
                    new_x.append(temp.reshape(temp.shape[0] * temp.shape[1]))
                          
            new_x = np.array(new_x)
            
            X = new_x

            #Reshape target
            reshaped_target = np.zeros((y[:,0].shape[0], rows - block_size))

            #Change index of correct predictions to a 1
            reshaped_target[np.arange(reshaped_target.shape[0]), y[:,0]]=1
            y = reshaped_target
            y = y.reshape(y.shape[0] * y.shape[1])

            #Reshape predictions
            predictions = np.mean(predictions, axis=3)
            new_preds = list()
            for inp in predictions:
                for row in range(inp.shape[0] - block_size):            
                    temp = inp[row:row+in_range]
                    new_preds.append(temp.reshape(temp.shape[0] * temp.shape[1]))
            new_preds = np.array(new_preds)
            predictions = new_preds
            return X, y, predictions



if __name__ == '__main__':
    ol = object_finder(load_model_path=Path('batch100_fish_id.h5'))
    train_data, _, _, _ = ol.init_fish_data()
    #cv2.imwrite('image_data/edges61.jpg', cv2.Canny(train_data[60], 100, 200))
    #cv2.imwrite('image_data/edges62.jpg', cv2.Canny(train_data[61], 100, 200))
    #cv2.imwrite('image_data/edges63.jpg', cv2.Canny(train_data[62], 100, 200))
    bottom = 0
    top = 134
    print(ol.locate_fish(train_data[bottom]))
    print(ol.locate_bar(train_data[top]))
    #for i in range(29, 40):
    #print(str(i+1) + ": " + str(ol.locate_bar(train_data[i])))
    #pdb.set_trace()
    #for i in range(0, 10):
    #    print(locate_bar(train_data[i]))
    #train_fish()
    #model = load_model('batch100_fish_id.h5', compile=False)
    #print(locate_fish(cv2.imread('data/1.jpg')))
    #train_bar()
