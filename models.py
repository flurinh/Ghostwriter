import torch
import torch.nn as nn
import torch.nn.functional as F

# stupid bidirectional lstm
# TODO: implement as a class

class Gw00(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)

def window_loss():
    return 1
    
def action_loss(y_true, y_pred):
    loss = window_loss()
    return loss

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=True),
                            input_shape=input_shape))
    model.add(Bidirectional(LSTM(10)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))

    model.add(Activation('softmax'))

    model.compile(loss=my_loss, optimizer='rmsprop', metrics=['accuracy'])

    return model

def fit(data,
        batch_size=10,
        num_epochs=3,
        test_size=0.3):
    print('Start fitting...')
    random_state=42
    #labels = dict()
    print('Loading data.')
    X, y = load_data()
    
    num_classes = len(np.unique(y))
    num_trials = X[0].shape[0]

    y = np_utils.to_categorical(y, num_classes) #probably doesnt work!

    config = dict()
    config['labels'] = y
    config['num_classes'] = num_classes
    config['num_trials'] = num_trials
    #config['num_sess'] = num_sessions
    
    input_shape = (batch_size, X[0].shape[1], X[0].shape[2]) # 10 x frames_per_trial x features

    config_file_path = 'config_file'
    np.save(config_file_path, config)

    model = create_model(input_shape=input_shape, num_classes=num_classes)
    
    architecture_file_path = 'architecture'
    open(architecture_file_path, 'w').write(model.to_json())

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=test_size,
                                                    random_state=random_state)
    
    print('batchsize:',batch_size)
    train_gen = generate_batch(Xtrain, Ytrain)
    test_gen = generate_batch(Xtest, Ytest)

    train_num_batches = len(Xtrain) // batch_size
    test_num_batches = len(Xtest) // batch_size

    weigth_file_path = 'weights'
    checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=True)
    
    history = model.fit_generator(generator=train_gen, 
                                  steps_per_epoch=train_num_batches,
                                  epochs=num_epochs,
                                  verbose=1, 
                                  validation_data=test_gen, 
                                  validation_steps=test_num_batches,
                                  callbacks=[checkpoint])
    
    model.save_weights(weight_file_path)

    return history

def generate_batch(X, y, batch_size):
    # the problem with batch generation is the dependency on long enough time frames BEFORE the tap!
    num_batches = len(X) // batch_size
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = (batchIdx + 1) * batch_size
            yield np.array(X[start:end]), y[start:end]

# for an intuition compare https://github.com/junyanz/CycleGAN, Apple's paper and the handpose competition of machine perception

# typing video classification

# ROI model extension

# ROI model with hand pose detection

# ROI model/hand pose detection/input UI