if __name__=='__main__':
    import os
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    from keras import models, layers
    from keras.optimizers import Adam, RMSprop
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
    import keras.backend as K
    from sklearn.model_selection import train_test_split

    SEED = 33
    NET_SCALING = (1, 1)
    GAUSSIAN_NOISE = 0.1
    warnings.filterwarnings("ignore")

    # Parameters of image augmentation
    aug_params = dict(featurewise_center=False,
                      samplewise_center=False,
                      rotation_range=30,
                      horizontal_flip=True,
                      vertical_flip=True,
                      width_shift_range=0.1,
                      height_shift_range=0.1,
                      shear_range=0.01,
                      zoom_range=[0.9, 1.1],
                      fill_mode='reflect',
                      data_format='channels_last')

    # ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    def rle_decode(mask_rle, shape=(768, 768)):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T  # Needed to align to RLE direction

    # https://www.kaggle.com/code/kmader/baseline-u-net-model-part-1
    def masks_as_image(in_mask_list):
        # Take the individual ship masks and create a single mask array for all ships
        all_masks = np.zeros((768, 768), dtype=np.int16)
        for mask in in_mask_list:
            if isinstance(mask, str):
                all_masks += rle_decode(mask)
        return np.expand_dims(all_masks, -1)

    # https://www.kaggle.com/code/kmader/baseline-u-net-model-part-1
    def make_image_gen(in_df, batch_size=64):
        # Collects normalized RGB photos and concatenated masks into a batch
        all_batches = list(in_df.groupby('ImageId'))
        out_rgb = []
        out_mask = []
        while True:
            np.random.shuffle(all_batches)
            for c_img_id, c_masks in all_batches:
                rgb_path = os.path.join(TRAIN_PATH, c_img_id)
                c_img = imread(rgb_path)
                c_mask = masks_as_image(c_masks['EncodedPixels'].values)
                if IMG_SCALING is not None:
                    c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                    c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
                out_rgb += [c_img]
                out_mask += [c_mask]
                if len(out_rgb) >= batch_size:
                    yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                    out_rgb, out_mask = [], []

    # Plot of Dice scores and loss values for training and testing model
    def plot_accuracy(history, train_iter=1, metric='dice_coef', loss_func='IoU_loss'):
        # Building plot for the history of model training
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history[f'{metric}'])
        plt.plot(history.history[f'val_{metric}'])
        plt.title('Model Dice score')
        plt.ylabel('Score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(np.log(1 + np.array(history.history['loss'])))
        plt.plot(np.log(1 + np.array(history.history['val_loss'])))
        plt.title(loss_func)
        plt.ylabel('Score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.savefig(f'plot_training_metrics/plot-{train_iter}.png')

    # https://www.kaggle.com/code/kmader/baseline-u-net-model-part-1
    def create_aug_gen(in_gen, seed=None):
        np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
        # Creat ImageDataGenerators
        X_train_datagen = ImageDataGenerator(**aug_params)
        y_valid_datagen = ImageDataGenerator(**aug_params)
        for in_x, in_y in in_gen:
            seed = np.random.choice(range(9999))
            # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
            g_x = X_train_datagen.flow(255 * in_x,
                                       batch_size=in_x.shape[0],  # batch
                                       seed=seed,
                                       shuffle=True)
            g_y = y_valid_datagen.flow(in_y,
                                       batch_size=in_x.shape[0],  # batch
                                       seed=seed,
                                       shuffle=True)

            yield next(g_x) / 255.0, next(g_y)

    def upsample_simple(filters, kernel_size, strides, padding):
        return layers.UpSampling2D(strides)

    # Score functions
    def dice_coef(y_true, y_pred, eps=1e-6):
        y_true = K.cast(y_true, 'float32')
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
        dice_score = (2. * intersection + eps) / (union + eps)
        return K.mean(dice_score, axis=0)

    # Loss functions
    def IoU_loss(y_true, y_pred, eps=1e-6):
        y_true = K.cast(y_true, 'float32')
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
        # loss_score = - K.mean((intersection + eps) / (union + eps), axis=0)
        # loss_score = - K.mean((intersection + eps) / (union + K.sum(y_true, axis=[1, 2, 3]) + eps), axis=0)  - best score: 0.61
        loss_score = - K.mean((intersection + eps) / (union - 0.9 * K.sum(y_true, axis=[1, 2, 3]) + eps), axis=0)   # 0.605 with k=0.5
        return loss_score

    def build_unet_model(input_shape, upsample):
        NET_SCALING = (1, 1)
        GAUSSIAN_NOISE = 0.1

        input_img = layers.Input(input_shape, name='RGB_Input')
        pp_in_layer = input_img
        pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)
        pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
        pp_in_layer = layers.BatchNormalization()(pp_in_layer)

        c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(pp_in_layer)
        c1 = layers.BatchNormalization()(c1)
        c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.BatchNormalization()(c2)
        c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.BatchNormalization()(c3)
        c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.BatchNormalization()(c4)
        c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.BatchNormalization()(c5)
        c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

        u6 = upsample(64, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.BatchNormalization()(c6)
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

        u7 = upsample(32, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.BatchNormalization()(c7)
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

        u8 = upsample(16, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.BatchNormalization()(c8)
        c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

        u9 = upsample(8, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.BatchNormalization()(c9)
        c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

        d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        d = layers.UpSampling2D(NET_SCALING)(d)
        seg_model = models.Model(inputs=[input_img], outputs=[d])
        return seg_model

    # CALBACKS
    # Seve best weights
    weight_path = "pretrained_models/best_weights.hdf5"
    checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', mode='max',
                                 verbose=1, save_best_only=True, save_weights_only=True)
    # Reduce LR
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
                                       patience=3, verbose=1, mode='max',
                                       min_delta=0.0001, cooldown=0, min_lr=1e-8)

    early = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=12)
    callbacks_list = [checkpoint, reduceLROnPlat, early]

# Paths to the satellite images
ROOT_images = 'E:\DataSets\Ship Detection (Airbus, Kaggle)'
TEST_PATH = os.path.join(ROOT_images, "test_v2")
TRAIN_PATH = os.path.join(ROOT_images, "train_v2")

# Loading DataFrames
df_train = pd.read_csv('balanced_dfs/train_ship_segmentations_v2.csv')
balanced_train_df_1 = pd.read_csv('balanced_dfs/balanced_train_df_1.csv')
balanced_train_df_2 = pd.read_csv('balanced_dfs/balanced_train_df_2.csv')


# 1. Preparation for FIRST training iteration
LR = 1e-3
BATCH_SIZE = 30
IMG_SCALING = (2, 2)  # resizing image from 768 to 384 pixels
INPUT_SHAPE = (384, 384, 3)   # (384, 384, 768, 768)
# VALID_IMG_COUNT = 1000  # number of validation images to use

# Train-validation split for FIRST training iteration
train_ids_1, valid_ids_1 = train_test_split(balanced_train_df_1,
                                            test_size=0.20,
                                            shuffle=True,
                                            random_state=SEED,
                                            stratify=balanced_train_df_1['ships'])

print('Build image generator for FIRST training iteration...')
# Train generator-1
train_df_1 = pd.merge(df_train, train_ids_1, 'right', on='ImageId')
train_gen_1 = make_image_gen(train_df_1, BATCH_SIZE)

# Validation generator-1
valid_df_1 = pd.merge(df_train, valid_ids_1, 'right', on='ImageId')
# valid_x, valid_y = next(make_image_gen(valid_df_1, VALID_IMG_COUNT))
valid_gen_1 = make_image_gen(valid_df_1, BATCH_SIZE)

# Build U-bet model
seg_model = build_unet_model(INPUT_SHAPE, upsample_simple)

# Compile model
print(f'Compile model with DF: 100% images with ships')
seg_model.compile(optimizer=RMSprop(lr=LR),    #Adam(LR, decay=1e-6),
                  loss=IoU_loss,
                  metrics=[dice_coef, 'binary_accuracy'])

# FIRST iteration of model training
NB_EPOCHS = 30
MAX_TRAIN_STEPS = 35
print('Started FIRST training iteration')
history_1 = seg_model.fit(create_aug_gen(make_image_gen(train_df_1, BATCH_SIZE)),
                          steps_per_epoch=MAX_TRAIN_STEPS,
                          epochs=NB_EPOCHS,
                          validation_data=valid_gen_1,   # (valid_x, valid_y),
                          validation_steps=MAX_TRAIN_STEPS,
                          callbacks=callbacks_list,
                          verbose=1)
# Save plot
plot_accuracy(history_1, train_iter=1, metric='dice_coef', loss_func='IoU_loss')

# Save model with best weights
seg_model.load_weights(weight_path)
seg_model.save(f'pretrained_models/trained_local/{INPUT_SHAPE[0]}_best_weight_model_1.h5')

# 2. Preparation for the SECOND training iteration
LR = 1e-3
BATCH_SIZE = 30
IMG_SCALING = (2, 2)  # resizing image from 768 to 384 pixels
INPUT_SHAPE = (384, 384, 3)   # 384, 384, 768, 768,
# VALID_IMG_COUNT = 1000  # number of validation images to use

# Train-validation split for the SECOND training iteration
train_ids_2, valid_ids_2 = train_test_split(balanced_train_df_2,
                                            test_size=0.20,
                                            shuffle=True,
                                            random_state=SEED,
                                            stratify=balanced_train_df_2['ships'])

print('Build image generator for the SECOND training iteration...')
# Train generator-1
train_df_2 = pd.merge(df_train, train_ids_2, 'right', on='ImageId')
train_gen_2 = make_image_gen(train_df_2, BATCH_SIZE)

# Validation generator-1
valid_df_2 = pd.merge(df_train, valid_ids_2, 'right', on='ImageId')
# valid_x_2, valid_y_2 = next(make_image_gen(valid_df_2, VALID_IMG_COUNT))
valid_gen_2 = make_image_gen(valid_df_2, BATCH_SIZE)

print(f'Compile model with DF: 50% with ships, 50% without')
seg_model.compile(optimizer=RMSprop(lr=LR),    #Adam(LR, decay=1e-6),
                  loss=IoU_loss,
                  metrics=[dice_coef, 'binary_accuracy'])

# SECOND iteration of model training
NB_EPOCHS = 30
MAX_TRAIN_STEPS = 35
print('Started SECOND training iteration')
history_2 = seg_model.fit(create_aug_gen(make_image_gen(train_df_2, BATCH_SIZE)),
                          steps_per_epoch=MAX_TRAIN_STEPS,
                          epochs=NB_EPOCHS,
                          validation_data=valid_gen_2,   # (valid_x_2, valid_y_2),
                          validation_steps=MAX_TRAIN_STEPS,
                          callbacks=callbacks_list,
                          verbose=1
                          )

# Save model with best weights
seg_model.load_weights(weight_path)
seg_model.save(f'pretrained_models/trained_local/{INPUT_SHAPE[0]}_best_weight_model_2.h5')
# Save plot
plot_accuracy(history_2, train_iter=2, metric='dice_coef', loss_func='IoU_loss')
