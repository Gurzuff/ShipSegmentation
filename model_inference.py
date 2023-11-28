if __name__ == '__main__':
    import os
    import sys
    import numpy as np

    import keras.backend as K
    from keras import models, layers

    from PIL import Image
    import matplotlib.pyplot as plt

    # Custom score function
    def dice_coef(y_true, y_pred, eps=1e-6):
        y_true = K.cast(y_true, 'float32')
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
        dice_score = (2. * intersection + eps) / (union + eps)
        return K.mean(dice_score, axis=0)

    # Custom loss function
    def IoU_loss(y_true, y_pred, eps=1e-6):
        y_true = K.cast(y_true, 'float32')
        empty_image = K.equal(K.max(y_true), 0.0)
        if K.any(empty_image):
            y_true = 1 - y_true
            y_pred = 1 - y_pred
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
        return -K.mean((intersection + eps) / (union + eps), axis=0)

    # Visualization of several segmented images
    def segmentation_random_images(count_imgs=5, model_input=(384, 384), ROOT='test_images'):
        # Select n names from all test images
        NAME_imgs = os.listdir(ROOT)
        random_imgs = np.random.choice(NAME_imgs, count_imgs)
        # Set plot size
        plt.figure(figsize=(3*count_imgs, 1.5*count_imgs))
        for i in range(count_imgs):
            # first row for rgb images
            plt.subplot(2, count_imgs, i + 1)
            img_path = os.path.join(ROOT, random_imgs[i])
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f'{random_imgs[i]}')
            # second row for segmented masks
            plt.subplot(2, count_imgs, i + 1 + count_imgs)
            res_img = img.resize(model_input)
            res_img = np.expand_dims(res_img, 0) / 255.0
            mask_img = best_weight_model.predict(res_img)
            plt.imshow(mask_img[0, :, :, 0], vmin=0, vmax=1)
        plt.savefig(fr'segmented_images\segmented_{count_imgs}_images.png')
        plt.show()

    # Visualization of one loaded segmented image
    def segmentation_load_image(image_path='', model_input=(384, 384)):
        if image_path == '':
            pass
        else:
            img = Image.open(image_path)
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title('Image')
            plt.subplot(1, 2, 2)
            res_img = img.resize(model_input)
            res_img = np.expand_dims(res_img, 0) / 255.0
            mask_img = best_weight_model.predict(res_img)
            plt.imshow(mask_img[0, :, :, 0], vmin=0, vmax=1)
            plt.title('Segmented mask')
            plt.savefig(fr'segmented_images\segmented_loaded_image.png')
            plt.show()

# Load model with best weights
# ROOT_model = r'pretrained_models\trained_on_Kaggle'
ROOT_model = r'pretrained_models\trained_local'
NAME_model = '384_best_weight_model_2.h5'
best_weight_model = models.load_model(os.path.join(ROOT_model, NAME_model),
                                      custom_objects={'dice_coef': dice_coef,
                                                      'IoU_loss': IoU_loss}
                                      )
# Set input_shape for selected model
model_input = (int(NAME_model[:3]), int(NAME_model[:3]))

# For segmentation several images from directory 'test_images'
# Set the number of images (preferably from 2 to 5) and the size of model_input
segmentation_random_images(count_imgs=5, model_input=model_input, ROOT='test_images')

# For segmentation  a single loaded image from an external source
# Set the 'image_path' to the image and the size of model_input
segmentation_load_image(image_path='', model_input=model_input)
