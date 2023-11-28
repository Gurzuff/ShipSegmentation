if __name__=='__main__':
    import os
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

# Paths to ship masks for training data
ROOT_dfs = 'balanced_dfs'
SHIP_MASK_DF_PATH = os.path.join(ROOT_dfs, "train_ship_segmentations_v2.csv")

# Max number of samples in ships group
SAMPLES_PER_GROUP_1 = 4000   # for FIRST train iteration (100% img with mask)
SAMPLES_PER_GROUP_2 = 30000  # for SECOND train iteration (50% with, 50% without)

# Loading ship masks DataFrame
df_train = pd.read_csv(SHIP_MASK_DF_PATH)

# 1. CREATE DF for 1st train iteration: 100% images with masks
# Temporary DF
df_only_ships = df_train[df_train.EncodedPixels.notnull()]
df_only_ships['ships'] = 1

# Create DF with unique images and quantity of ship masks
df_unique_im = df_only_ships.groupby(['ImageId'])['ships'].sum().reset_index()
# Remove temporary DF
del df_only_ships

# Add new features (in float and list format)
df_unique_im['has_ship'] = df_unique_im['ships'].astype('float')
df_unique_im['has_ship_vec'] = df_unique_im['has_ship'].map(lambda x: [x])

# Balance the groups by the number of masks in the image
balanced_train_df_1 = df_unique_im.groupby('ships')\
    .apply(lambda x: x.sample(SAMPLES_PER_GROUP_1) if len(x) > SAMPLES_PER_GROUP_1 else x)
# Save to file.csv
balanced_train_df_1.to_csv('balanced_dfs/balanced_train_df_1.csv', sep=',', header=True, index=None)


# 2. CREATE DF for 2bd train iteration: 50% images WITH masks + 50 images WITHOUT masks
# Empty flag for filters
df_train['flag'] = 0
# Number of unique images with ship
n_samples = df_train.ImageId[df_train.EncodedPixels.notnull()].nunique()
# Add flag=1 for images with ship
df_train['flag'][df_train.EncodedPixels.notnull()] = 1
# List indexes of random 42556 images among 150.000 images without ships
index_samples = df_train[df_train.EncodedPixels.isnull()].sample(n_samples).index
# Add flag=1 for 42556 random images without ships
df_train['flag'][df_train.index.isin(index_samples)] = 1
# Short DF with 50% unique images with ship and 50% images without ship
df_short_train = df_train[['ImageId', 'EncodedPixels']][df_train.flag == 1]
# Add feature: 1 - mask present, 0 - mask absent
df_short_train['ships'] = 1
df_short_train['ships'][df_train.EncodedPixels.isnull()] = 0
# Create DF with unique images and quantity of ship masks
df_unique_im_2 = df_short_train.groupby(['ImageId'])['ships'].sum().reset_index()
# Remove feature "ships"
del df_short_train
# Add new features (in float and list format)
df_unique_im_2['has_ship'] = df_unique_im_2['ships'].astype('float')
df_unique_im_2['has_ship'][df_unique_im_2['ships'] > 0] = 1.0
df_unique_im_2['has_ship_vec'] = df_unique_im_2['has_ship'].map(lambda x: [x])
# Balance the groups by the number of masks in the image
balanced_train_df_2 = df_unique_im_2.groupby('ships')\
    .apply(lambda x: x.sample(SAMPLES_PER_GROUP_2) if len(x) > SAMPLES_PER_GROUP_2 else x)
# Save to file.csv
balanced_train_df_2.to_csv('balanced_dfs/balanced_train_df_2.csv', sep=',', header=True, index=None)

print('balanced_train_df_1.shape[0] =', balanced_train_df_1.shape[0])
print(f'unique img WITH mask: {balanced_train_df_1[balanced_train_df_1.ships!=0].shape[0]}, \
        unique img WITHOUT mask: {balanced_train_df_1[balanced_train_df_1.ships==0].shape[0]}')
print('-'*10)
print('balanced_train_df_2.shape[0] =', balanced_train_df_2.shape[0])
print(f'unique img WITH mask: {balanced_train_df_2[balanced_train_df_2.ships!=0].shape[0]}, \
        unique img WITHOUT mask: {balanced_train_df_2[balanced_train_df_2.ships==0].shape[0]}')
