import numpy as np

def infer(new_model, image1_pad,
        h, w, num_patches_x, num_patches_y, 
        patch_size_x, patch_size_y):
    img_reconstructed = np.zeros((h, w), dtype=np.float32)
    for i in range(0,num_patches_y):
        for j in range(0,num_patches_x):
#             img_reconstructed[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1)]=patches_pred[count]
            patch = image1_pad[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1)]
            predicted = new_model.predict(np.expand_dims(patch, axis=0))[:,:,:,1].astype(np.float32)
            # del patch
            img_reconstructed[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1)] = predicted
            # del predicted
    del patch, predicted
    return img_reconstructed
