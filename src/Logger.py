import pathlib
import matplotlib.pyplot as plt
import src.plot as _plt
import numpy as np
from icecream import ic
class Logger():
    def createLogFolders(self, dataset):
        figures_path = 'figures' + dataset.__class__.__name__ + '/'
        pathlib.Path(figures_path).mkdir(parents=True, exist_ok=True)
        self.title_name = 'ResUnet'
    def plotFigure(self, figure, name='figure', cmap = plt.cm.gray, savefig=False):
        plt.figure(figsize=(15,15))
        plt.imshow(figure, cmap=plt.cm.gray)

        # title_name = 'ResUnet'
        plt.axis('off')
        if savefig == True:
            plt.savefig('figures/' + name, dpi=150, bbox_inches='tight')
    
    def snipDataset(self, idx, coords_train, patchesHandler, image_stack, label_mask):
        print(coords_train[idx])
        image_patch, reference_patch = patchesHandler.getPatch(
            image_stack, label_mask, coords_train, idx = idx)
        ic(np.mean(image_patch[...,[1,2,3]]), np.mean(image_patch[...,[11,12,13]]))
        _plt.plotCropSample4(image_patch[...,[1,2,3]], image_patch[...,[11,12,13]], 
                reference_patch, reference_patch,
                lims = None, 
                titles = ['Optical T0', 'Optical T1', 'Reference', 'Reference 2'],
                cmaps = [plt.cm.gray, plt.cm.gray, plt.cm.gray, plt.cm.gray],
                maskBackground = [False, False, False, False],
                invertMask = [False, False, False, False])        

    def plotHistory(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('loss_history.png')


    def plotLossTerms(self, history):
        l = ['loss', 'val_loss', 'KL_term', 'val_KL_term', 'loglikelihood_term', 'val_loglikelihood_term']


        plt.figure(2)
        plt.plot(history.history[l[0]])
        plt.plot(history.history[l[1]])
        plt.plot(history.history[l[2]])
        plt.plot(history.history[l[3]])
        plt.plot(history.history[l[4]])
        plt.plot(history.history[l[5]])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(l, loc='upper left')
        plt.savefig('loss_history.png')

        plt.figure(3)
        plt.plot(history.history[l[2]])
        plt.plot(history.history[l[3]])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(l[2:4], loc='upper left')
        plt.savefig('loss_history.png')

        plt.figure(4)
        plt.plot(history.history[l[4]])
        plt.plot(history.history[l[5]])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(l[4:6], loc='upper left')
        plt.savefig('loss_history.png')

    def plotAnnealingCoef(self, history):
        plt.figure(5)
        plt.plot(history.history['annealing_coef'],c='r',marker='+')
        plt.title('Annealing coef')
        plt.xlabel('Epoch')
        plt.ylabel('Annealing coef') 

        plt.figure(6)
        plt.plot(history.history['global_step_get'],c='r',marker='+')
        plt.title('Global coef')
        plt.xlabel('Epoch')
        plt.ylabel('Global coef') 

        plt.figure(7)
        plt.plot(history.history['annealing_step_get'],c='r',marker='+')
        plt.title('Annealing step')
        plt.xlabel('Epoch')
        plt.ylabel('Annealing step') 