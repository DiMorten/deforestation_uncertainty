import pathlib
import matplotlib.pyplot as plt
import src.plot as _plt
import numpy as np
from icecream import ic
class Logger():
    def createLogFolders(self, dataset):
        figures_path = 'output/figures' + dataset.__class__.__name__ + '/'
        pathlib.Path(figures_path).mkdir(parents=True, exist_ok=True)
        self.title_name = 'ResUnet'
    def plotFigure(self, figure, name='output/figure', cmap = plt.cm.gray, savefig=False, figsize=(15,15), dpi=200):
        plt.figure(figsize=figsize)
        plt.imshow(figure, cmap=cmap)

        # title_name = 'ResUnet'
        plt.axis('off')
        if savefig == True:
            plt.savefig('output/figures/' + name, dpi=150, bbox_inches='tight')

    def plotFigure2(self, figure, name='output/figure', cmap = plt.cm.gray, savefig=False, figsize=(15,15), dpi=300):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(figure)

        fig.savefig('output/figures/Para' + name, dpi=dpi, bbox_inches='tight')

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

    def getStats(self, value):
        ic(np.min(value), np.mean(value), np.max(value))



    def plotCropSample(self, trainer):
        self.plotCropSampleFlag = True
        if self.plotCropSampleFlag == True:
                # import matplotlib
                # customCmap = matplotlib.colors.ListedColormap(['black', 'red'])
                ic(trainer.dataset.previewLims1, trainer.dataset.previewLims2)
                lims = trainer.dataset.previewLims1
                ic(np.unique(trainer.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]]))
                lims = trainer.dataset.previewLims2
                ic(np.unique(trainer.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]], return_counts=True))

                _plt.plotCropSample4(trainer.image_stack[...,trainer.dataset.previewBands], trainer.mean_prob, 
                        trainer.error_mask_to_show_rgb[...,::-1], trainer.uncertainty_to_show, 
                        lims = trainer.dataset.previewLims1, 
                        titles = ['Optical', 'Predict Probability', 'Predicted', 'Uncertainty'],
                        cmaps = [plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                        maskBackground = [False, True, False, True],
                        invertMask = [False, False, False, False])
                plt.savefig('output/figures/' + trainer.dataset.__class__.__name__ + 'PredictSampleUncertainty1.png', dpi=150, bbox_inches='tight')

                _plt.plotCropSample4(trainer.image_stack[...,trainer.dataset.previewBands], trainer.mean_prob, 
                        trainer.error_mask_to_show_rgb[...,::-1], trainer.uncertainty_to_show, 
                        lims = trainer.dataset.previewLims2, 
                        titles = ['Optical', 'Predict Probability', 'Predicted', 'Uncertainty'],
                        cmaps = [plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                        maskBackground = [False, True, False, True],
                        invertMask = [False, False, False, False])
                plt.savefig('output/figures/' + trainer.dataset.__class__.__name__ + 'PredictSampleUncertainty2.png', dpi=150, bbox_inches='tight')

    def plotCropSample(self, trainer):

        uncertainty_vlims = [np.min(trainer.uncertainty_to_show), np.max(trainer.uncertainty_to_show)]

        self.plotCropSampleFlag = True
        if self.plotCropSampleFlag == True:
            ic(trainer.dataset.previewLims1, trainer.dataset.previewLims2)
            lims = trainer.dataset.previewLims1
            ic(np.unique(trainer.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]]))
            lims = trainer.dataset.previewLims2
            ic(np.unique(trainer.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]], return_counts=True))

            _plt.plotCropSample4(trainer.image_stack[...,trainer.dataset.previewBands], trainer.mean_prob, 
                    trainer.error_mask_to_show_rgb[...,::-1], trainer.uncertainty_to_show, 
                    lims = trainer.dataset.previewLims1, 
                    titles = ['Snippet', 'Predicted Probability', 'Predicted', 'Uncertainty'],
                    cmaps = [plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                    maskBackground = [False, True, False, True],
                    invertMask = [False, False, False, False], uncertainty_vlims = uncertainty_vlims)
            plt.savefig('output/figures/' + trainer.dataset.__class__.__name__ + 'PredictSampleUncertainty1.png', dpi=150, bbox_inches='tight')

            _plt.plotCropSample4(trainer.image_stack[...,trainer.dataset.previewBands], trainer.mean_prob, 
                    trainer.error_mask_to_show_rgb[...,::-1], trainer.uncertainty_to_show, 
                    lims = trainer.dataset.previewLims2, 
                    titles = ['Snippet', 'Predicted Probability', 'Predicted', 'Uncertainty'],
                    cmaps = [plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                    maskBackground = [False, True, False, True],
                    invertMask = [False, False, False, False], uncertainty_vlims = uncertainty_vlims)
            plt.savefig('output/figures/' + trainer.dataset.__class__.__name__ + 'PredictSampleUncertainty2.png', dpi=150, bbox_inches='tight')

            _plt.plotCropSample4(trainer.image_stack[...,trainer.dataset.previewBands], trainer.mean_prob, 
                    trainer.error_mask_to_show_rgb[...,::-1], trainer.uncertainty_to_show, 
                    lims = trainer.dataset.previewLims2, 
                    titles = ['Snippet', 'Predicted Probability', 'Predicted', 'Uncertainty'],
                    cmaps = [plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                    maskBackground = [False, True, False, True],
                    invertMask = [False, False, False, False], uncertainty_vlims = uncertainty_vlims,
                    colorbar = True)
            plt.savefig('output/figures/' + trainer.dataset.__class__.__name__ + 'PredictSampleUncertaintyColorbar.png', dpi=150, bbox_inches='tight')

    def plotCropSampleT0T1(self, trainer):
        previewBandsT0 = [3,2,1]
        previewBandsT1 = [13,12,11]
        uncertainty_vlims = [np.min(trainer.uncertainty_to_show), np.max(trainer.uncertainty_to_show)]

        self.plotCropSampleFlag = True
        if self.plotCropSampleFlag == True:

            # import matplotlib
            # customCmap = matplotlib.colors.ListedColormap(['black', 'red'])
            '''
            ic(trainer.dataset.previewLims1, trainer.dataset.previewLims2)
            lims = trainer.dataset.previewLims1
            ic(np.unique(trainer.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]]))
            lims = trainer.dataset.previewLims2
            ic(np.unique(trainer.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]], return_counts=True))
            '''
            _plt.plotCropSample5(trainer.image_stack[...,previewBandsT0], trainer.image_stack[...,previewBandsT1], trainer.mean_prob, 
                    trainer.error_mask_to_show_rgb[...,::-1], trainer.uncertainty_to_show, 
                    lims = trainer.dataset.previewLims1, 
                    titles = ['Snippet $\mathregular{T_{-1}}$', 'Snippet $\mathregular{T_0}$', 'Predicted Probability', 'Predicted', 'Uncertainty'],
                    cmaps = [plt.cm.gray, plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                    maskBackground = [False, False, True, False, True],
                    invertMask = [False, False, False, False, False], uncertainty_vlims = uncertainty_vlims)
            plt.savefig('output/figures/' + trainer.dataset.__class__.__name__ + 'PredictSampleUncertainty1_.png', dpi=150, bbox_inches='tight')

            _plt.plotCropSample5(trainer.image_stack[...,previewBandsT0], trainer.image_stack[...,previewBandsT1], trainer.mean_prob, 
                    trainer.error_mask_to_show_rgb[...,::-1], trainer.uncertainty_to_show, 
                    lims = trainer.dataset.previewLims2, 
                    titles = ['Snippet $\mathregular{T_{-1}}$', 'Snippet $\mathregular{T_0}$', 'Predicted Probability', 'Predicted', 'Uncertainty'],
                    cmaps = [plt.cm.gray, plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                    maskBackground = [False, False, True, False, True],
                    invertMask = [False, False, False, False, False], uncertainty_vlims = uncertainty_vlims)
            plt.savefig('output/figures/' + trainer.dataset.__class__.__name__ + 'PredictSampleUncertainty2_.png', dpi=150, bbox_inches='tight')

            _plt.plotCropSample5(trainer.image_stack[...,previewBandsT0], trainer.image_stack[...,previewBandsT1], trainer.mean_prob, 
                    trainer.error_mask_to_show_rgb[...,::-1], trainer.uncertainty_to_show, 
                    lims = trainer.dataset.previewLims2, 
                    titles = ['Snippet $\mathregular{T_{-1}}$', 'Snippet $\mathregular{T_0}$', 'Predicted Probability', 'Predicted', 'Uncertainty'],
                    cmaps = [plt.cm.gray, plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                    maskBackground = [False, False, True, False, True],
                    invertMask = [False, False, False, False, False], uncertainty_vlims = uncertainty_vlims,
                    colorbar = True)
            plt.savefig('output/figures/' + trainer.dataset.__class__.__name__ + 'PredictSampleUncertaintyColorbar_.png', dpi=150, bbox_inches='tight')

