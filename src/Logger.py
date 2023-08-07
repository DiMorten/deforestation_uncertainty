import pathlib
import matplotlib.pyplot as plt
import src.plot as _plt
import importlib
importlib.reload(_plt)
import numpy as np
from icecream import ic
import pdb
class Logger():
    def createLogFolders(self, dataset):
        # figures_path = 'output/figures' + dataset.__class__.__name__ + '/'
        # pathlib.Path(figures_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path("output/figures/").mkdir(parents=True, exist_ok=True)
        self.title_name = 'ResUnet'
    def plotFigure(self, figure, name='output/figure', cmap = plt.cm.gray, 
                   savefig=False, figsize=(10,10), dpi=200, axis_off = True):
        plt.figure(figsize=figsize)
        plt.imshow(figure, cmap=cmap)

        # title_name = 'ResUnet'
        if axis_off == True:
            plt.axis('off')
        if savefig == True:
            plt.savefig('output/figures/' + name, dpi=150, bbox_inches='tight')

    def plotFigure2(self, figure, name='output/figure', cmap = plt.cm.gray, savefig=False, figsize=(15,15), dpi=300):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(figure)

        fig.savefig('output/figures/Para' + name, dpi=dpi, bbox_inches='tight')

    def snipDataset(self, dataset, idx, coords_train, patchesHandler, image_stack, label_mask):
        print(coords_train[idx])
        image_patch, reference_patch = patchesHandler.getPatch(
            image_stack, label_mask, coords_train, idx = idx)

        ic(np.mean(image_patch[...,dataset.previewBandsSnip[0]]), np.mean(image_patch[...,dataset.previewBandsSnip[1]]))
        _plt.plotCropSample4(image_patch[...,dataset.previewBandsSnip[0]], image_patch[...,dataset.previewBandsSnip[1]], 
                reference_patch, reference_patch,
                lims = None, 
                titles = ['Optical T0', 'Optical T1', 'Reference', 'Reference 2'],
                cmaps = [plt.cm.gray, plt.cm.gray, plt.cm.gray, plt.cm.gray],
                maskBackground = [False, False, False, False],
                invertMask = [False, False, False, False])        

    def plotHistory(self, manager):
        plt.clf()
        plt.plot(manager.history.history['loss'])
        plt.plot(manager.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('loss_history_exp{}.png'.format(manager.exp))


    def plotLossTerms(self, manager):
        l = ['loss', 'val_loss', 'KL_term', 'val_KL_term', 'loglikelihood_term', 'val_loglikelihood_term']


        plt.figure(2)
        plt.clf()
        plt.plot(manager.history.history[l[0]])
        plt.plot(manager.history.history[l[1]])
        plt.plot(manager.history.history[l[2]])
        plt.plot(manager.history.history[l[3]])
        plt.plot(manager.history.history[l[4]])
        plt.plot(manager.history.history[l[5]])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(l, loc='upper left')
        plt.savefig('loss_history_exp{}.png'.format(manager.exp))

        plt.figure(3)
        plt.clf()
        plt.plot(manager.history.history[l[2]])
        plt.plot(manager.history.history[l[3]])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(l[2:4], loc='upper left')
        plt.savefig('loss_history_KL_term_exp{}.png'.format(manager.exp))

        plt.figure(4)
        plt.clf()
        plt.plot(manager.history.history[l[4]])
        plt.plot(manager.history.history[l[5]])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(l[4:6], loc='upper left')
        plt.savefig('loss_history_log_likelihood_term_exp{}.png'.format(manager.exp))

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



    def plotCropSample(self, manager):
        self.plotCropSampleFlag = True
        if self.plotCropSampleFlag == True:
                # import matplotlib
                # customCmap = matplotlib.colors.ListedColormap(['black', 'red'])
                ic(manager.dataset.previewLims1, manager.dataset.previewLims2)
                lims = manager.dataset.previewLims1
                ic(np.unique(manager.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]]))
                lims = manager.dataset.previewLims2
                ic(np.unique(manager.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]], return_counts=True))

                _plt.plotCropSample4(manager.image_stack[...,manager.dataset.previewBands], manager.mean_prob, 
                        manager.error_mask_to_show_rgb[...,::-1], manager.uncertainty_to_show, 
                        lims = manager.dataset.previewLims1, 
                        titles = ['Optical', 'Predict Probability', 'Predicted', 'Uncertainty'],
                        cmaps = [plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                        maskBackground = [False, True, False, True],
                        invertMask = [False, False, False, False])
                save_name = 'output/figures/{}PredictSampleUncertainty1_exp{}.png'.format(
                    manager.dataset.__class__.__name__, str(manager.exp))
                print("Saving crop sample 1 in",save_name)
                plt.savefig(save_name, dpi=150, bbox_inches='tight')

                _plt.plotCropSample4(manager.image_stack[...,manager.dataset.previewBands], manager.mean_prob, 
                        manager.error_mask_to_show_rgb[...,::-1], manager.uncertainty_to_show, 
                        lims = manager.dataset.previewLims2, 
                        titles = ['Optical', 'Predict Probability', 'Predicted', 'Uncertainty'],
                        cmaps = [plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                        maskBackground = [False, True, False, True],
                        invertMask = [False, False, False, False])
                save_name = 'output/figures/{}PredictSampleUncertainty2_exp{}.png'.format(
                    manager.dataset.__class__.__name__, str(manager.exp))
                print("Saving crop sample 2 in",save_name)                
                plt.savefig(save_name, dpi=150, bbox_inches='tight')

    def plotCropSample(self, manager):
        uncertainty_vlims = [np.min(manager.uncertainty_to_show), np.max(manager.uncertainty_to_show)]

        self.plotCropSampleFlag = True
        if self.plotCropSampleFlag == True:
            ic(manager.dataset.previewLims1, manager.dataset.previewLims2)
            lims = manager.dataset.previewLims1
            ic(np.unique(manager.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]]))
            lims = manager.dataset.previewLims2
            ic(np.unique(manager.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]], return_counts=True))

            _plt.plotCropSample4(manager.image_stack[...,manager.dataset.previewBandsSnip[-1]], manager.mean_prob, 
                    manager.error_mask_to_show_rgb[...,::-1], manager.uncertainty_to_show, 
                    lims = manager.dataset.previewLims1, 
                    titles = ['Snippet', 'Predicted Probability', 'Predicted', 'Uncertainty'],
                    cmaps = [plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                    maskBackground = [False, True, False, True],
                    invertMask = [False, False, False, False], uncertainty_vlims = uncertainty_vlims)
            save_name = 'output/figures/{}PredictSampleUncertainty1_exp{}.png'.format(
                manager.dataset.__class__.__name__, str(manager.exp))
            plt.savefig(save_name, dpi=150, bbox_inches='tight')

            _plt.plotCropSample4(manager.image_stack[...,manager.dataset.previewBandsSnip[-1]], manager.mean_prob, 
                    manager.error_mask_to_show_rgb[...,::-1], manager.uncertainty_to_show, 
                    lims = manager.dataset.previewLims2, 
                    titles = ['Snippet', 'Predicted Probability', 'Predicted', 'Uncertainty'],
                    cmaps = [plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                    maskBackground = [False, True, False, True],
                    invertMask = [False, False, False, False], uncertainty_vlims = uncertainty_vlims)
            save_name = 'output/figures/{}PredictSampleUncertainty2_exp{}.png'.format(
                manager.dataset.__class__.__name__, str(manager.exp))
            plt.savefig(save_name, dpi=150, bbox_inches='tight')

            _plt.plotCropSample4(manager.image_stack[...,manager.dataset.previewBandsSnip[-1]], manager.mean_prob, 
                    manager.error_mask_to_show_rgb[...,::-1], manager.uncertainty_to_show, 
                    lims = manager.dataset.previewLims2, 
                    titles = ['Snippet', 'Predicted Probability', 'Predicted', 'Uncertainty'],
                    cmaps = [plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                    maskBackground = [False, True, False, True],
                    invertMask = [False, False, False, False], uncertainty_vlims = uncertainty_vlims,
                    colorbar = True)
            plt.savefig('output/figures/' + manager.dataset.__class__.__name__ + 'PredictSampleUncertaintyColorbar.png', dpi=150, bbox_inches='tight')


    def plotCropSampleLandsat(self, manager, landsat_ims):

        uncertainty_vlims = [np.min(manager.uncertainty_to_show), np.max(manager.uncertainty_to_show)]

        self.plotCropSampleFlag = True
        if self.plotCropSampleFlag == True:
            ic(manager.dataset.previewLims1, manager.dataset.previewLims2)
            lims = manager.dataset.previewLims1
            ic(np.unique(manager.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]]))
            lims = manager.dataset.previewLims2
            ic(np.unique(manager.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]], return_counts=True))

            ims = [landsat_ims[0], landsat_ims[1], landsat_ims[2],
                    manager.mean_prob, 
                    manager.error_mask_to_show_rgb[...,::-1], manager.uncertainty_to_show]
            print([x.shape for x in ims])

            # manager.dataset.prodes_dates = ['21/07/2018', '24/07/2019', '26/07/2020']
            # manager.dataset.prodes_dates = ['02/08/2019', '05/08/2020', '22/07/2021']
            
            titles = ['Snippet $\mathregular{T_{-1}}$'+' ({})'.format(manager.dataset.prodes_dates_to_print[0]), 
                      'Snippet $\mathregular{T_{0}}$'+' ({})'.format(manager.dataset.prodes_dates_to_print[1]), 
                      'Snippet $\mathregular{T_{1}}$'+' ({})'.format(manager.dataset.prodes_dates_to_print[2]), 
                            'Prediction Probability at $\mathregular{T_{0}}$', 
                            'Prediction at $\mathregular{T_{0}}$', 
                            'Uncertainty at $\mathregular{T_{0}}$']
            cmaps = [plt.cm.gray, plt.cm.gray, plt.cm.gray,
                             'jet', plt.cm.gray, 'jet']

            # manager.dataset.hspace = [-0.1, 0.03]
            _plt.plotCropSample6(ims[:], 
                    lims = manager.dataset.previewLims1, 
                    titles = titles,
                    cmaps = cmaps,
                    uncertainty_vlims = uncertainty_vlims,
                    polygons = manager.dataset.polygons[0],
                    hspace = manager.dataset.hspace[0]) # -0.1
            save_name = 'output/figures/{}PredictSampleUncertaintyLandsat1_exp{}.png'.format(
                manager.dataset.__class__.__name__, str(manager.exp))
            plt.savefig(save_name, dpi=150, bbox_inches='tight')
            

            _plt.plotCropSample6(ims[:], 
                    lims = manager.dataset.previewLims2, 
                    titles = titles,
                    cmaps = cmaps,
                    uncertainty_vlims = uncertainty_vlims,
                    polygons = manager.dataset.polygons[1],
                    hspace = manager.dataset.hspace[1]) # 0.
            save_name = 'output/figures/{}PredictSampleUncertaintyLandsat2_exp{}.png'.format(
                manager.dataset.__class__.__name__, str(manager.exp))
            plt.savefig(save_name, dpi=150, bbox_inches='tight')

            _plt.plotCropSample6(ims[:], 
                    lims = manager.dataset.previewLims2, 
                    titles = titles,
                    cmaps = cmaps,
                    #maskBackground = [False, True, False, True],
                    #invertMask = [False, False, False, False], 
                    uncertainty_vlims = uncertainty_vlims,
                    polygons = manager.dataset.polygons[1],
                    colorbar = True,
                    hspace = manager.dataset.hspace[1])
            plt.savefig('output/figures/' + manager.dataset.__class__.__name__ + 'PredictSampleUncertaintyLandsatColorbar.png', dpi=150, bbox_inches='tight')

    def plotCropSampleT0T1(self, manager):
        previewBandsT0 = manager.dataset.previewBandsSnip[-2]
        previewBandsT1 = manager.dataset.previewBandsSnip[-1]

        uncertainty_vlims = [np.min(manager.uncertainty_to_show), np.max(manager.uncertainty_to_show)]

        self.plotCropSampleFlag = True
        if self.plotCropSampleFlag == True:

            # import matplotlib
            # customCmap = matplotlib.colors.ListedColormap(['black', 'red'])
            '''
            ic(manager.dataset.previewLims1, manager.dataset.previewLims2)
            lims = manager.dataset.previewLims1
            ic(np.unique(manager.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]]))
            lims = manager.dataset.previewLims2
            ic(np.unique(manager.mask_amazon_ts[lims[0]:lims[1], lims[2]:lims[3]], return_counts=True))
            '''
            _plt.plotCropSample5(manager.image_stack[...,previewBandsT0], manager.image_stack[...,previewBandsT1], manager.mean_prob, 
                    manager.error_mask_to_show_rgb[...,::-1], manager.uncertainty_to_show, 
                    lims = manager.dataset.previewLims1, 
                    titles = ['Snippet $\mathregular{T_{-1}}$', 'Snippet $\mathregular{T_0}$', 'Predicted Probability', 'Predicted', 'Uncertainty'],
                    cmaps = [plt.cm.gray, plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                    maskBackground = [False, False, True, False, True],
                    invertMask = [False, False, False, False, False], uncertainty_vlims = uncertainty_vlims)
            plt.savefig('output/figures/' + manager.dataset.__class__.__name__ + 'PredictSampleUncertainty1_.png', dpi=150, bbox_inches='tight')

            _plt.plotCropSample5(manager.image_stack[...,previewBandsT0], manager.image_stack[...,previewBandsT1], manager.mean_prob, 
                    manager.error_mask_to_show_rgb[...,::-1], manager.uncertainty_to_show, 
                    lims = manager.dataset.previewLims2, 
                    titles = ['Snippet $\mathregular{T_{-1}}$', 'Snippet $\mathregular{T_0}$', 'Predicted Probability', 'Predicted', 'Uncertainty'],
                    cmaps = [plt.cm.gray, plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                    maskBackground = [False, False, True, False, True],
                    invertMask = [False, False, False, False, False], uncertainty_vlims = uncertainty_vlims)
            plt.savefig('output/figures/' + manager.dataset.__class__.__name__ + 'PredictSampleUncertainty2_.png', dpi=150, bbox_inches='tight')

            _plt.plotCropSample5(manager.image_stack[...,previewBandsT0], manager.image_stack[...,previewBandsT1], manager.mean_prob, 
                    manager.error_mask_to_show_rgb[...,::-1], manager.uncertainty_to_show, 
                    lims = manager.dataset.previewLims2, 
                    titles = ['Snippet $\mathregular{T_{-1}}$', 'Snippet $\mathregular{T_0}$', 'Predicted Probability', 'Predicted', 'Uncertainty'],
                    cmaps = [plt.cm.gray, plt.cm.gray, 'jet', plt.cm.gray, 'jet'],
                    maskBackground = [False, False, True, False, True],
                    invertMask = [False, False, False, False, False], uncertainty_vlims = uncertainty_vlims,
                    colorbar = True)
            plt.savefig('output/figures/' + manager.dataset.__class__.__name__ + 'PredictSampleUncertaintyColorbar_.png', dpi=150, bbox_inches='tight')

