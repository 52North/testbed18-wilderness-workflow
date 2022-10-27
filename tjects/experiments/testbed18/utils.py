import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from tlib import tlearn, ttorch, tgeo, tutils
from tjects.experiments.testbed18 import config


def load_file_infos(raw: bool = False):
    """
    Load file infos object which includes a dataframe of all file infos.

    :param raw: if True, load original file infos from anthroprotect dataset; if False, lad infos from working_dir
    :return: file infos object
    """

    if raw:
        csv_file = config.file_infos_path_raw
    else:
        csv_file = os.path.join(load_trainer().log_dir, 'file_infos.csv')

    fi = tlearn.data.files.FileInfosGeotif(
        csv_file=csv_file,
        folder=config.data_folder_tiles,
    )

    return fi


def load_trainer():
    """
    Load trainer from working_dir.

    :return: loaded trainer object
    """

    trainer = ttorch.train.load_trainer(log_dir=config.log_path, device=config.device)
    trainer.datamodule.folder = config.data_folder_tiles
    return trainer


def load_asos():
    """
    Load asos object from working_dir.

    :return: asos object
    """

    log_dir = load_trainer().log_dir
    asos = tutils.files.load_with_pickle(file=os.path.join(log_dir, 'asos', 'asos.pkl'))
    asos.output_folder = os.path.join(log_dir, 'asos')

    return asos


def get_corresponding_folder(folder, folder_name):
    """
    There are three subfolders in the anthroprotect dataset: s2, s2_scl, lcs. The last part of a path can be changed
    with this function.

    :param folder: path to folder, e.g. .../anthroprotect/s2
    :param folder_name: new subfolder name, e.g. 'lcs'
    :return: new path string, e.g. .../anthroprotect/lcs
    """

    return os.path.join('/'.join(folder.split('/')[:-1]), folder_name)


def get_corresponding_files(files, folder_name):
    """
    Changes the last folder in directory. E.g. changes '.../tiles/s2/file.tif' to '.../tiles/lcs/file.tif'.

    :param files: list of file directories
    :param folder_name: new folder name, e.g. 'lcs'
    :return: list of changed file directories
    """

    files_was_list = True
    if isinstance(files, str):
        files_was_list = False
        files = [files]

    corresponding_files = []
    for file in files:

        file = file.split('/')
        file[-2] = folder_name
        file = '/'.join(file)

        corresponding_files.append(file)

    if files_was_list:
        return corresponding_files
    else:
        return corresponding_files[0]

    
def check_if_prepend_folder(*files):
    """
    Return bool if only file name is given and folder path must be prepend.
    
    :param files: list of files
    :return: bool if only file name is given
    """

    # if only file name and no dir prepend folder to files
    if files[0].split('/')[0] != config.data_folder_tiles.split('/')[0]:
        return True
    else:
        return False


def predict_score(*files, batch_size: int=config.batch_size, disable_tqdm: bool = False):
    """
    Load trainer and predict classification scores of given files using the whole network.

    :param files: list of files
    :param batch_size: batch size
    :param disable_tqdm: bool if tqdm shall be disabled
    :return: predictions
    """

    trainer = load_trainer()
    dataset = trainer.datamodule.get_dataset(files, prepend_folder=check_if_prepend_folder(*files))
    preds = trainer.predict_dataset(dataset, batch_size=batch_size, disable_tqdm=disable_tqdm)

    return preds


def predict_activation_maps(*files, batch_size: int = config.batch_size, disable_tqdm: bool = False):
    """
    Load trainer and predict activation maps using the unet.

    :param files: list of files:
    :param batch_size: batch_size
    :param disable_tqdm: bool if tqdm shall be disabled
    :return: predicted activation maps
    """

    trainer = load_trainer()
    model = trainer.model.unet
    model = ttorch.modules.wrapper.AutoMoveData(model)

    # split images before predicting if they are larger than config.max_image_size
    model = ttorch.modules.wrapper.SplitPredictMerge(
        model=model,
        max_image_size=config.max_image_size,
        in_channels=config.in_channels,
        disable_tqdm=disable_tqdm
    )

    dataset_kwargs = {
        'channels': None,
        'x_normalization': (0, 10000),
        'clip_range': (0, 1),
        'y_normalization': None,
        'n_classes': 1,
        'use_rasterio': True,
        'rgb_channels': [2, 1, 0],
        'val_range': (0, 1024)
    }

    labels = [0] * len(files)
    dataset = ttorch.data.images.Dataset(files=files, labels=labels, **dataset_kwargs)
    # dataset = trainer.datamodule.get_dataset(files, prepend_folder=check_if_prepend_folder(*files))

    preds = ttorch.utils.predict_dataset(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        disable_tqdm=disable_tqdm,
    )

    return preds


def predict_sensitivity_maps(*files, batch_size: int = config.batch_size, disable_tqdm: bool = False):
    """
    Predict activation maps and then sensitivity maps and return sensitivity maps.

    :param files: list of files:
    :param batch_size: batch_size
    :param disable_tqdm: bool if tqdm shall be disabled
    :return: predicted sensitivity maps
    """

    asos = load_asos()
    
    unet_maps = predict_activation_maps(*files, batch_size=batch_size, disable_tqdm=disable_tqdm)
    sensitivities = asos.predict_sensitivities(unet_maps, disable_tqdm=disable_tqdm)

    return sensitivities


def predict_osm(*files, patch_size: int = 8, stride: int = 4):
    """
    Predicts occlusion sensitivity map accoring to Zeiler and Fergus of given files.

    :param files: list of file names or file directories
    :param patch_size: patch size
    :param stride: stride
    :return: negative of occlusion sensitivity map (negative, to make it comparable to ASOS)
    """

    trainer = load_trainer()
    model = trainer.model
    dataset = trainer.datamodule.get_dataset(files, prepend_folder=check_if_prepend_folder(*files))

    osms = []
    for sample in tqdm(dataset, desc='osm sample', leave=False):
        x = sample['x']
        
        # split x into 256*256 tiles
        split_merge = ttorch.data.images.SplitMerge(tensor=x.unsqueeze(0), split_size=256, expand_value=0)
        splitted_xs = split_merge.split()
        
        # get osm for each splitted_x
        splitted_osms = []
        for splitted_x in tqdm(splitted_xs, desc='tiles of sample', leave=False):
            
            # calculate osm
            osm = tlearn.interpret.osm.get_osms(
                model=model, x=splitted_x, patch_size=patch_size, stride=stride, fill_value=0, disable_tqdm=True)
            osm = osm[0]  # we only have one class
            
            splitted_osms.append(osm)
        splitted_osms = torch.stack(splitted_osms)
        merged_osms = split_merge.merge(splitted_osms.unsqueeze(1))
        
        osms.append(merged_osms.squeeze())
    osms = torch.stack(osms)
    osms = np.array(osms)

    return - osms


class GenericS2Plotter:
    """
    Generic S2 plotter
    Works with file paths or arrays
    """
    def __init__(self, val_range_s2: tuple = (0, 2**10)):

        self.channel_indices = {'B2': 0, 'B3': 1, 'B4': 2, 'B5': 3, 'B6': 4,
                                'B7': 5, 'B8': 6, 'B8A': 7, 'B11': 8, 'B12': 9}
        self.channel_names = ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12')
        self.val_range_s2 = val_range_s2
        self.cmap_mask = 'gray'
        self.cmap_ndvi = 'RdYlGn'

    def plot(
        self,
        file,

        plot_s2: bool = True,
        plot_s2_channels: bool = True,

        fig_height: float = 4.6,
    ):
        """
        Plots using matplotlib and plt.show(). No figure or axis is returned.

        :param file: file name or file dir
        :param plot_s2: bool if the following is plotted from Sentinel-2 image: rgb, false color, scl, ndvi, evi;
            note, that the scl plot might have artifacts
        :param s2_channels: bool if all channels of Sentinel-2 image are plotted individually
        :param fig_height: height of the plotted figures
        :return:
        """

        # plot s2 channels
        if plot_s2_channels:
            fig = tgeo.geotif.plot_channels(file=file, ncols=5, figsize=(fig_height * 5, fig_height * 2))
            for idx, ax in enumerate(fig.axes):
                if ax.title.get_text() == '':
                    ax.set_title(self.channel_names[idx])
            fig.tight_layout()
            plt.show()

        # plot s2
        if plot_s2:
            fig, axs = plt.subplots(1, 5, figsize=(fig_height * 5, fig_height))

            # rgb image
            try:  # with channel names
                axs[0] = tgeo.geotif.plot_image(
                    file=file, channels=['B4', 'B3', 'B2'], val_range=self.val_range_s2, ax=axs[0])
            except:  # use channel indices
                channels = [self.channel_indices['B4'], self.channel_indices['B3'], self.channel_indices['B2']]
                axs[0] = tgeo.geotif.plot_image(file=file, channels=channels, val_range=self.val_range_s2, ax=axs[0])

            axs[0].set_title('RGB')

            # false color image
            try:  # with channel names
                axs[1] = tgeo.geotif.plot_image(
                    file=file, channels=['B8', 'B4', 'B3'], val_range=self.val_range_s2, ax=axs[1])
            except:  # use channel indices
                channels = [self.channel_indices['B8'], self.channel_indices['B4'], self.channel_indices['B3']]
                axs[1] = tgeo.geotif.plot_image(file=file, channels=channels, val_range=self.val_range_s2, ax=axs[1])

            axs[1].set_title('NIR-R-G')

            # ndvi
            try:  # with channel names
                image = tgeo.geotif.get_array(file=file, channels=('B8', 'B4'))
            except:  # use channel indices
                channels = [self.channel_indices['B8'], self.channel_indices['B4']]
                image = tgeo.geotif.get_array(file=file, channels=channels)

            ndvi = tgeo.utils.get_ndvi(nir=image[0], red=image[1])

            axs[2] = tutils.plots.plot_original_image_size(
                array=ndvi, plot_cbar=True, cmap=self.cmap_ndvi, vmin=-1, vmax=1, fig=fig, ax=axs[2])
            axs[2].set_title('NDVI')

            # evi
            try:  # with channel names
                image = tgeo.geotif.get_array(file=file, channels=('B8', 'B4', 'B2'))
            except:  # use channel indices
                channels = [self.channel_indices['B8'], self.channel_indices['B4'], self.channel_indices['B2']]
                image = tgeo.geotif.get_array(file=file, channels=channels)

            evi = tgeo.utils.get_evi(nir=image[0], red=image[1], blue=image[2])

            q = 0.02

            axs[3] = tutils.plots.plot_original_image_size(
                array=evi,
                plot_cbar=True,
                cmap=self.cmap_ndvi,
                vmin=np.quantile(evi, q=q),
                vmax=np.quantile(evi, q=1-q),
                fig=fig,
                ax=axs[3])
            axs[3].set_title('EVI')


class Plotter:
    """
    This class helps to plot the results.

    :param folder: folder which contains Sentinel-2 geotifs
    """

    def __init__(self, folder=config.data_folder_tiles):

        self.folder = folder
        
        # plot parameters
        self.cmap_unet_map = 'coolwarm'

    def get_file_dirs(self, *files):
        """
        Return file paths according to given self folder.

        :param files: list of files; if only file name (without directory) is given, paths are joined
        :return: list of files directories
        """

        # if only file name and not directory ... add folders to file name
        if files[0].split('/')[0] != self.folder.split('/')[0]:

            files = [os.path.join(self.folder, file) for file in files]

        return files

    @staticmethod
    def get_vlim(array, q=0.02):
        """
        Returns vmin and vmax for given array and quantile.

        :param array: numpy array
        :param q: quantile
        :return: (vmin, vmax)
        """

        return tutils.plots.get_diverging_clim(array=array, q=q)

    def plot(
        self,
        file,

        plot_unet_maps: bool = False,
        plot_all_unet_maps: bool = False,
        plot_sensitivities: bool = False,
        plot_osm: bool = False,
        fig_height: float = 4.6,
    ):
        """
        Plots using matplotlib and plt.show(). No figure or axis is returned.

        :param file: file name or file dir
        :param plot_unet_maps: bool if unet maps shall be plotted
        :param plot_all_unet_maps: this is only relevant if there are 3 unet maps; if False only rgb image of unet maps
            if plotted; if True all three channels are plotted as well
        :param plot_sensitivities: bool if sensitivity map shall be plotted
        :param plot_osm: bool if occlusion sensitivity map according to Zeiler and Fergus shall be plotted;
            note that this takes some time, because of the computational expensive prediction
        :param fig_height: height of the plotted figures
        :return:
        """

        file = self.get_file_dirs(file)[0]

        # predict
        if plot_unet_maps or plot_sensitivities:
            unet_map = predict_activation_maps(file, disable_tqdm=True)[0]
            if plot_sensitivities:
                asos = load_asos()
                sensitivity = asos.predict_sensitivities(unet_map.unsqueeze(0), disable_tqdm=True)[0]

        # plot unet maps
        if plot_unet_maps:

            n_unet_maps = unet_map.shape[0]

            if plot_all_unet_maps or n_unet_maps != 3:
                fig, axs = plt.subplots(ncols=n_unet_maps, figsize=(fig_height * n_unet_maps, fig_height))
            
            if n_unet_maps == 1:
                axs.imshow(unet_map[0], clim=(-1, 1), cmap=self.cmap_unet_map)
                axs.axis(False)

            elif n_unet_maps > 1 and (plot_all_unet_maps or n_unet_maps != 3):
                for i in range(n_unet_maps):
                    axs[i].imshow(unet_map[i], clim=(-1, 1), cmap=self.cmap_unet_map)
                    axs[i].axis(False)
                    axs[i].set_title(f'U-Net map {i}')
            
            if plot_all_unet_maps or n_unet_maps != 3: 
                fig.tight_layout()
                plt.show()

            if unet_map.shape[0] == 3:  # plot rgb image of unet maps
                unet_map_rgb = unet_map.clone()
                unet_map_rgb = np.array(unet_map_rgb).transpose(1, 2, 0)
                unet_map_rgb = (unet_map_rgb + 1) / 2

                fig, ax = plt.subplots(figsize=(fig_height, fig_height))
                ax.imshow(unet_map_rgb)
                ax.axis(False)
                ax.set_title('U-Net map RGB')
                fig.tight_layout()
                plt.show()

        # plot sensitivities
        if plot_sensitivities:
            fig, ax = asos.plot_sensitivity_map(
                sensitivity, figsize=(fig_height, fig_height))
            ax.set_title('sensitivities')
            plt.show()

        # plot osms
        if plot_osm:
            fig, ax = plt.subplots(figsize=(fig_height, fig_height))
            osm = predict_osm(file)[0]

            asos = load_asos()
            vlim = self.get_vlim(osm)
            ax.imshow(osm, cmap=asos.cmap, vmin=-vlim, vmax=vlim)
            ax.axis(False)
            ax.set_title('OSM')
            fig.tight_layout()
            plt.show()


    def plot_on_map(
        self,
        *files,

        plot_preds: bool=False,
        plot_unet_maps: bool = False,
        plot_all_unet_maps: bool = False,
        plot_sensitivities: bool = False,
        plot_osms: bool = False,

        batch_size: int = config.batch_size,
        m=None,
        **map_kwargs,
    ):
        """
        Plots on ipyleaflet map.

        :param files: list of file names or file dirs
        :param plot_unet_maps: bool if unet maps are plotted
        :param plot_all_unet_maps: this is only relevant if there are 3 unet maps; if False only rgb image of unet maps
            if plotted; if True all three channels are plotted as well
        :param plot_sensitivities: bool if sensitivity map shall be plotted
        :param plot_osm: bool if occlusion sensitivity map according to Zeiler and Fergus shall be plotted;
            note that this takes some time, because of the computational expensive prediction
        :param batch_size: batch_size
        :param m: ipyleaflet map
        :param map_kwargs: ipyleaflet map kwargs
        :return: ipyleaflet map
        """

        if m is None:
            m = tgeo.utils.get_map(**map_kwargs)

        # create sublists with given batch size and run for each sublist
        files_sublists = tutils.lists.create_sublists(files, batch_size)
        for files in tqdm(files_sublists, desc='batch'):
            unet_maps, sensitivities = None, None
            del unet_maps
            del sensitivities

            if plot_preds:
                file_infos = load_file_infos().df
                file_infos = file_infos[file_infos.index.isin(files)]

            files = self.get_file_dirs(*files)

            # plot predictions (true or false)
            if plot_preds:

                truly_predicted_files = file_infos[file_infos['label'] == file_infos['pred']].index.to_list()
                falsely_predicted_files = file_infos[file_infos['label'] != file_infos['pred']].index.to_list()

                # get directories
                truly_predicted_files = [os.path.join(config.data_folder_tiles, file) for file in truly_predicted_files]
                falsely_predicted_files = [os.path.join(config.data_folder_tiles, file) for file in falsely_predicted_files]
                
                if len(truly_predicted_files) > 0:
                    m = tgeo.geotif.location_on_map(
                        *truly_predicted_files, color='green', show_marker=False, description='truly predicted', m=m)

                if len(falsely_predicted_files) > 0:
                    m = tgeo.geotif.location_on_map(
                        *falsely_predicted_files, color='red', show_marker=True, description='falsely predicted', m=m)
            
            # predict
            if plot_unet_maps or plot_sensitivities:
                unet_maps = predict_activation_maps(*files, batch_size=batch_size, disable_tqdm=True)
                if plot_sensitivities:
                    asos = load_asos()
                    sensitivities = asos.predict_sensitivities(unet_maps, disable_tqdm=True)
            
            # plot unet maps
            if plot_unet_maps:

                n_unet_maps = unet_maps.shape[1]
                
                if plot_all_unet_maps or n_unet_maps != 3:
                    
                    for i in range(n_unet_maps):  # for each unet map channel
                        
                        m = tgeo.geotif.arrays_on_map(
                            files=files, arrays=np.array(unet_maps)[:, i], description=f'U-Net map {i}',
                            cmap=self.cmap_unet_map, vmin=-1, vmax=1, m=m,
                            )

                if n_unet_maps == 3:  # plot rgb image of unet maps
                    unet_maps_rgb = unet_maps.clone()
                    unet_maps_rgb = np.array(unet_maps_rgb).transpose(0, 2, 3, 1)
                    unet_maps_rgb = (unet_maps_rgb + 1) / 2
                    m = tgeo.geotif.arrays_on_map(files=files, arrays=unet_maps_rgb, description='U-Net map RGB', m=m)

            # plot sensitivities
            if plot_sensitivities:

                m = tgeo.geotif.arrays_on_map(
                    files=files,
                    arrays=sensitivities,
                    description='sensitivity',
                    cmap=asos.cmap,
                    vmin=-asos.vlim,
                    vmax=asos.vlim,
                    m=m
                )

            # plot osms
            if plot_osms:
                osms = predict_osm(*files)

                asos = load_asos()
                vlim = self.get_vlim(osms)
                vlim = asos.vlim
                m = tgeo.geotif.arrays_on_map(
                    files=files, arrays=osms, description='osms', cmap=asos.cmap, vmin=-vlim, vmax=vlim, m=m)

                # separate
                #for i in range(len(files)):
                #    file = files[i]
                #    osm = osms[i]
                #    vlim = self.get_vlim(osm)
                #    m = tgeo.geotif.arrays_on_map(
                #        files=file, arrays=osm, description='osm', cmap=asos.cmap, vmin=-vlim, vmax=vlim, m=m)


        
        return m

    def to_tif(
        self,
        *files,
        output_folder,

        plot_unet_maps: bool = False,
        plot_all_unet_maps: bool = False,
        plot_sensitivities: bool = False,
        plot_osms: bool = False,

        batch_size: int = config.batch_size,
    ):
        """
        Writes readable tif files (rgb or grey-channel) into given output_folder.

        :param files: list of file names or file dirs
        :param output_folder: output folder in which tifs are saved
        :param plot_unet_maps: bool if unet maps are plotted
        :param plot_all_unet_maps: this is only relevant if there are 3 unet maps; if False only rgb image of unet maps
            if plotted; if True all three channels are plotted as well
        :param plot_sensitivities: bool if sensitivity map shall be plotted
        :param plot_osm: bool if occlusion sensitivity map according to Zeiler and Fergus shall be plotted;
            note that this takes some time, because of the computational expensive prediction
        :param batch_size: batch size
        :return:
        """

        # create sublists with given batch size and run for each sublist
        files_sublists = tutils.lists.create_sublists(files, batch_size)
        for files in tqdm(files_sublists, desc='batch'):
            unet_maps, sensitivities = None, None
            del unet_maps
            del sensitivities

            files = self.get_file_dirs(*files)

            # predict
            if plot_unet_maps or plot_sensitivities:
                unet_maps = predict_activation_maps(*files, batch_size=batch_size, disable_tqdm=True)
                if plot_sensitivities:
                    asos = load_asos()
                    sensitivities = asos.predict_sensitivities(unet_maps, disable_tqdm=True)
            
            # plot unet maps
            if plot_unet_maps:

                if plot_all_unet_maps or unet_maps.shape[1] != 3:
                    for j in range(unet_maps.shape[1]):

                        tgeo.geotif.arrays_to_rgb_tif(
                            files=files,
                            arrays=unet_maps[:, j],
                            output_folder=tutils.files.join_paths(output_folder, f'unet_map_{j}'),
                            use_plot=True,
                            cmap=self.cmap_unet_map,
                            vmin=-1,
                            vmax=1,
                        )

                if unet_maps.shape[1] == 3:
                    unet_maps_rgb = unet_maps.clone()
                    unet_maps_rgb = np.array(unet_maps_rgb)
                    unet_maps_rgb = (unet_maps_rgb + 1) / 2
                        
                    tgeo.geotif.arrays_to_rgb_tif(
                        files=files,
                        arrays=unet_maps_rgb,
                        output_folder=tutils.files.join_paths(output_folder, f'unet_maps'),
                    )

            # plot sensitivities
            if plot_sensitivities:
                tgeo.geotif.arrays_to_rgb_tif(
                    files=files,
                    arrays=sensitivities,
                    output_folder=tutils.files.join_paths(output_folder, f'sensitivities'),
                    use_plot=True,
                    cmap=asos.cmap,
                    vmin=-asos.vlim,
                    vmax=asos.vlim,
                )

            # plot osms
            if plot_osms:
                osms = predict_osm(*files)
                asos = load_asos()

                for i in range(len(files)):
                    file = files[i]
                    osm = osms[i]

                    for q in [0.005, 0.02, 0.1, 0.5]:
                        vlim = self.get_vlim(osm, q=q)
                        print(f'{file}, q={q}: {vlim/asos.vlim}*asos.vlim')
                        tgeo.geotif.arrays_to_rgb_tif(
                            files=file,
                            arrays=osm,
                            output_folder=tutils.files.join_paths(output_folder, f'osms_quantile-{q}'),
                            use_plot=True,
                            cmap=asos.cmap,
                            vmin=-vlim,
                            vmax=vlim,
                        )
                    for k in [1, 2, 5, 10, 100, 500, 1000]:
                        vlim = k * asos.vlim
                        tgeo.geotif.arrays_to_rgb_tif(
                            files=file,
                            arrays=osm,
                            output_folder=tutils.files.join_paths(output_folder, f'osms_asoslim-{k}'),
                            use_plot=True,
                            cmap=asos.cmap,
                            vmin=-vlim,
                            vmax=vlim,
                        )
