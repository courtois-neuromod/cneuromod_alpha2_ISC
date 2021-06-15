import glob
import itertools
from nilearn import image, plotting, surface

subjects = ['sub-01', 'sub-02', 'sub-03',
            'sub-04', 'sub-05', 'sub-06']
tasks = ['bourne', 'figures', 'life', 'life_run-2', 'wolf']


def surface_isc_plots(subject, task, views=['lateral', 'medial']):
    """
    Parameters
    ----------
    subject : str
        Subject identifier for which to generate ISC plots.
    task: list
        Tasks for which to generate ISC plots.
    views : list
        View for which to generate ISC plots. Accepted values are
        ['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'].
        Defaults to ['lateral', 'medial'].
    """

    for view, task in itertools.product(views, tasks):
        isc_files = sorted(glob.glob(f'ISC_{task}_{subject}.nii.gz'))
        average_isc = image.mean_img(isc_files)

        # plot right hemisphere
        texture = surface.vol_to_surf(average_isc, fsaverage.pial_right)
        plotting.plot_surf_stat_map(
            fsaverage.pial_right, texture, hemi=hemi,
            colorbar=colorbar, threshold=threshold, vmax=vmax,
            bg_map=fsaverage.sulc_right, view=view)
        plt.savefig(f'right_{view}_surfplot_ISC_on_{task}_{subject}.png',
                    bbox_inches='tight')

        # plot left hemisphere
        texture = surface.vol_to_surf(average_isc, fsaverage.pial_left)
        plotting.plot_surf_stat_map(
            fsaverage.pial_left, texture, hemi=hemi,
            colorbar=colorbar, threshold=threshold, vmax=vmax,
            bg_map=fsaverage.sulc_left, view=view)
        plt.savefig(f'left_{view}_surfplot_ISC_on_{task}_{subject}.png',
                    bbox_inches='tight')


def plot_corr_mtx(kind, data_dir, mask_img):
    """
    kind : str
        Kind of ISC, must be in ['spatial', 'temporal']
    data_dir : str
        The path to the postprocess data directory on disk.
        Should contain all generated ISC maps.
    mask_img : str
        Path to the mask image on disk.
    """
    from netneurotools.plotting import plot_mod_heatmap
    methods = ['anat_inter_subject', 'pairwise_scaled_orthogonal']
    if kind not in ['spatial', 'temporal']:
        err_msg = 'Unrecognized ISC type! Must be spatial or temporal'
        raise ValueError(err_msg)

    for method in methods:
        isc_files = sorted(glob.glob(opj(
            data_dir, f'{kind}ISC*{method}*.nii.gz')))
        masker = NiftiMasker(mask_img=mask_img)

        isc = [masker.fit_transform(i).mean(axis=0) for i in isc_files]
        corr = np.corrcoef(np.row_stack(isc))

        # our 'communities' are which film was presented
        movies = [i.split('_on_')[-1].strip('.nii.gz') for i in isc_files]
        num = [i for i, m in enumerate(set(movies))]
        mapping = dict(zip(set(movies), num))
        comm = list(map(mapping.get, movies))

        plot_mod_heatmap(corr, communities=np.asarray(comm),
                         inds=range(len(corr)), edgecolor='white')
        plt.savefig(f'{kind}ISC_correlation_matrix_with_{method}.png',
                    bbox_inches='tight')


def plot_axial_slice(tasks, data_dir, kind='temporal'):
    """
    Parameters
    ----------
    kind : str
        Kind of ISC, must be in ['spatial', 'temporal']
    data_dir : str
        The path to the postprocessed data directory on disk.
        Should contain all generated ISC maps.
    """
    methods = ['anat_inter_subject', 'pairwise_scaled_orthogonal', 'smoothing']

    if kind not in ['spatial', 'temporal']:
        err_msg = 'Unrecognized ISC type! Must be spatial or temporal'
        raise ValueError(err_msg)

    for task, method in itertools.product(tasks, methods):
        files = glob.glob(opj(
            data_dir, f'{kind}ISC_*{method}*_on_{task}.nii.gz'))
        files = [f for f in files if 'source' not in f]
        average = image.mean_img(files)

        # NOTE: threshold may need to be adjusted for each decoding task
        plotting.plot_stat_map(
            average,
            threshold=0.1, vmax=0.75, symmetric_cbar=False,
            display_mode='z', cut_coords=[-24, -6, 7, 25, 37, 51, 65]
        )
        plt.savefig(f'{kind}ISC_with_{method}_on_{task}.png',
                    bbox_inches='tight')