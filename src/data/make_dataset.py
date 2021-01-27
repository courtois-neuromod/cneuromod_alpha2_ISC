import re
import itertools
import numpy as np
import nibabel as nib
from pathlib import Path
from numpy.lib import recfunctions
from nilearn import image, input_data

subjects = ['sub-01', 'sub-02', 'sub-03',
            'sub-04', 'sub-05', 'sub-06']


def _subset_confounds(tsv):
    """
    Only retain those confounds listed in `keep_confounds`

    Parameters
    ----------
    tsv: str
        Local file path to the fMRIPrep generated confound files
    """
    keep_confounds = ['trans_x', 'trans_y', 'trans_z',
                      'rot_x', 'rot_y', 'rot_z',
                      'csf', 'white_matter']

    # load in tsv and subset to only include our desired regressors
    tsv = str(tsv)
    confounds = np.recfromcsv(tsv, delimiter='\t')
    selected_confounds = confounds[keep_confounds]
    return selected_confounds


def _get_segment(files, task):
    """
    Gets the segment of the movie (as encoded in the task
    value) for sorting. Since some segments were shown out
    of order, this is more reliable than a naive sort of
    the filenames.

    Parameters
    ----------
    files : list of str
        The list of filenames that should be sorted.

    task : str
        The name of the task, not encoding the segment.
        For example, if the task value is 'figures04'
        this string should be 'figures'.

    Returns
    -------
    segments: list of str
        The list of filenames, sorted by movie segment.
    """
    segments = sorted(files, key=lambda x: int(
        re.search(f'task-{task}(\d+)', x).group(1)))
    return segments


def _nifti_mask_movie(scan, mask, confounds, smoothing_fwhm=None):
    """
    Cleans movie data, including standardizing, detrending,
    and high-pass filtering at 0.01Hz. Corrects for supplied
    confounds. Optionally smooths time series.

    Parameters
    ----------
    scan: niimg_like
        An in-memory niimg
    mask: str
        The (brain) mask within which to process data.
    confounds: np.ndarray
        Any confounds to correct for in the cleaned data set.
    """
    # niftimask and clean data
    masker = input_data.NiftiMasker(mask_img=mask, standardize=True,
                                    detrend=True, high_pass=0.01, t_r=1.49,
                                    smoothing_fwhm=smoothing_fwhm)
    cleaned = masker.fit_transform(scan, confounds=confounds)
    return masker.inverse_transform(cleaned)


def create_data_dictionary():
    """
    Creates a data_dictionary for easily accessing all of the relevant
    parameters of a movie10 task.

    Returns
    -------
    data_dictionary :  dict
        A dictionary with the following fields:
         - segment_lengths : list
            The number of frames to consider in each segment
         - regr_str : str
            The string to use when querying regressors
         - tmpl_str : str
            The string to use when querying BOLD files, containing
            the template name
    """
    tasks = ['bourne', 'wolf', 'figures', 'life']

    # segment lengths varied slightly across subjects. `segment_length.py`
    # calculates the minimum segment length across subjects, indexed here.
    # this allows us to ensure that all subjects have the same amount of data.
    segment_lengths = [
        [403, 405, 405, 405, 405, 405, 405, 405, 405, 380],
        [406, 406, 406, 406, 406, 406, 406, 406, 406, 406,
         406, 406, 406, 406, 406, 406, 498],
        [410, 410, 410, 409, 408, 410, 410, 409, 410, 409, 409, 373],
        [406, 406, 406, 406, 384]
    ]

    # bourne and wolf do not have run key-value, figures and life do.
    # explicitly make this task pairing for file string, regressor string
    regr_str = ['desc-confounds_regressors.tsv',
                'run-1_desc-confounds_regressors.tsv']
    tmpl_str = ['space-MNI152NLin2009cAsym_desc-preproc_bold',
                'run-1_space-MNI152NLin2009cAsym_desc-preproc_bold']
    regr_pairs = [_ for x in regr_str for _ in (x,)*2]
    tmpl_pairs = [_ for x in tmpl_str for _ in (x,)*2]

    data_dictionary = {
        str(t) : {'segment_lengths': seg_len,
        'regr_str': regr, 'tmpl_str': tmpl}
         for t, seg_len, regr, tmpl in zip(
                            tasks, segment_lengths, regr_pairs, tmpl_pairs)}

    return tasks, data_dictionary


def subset_and_process_movie10(bold_files, regressors, segment_lengths,
                               n_segments=None, fwhm=None):
    """
    It also applies basic postprocessing, correcting for a subset of confounds
    (the six motion parameters, average WM and CSF signals), high-pass filtering
    at 0.01Hz, as well as standardizing and detrending the data.

    In order to make comparisons across subjects, it also subsets each
    segment to the minimum length available across all subjects. Please see
    the function `create_data_dictionary` for all utilized lengths.

    Note that the movie10 tasks are long, with the longest movie
    (Wolf of Wall Street) clocking in at almost 7000 frames. This function is
    therefore also designed to subset the movie to a set number of segments;
    for example, to match the number of frames across tasks. Although this
    behavior is off by default (and controllable with the n_segments
    argument), note that if you choose to process the whole movie you can
    expect a very high memory usage.

    Parameters
    ----------
    bold_files : list
        A list of the BOLD file names to subset and process.
    regressors : list
        A list of the regressor file names from which to extract confounds.
        See `_subset_confounds` for more information.
    segment_lengths : list
        A list of the number of frames to consider from each segment.
    n_segments : int
        The number of segments to subset from the movie.
        Will error if the number of segments requested is more than are
        available in the movie.
    fwhm : int
        The size of the Gaussian smoothing kernel to apply.

    Returns
    -------
    postproc_fname : str
        Filename for the concatenated postprocessed file, correcting for
        the provided confounds and optionally smoothed to supplied FWHM.
    """
    # use the brain mask directly from templateflow, so all subjects have
    # the same number of voxels.
    tpl_mask = './tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'

    if n_segments is not None:

        # We could easily see user error here; force a check
        assert n_segments <= len(bold_files)

        movie_segments = bold_files[:n_segments]
        regressor_segments = regressors[:n_segments]

    postproc_segments = []
    confounds = [_subset_confounds(r) for r in regressor_segments]

    for n, m in enumerate(movie_segments):
        postproc = _nifti_mask_movie(
            scan=m, mask=tpl_mask, smoothing_fwhm=fwhm,
            confounds=recfunctions.structured_to_unstructured(
                confounds[n])
        )
        postproc_segments.append(postproc.slicer[..., :seg_len[n]])

    postproc_fname = f'{s}/{s}_task-{t}_{tmpl}.nii.gz'.replace(
        'desc-preproc', f'desc-fwhm{fwhm}')
    movie = image.concat_imgs(postproc_segments)
    nib.save(movie, postproc_fname)
    return postproc_fname


if __name__ == "__main__":

    tasks, data_dictionary = create_data_dictionary()
    for s, t in itertools.product(subjects, tasks):

        seg_len, regr, tmpl = data_dictionary[t].values()

        files = Path(s).rglob(f'*{t}*{tmpl}.nii.gz')
        regressors = Path(s).rglob(f'*_task-{t}*{regr}')

        # get the sorted movie segments. check they're right w:
        # print(*segments, sep='\n')
        movie_segments = _get_segment([str(f) for f in files], t)
        regressor_segments = _get_segment([str(r) for r in regressors], t)

        subset_and_process_movie10(movie_segments, regressor_segments, seg_len,
                                   n_segments=5, fwhm=6)

        # this is slow, so keep track of subject, task
        print(s, t)
