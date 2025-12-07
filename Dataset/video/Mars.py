from __future__ import print_function, absolute_import

import logging
import os.path as osp
from scipy.io import loadmat
import numpy as np


class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """

    def __init__(self, root, sampling_step=64, seq_len=16, **kwargs):
        self.logger = logging.getLogger("ReIDAdapter.dataset")
        self.root = osp.join(root, 'Mars')
        self.train_name_path = osp.join(self.root, 'info/train_name.txt')
        self.test_name_path = osp.join(self.root, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self.root, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self.root, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self.root, 'info/query_IDX.mat')
        self._check_before_run()
        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        # self.attributes = pd.read_csv(osp.join(self.root, "mars_attr.csv"), encoding="gbk")
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]
        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True)
        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False)

        recombined_query, query_vid2clip_index = self._recombination_for_testset(query, seq_len=seq_len)
        recombined_gallery, gallery_vid2clip_index = self._recombination_for_testset(gallery, seq_len=seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        self.logger.info("=> MARS loaded")
        self.logger.info("Dataset statistics:")
        self.logger.info("  ------------------------------")
        self.logger.info("  subset   | # ids | # tracklets")
        self.logger.info("  ------------------------------")
        self.logger.info("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        self.logger.info("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        self.logger.info("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        self.logger.info("  ------------------------------")
        self.logger.info("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        self.logger.info("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        self.logger.info("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        self.recombined_query = recombined_query
        self.recombined_gallery = recombined_gallery
        self.query_vid2clip_index = query_vid2clip_index
        self.gallery_vid2clip_index = gallery_vid2clip_index

        self.num_train_pids, self.num_train_imgs, self.num_train_cams  = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        self.num_train_vids = 0                 # mars 没有

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, ):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            img_names = names[start_index-1:end_index]
            assert 1 <= camid <= 6

            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera*
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

    def _recombination_for_testset(self, dataset, seq_len=8):
        ''' Split all videos in test set into lots of equilong clips.
        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride
        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        # stride = len(dataset) // seq_len
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, (img_paths, pid, camid) in enumerate(dataset):
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            stride = len(img_paths) // seq_len
            for j in range(stride):
                begin_idx = j
                end_idx = seq_len * stride + j
                clip_paths = img_paths[begin_idx: end_idx: stride]
                assert (len(clip_paths) == seq_len)
                new_dataset.append((clip_paths, pid, camid))
            # process the remaining sequence that can't be divisible by seq_len*stride
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert ((vid2clip_index[idx, 1] - vid2clip_index[idx, 0]) == len(img_paths) // seq_len)

        for i in range(len(new_dataset)):
            id = int(new_dataset[i][0][0].split('/')[6])
            if id != new_dataset[i][1]:
                raise RuntimeError("The error happened '{}'".format(new_dataset[i]))

        return new_dataset, vid2clip_index.tolist()

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams,