import os
import os.path as osp
from collections import OrderedDict
import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.core import eval_map, eval_recalls
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines.transforms import *
from mmdet.datasets.pipelines.loading import LoadImageFromFile, LoadAnnotations
from mmdet.datasets.pipelines.formating import DefaultFormatBundle

class Transformations(object):
    """
    Loads images and annotations from the disk (LoadImageFromFile and LoadAnnotations)and applies various transformations to the input image.
    Different transformations are applied to both train and test sets.
    such as PhotoMetricDistortion, which augments input image using:
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    DefaultFormatBundle - converts images, bounding boxes, and labels to tensors

     Args:
        test_mode (bool): Whether the dataset is training or testing (validation) dataset.

    """

    def __init__(self, test_mode=False):
        if not test_mode:
            self.transforms = [LoadImageFromFile(), LoadAnnotations(),
                               Resize(img_scale=(224,224), keep_ratio=False), RandomShift(),
                               PhotoMetricDistortion(),
                               RandomFlip(flip_ratio=0.5), Normalize(mean=[0.5,0.5, 0.5], std=[0.5, 0.5, 0.5]),
                               DefaultFormatBundle()]
        else:
            self.transforms = [LoadImageFromFile(), LoadAnnotations(),
                               Resize(img_scale=(224,224), keep_ratio=False),
                               RandomFlip(flip_ratio=0.5), Normalize(mean=[0.5,0.5, 0.5], std=[0.5, 0.5, 0.5]),
                               DefaultFormatBundle()]
    def __call__(self, data):
        """Call function to apply transforms sequentially.
        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for transform in self.transforms:
            data = transform(data)
            if data is None:
                return None
        return data

class ObjectDetectionVechicle:
    """Custom dataset for detection. Takes input from the configuration file regarding the 
    dataset paths. Extracts image paths and stores them to be loaded. Also extracts bounding boxes
    and labels to be stored. All three are stored in dict.

    The implementation logic is referred to
    https://github.com/open-mmlab/mmdetection/blob/v2.18.0/mmdet/datasets/custom.py

    Args:
        ann_file (str): Annotation file path.
        data_root (str, optional): Data root for ``ann_file``, ``img_prefix`` if specified.
        img_prefix (str, optional): folder name in which images are stored
        test_mode (bool, optional): If set True, annotation will not be loaded.

    """
    CLASSES = ("car", "truck", "van", "longvechicle", "bus", "airliner", "propeller",
                        "trainer", "chartered", "fighter", "others", "stairtruck",
                        "pushbacktruck", "helicopter", "boat")

    def __init__(self,
                 ann_file,
                 data_root=None,
                 img_prefix='',
                 test_mode=False,
                 file_client_args=dict(backend='disk')):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.CLASSES = ("car", "truck", "van", "longvechicle", "bus", "airliner", "propeller",
                        "trainer", "chartered", "fighter", "others", "stairtruck",
                        "pushbacktruck", "helicopter", "boat")
        self.file_client = mmcv.FileClient(**file_client_args)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            # if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
            #     self.seg_prefix = osp.join(self.data_root, self.seg_prefix)

        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations()
        else:
            self.data_infos = self.load_annotations(self.ann_file)

        self.transform = Transformations(test_mode=test_mode)
        self._set_group_flag()

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def __getitem__(self, idx):
        """Get training/test data after transformations.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (without annotation if `test_mode` is set true).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def load_annotations(self):
        """
        Loads annotation file, and stores image path along with bounding boxes and labels in a dict.
        Returns:
            dict: Annotation info.
        """
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        image_list = mmcv.list_from_file(self.ann_file)
    
        data_infos = []
        # convert annotations to middle format
        for complete_path in image_list:
            image_id = os.path.split(complete_path)[-1]
            filename = complete_path
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
    
            data_info = dict(filename=image_id, width=width, height=height)
    
            # load annotations
            label_prefix = self.img_prefix.replace('images', 'labels')
            lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id[:-4]}.txt'))
    
            content = [line.strip().split(' ') for line in lines]
            bbox_names = [x[0] for x in content]
            bboxes = [[float(info) for info in x[1:]] for x in content]
    
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
    
            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                # else:
                #     print(bbox_name)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                    dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['ann']

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()

    def pre_transform(self, results):
        """Prepare results dict for transformations."""
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []

    def prepare_train_img(self, idx):
        """Get training data and annotations after transformations.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after transformations with new keys \
                introduced by transformations.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_transform(results)
        return self.transform(results)

    def prepare_test_img(self, idx):
        """Get testing data  after transformations.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after transformations with new keys introduced by \
                transformations.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_transform(results)
        return self.transform(results)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

from custom_dataloader import get_dataloader
cfg = 'Config File'

dataset = ObjectDetectionVechicle(ann_file=cfg.data.train.ann_file, data_root=cfg.data.train.data_root, img_prefix=cfg.data.train.img_prefix)
dataloader = get_dataloader(dataset=dataset, batch_size=8)