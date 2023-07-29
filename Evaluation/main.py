import os.path as osp
from Evaluation.evaluator import Eval_thread
from Evaluation.dataloader import EvalDataset
from option import get_option
opt = get_option()
def evaluate():

    pred_dir = opt.test_save_path
    output_dir = opt.save_path
    gt_dir = opt.dataset_root

    threads = []
    test_paths = opt.test_paths.split('+')
    for dataset_setname in test_paths:

        dataset_name = dataset_setname.split('/')[0]



        pred_dir_all = osp.join(pred_dir, dataset_setname)

        if dataset_name == 'DUTS':
            gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname)) + '/DUTS-TE-Mask'
        elif dataset_name == 'ECSSD' or dataset_name == 'DUT-OMRON'or dataset_name == 'cod' or dataset_name=="HKU-IS":
            gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname)) + '/GT'
        else:
            gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname)) + '/images'

        loader = EvalDataset(pred_dir_all, gt_dir_all)
        thread = Eval_thread(loader, dataset_setname, output_dir, cuda=True)
        threads.append(thread)
    for thread in threads:
        print(thread.run())
evaluate()