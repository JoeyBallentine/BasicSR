import argparse
import logging
import os

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from dataops.common import bgr2ycbcr, tensor2np
from models import create_model


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
    opt = option.dict_to_nonedict(opt)

    # logger
    util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # Create test dataset and dataloader
    test_loaders = []
    znorm = False  # TMP
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:,d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)
        # Temporary, will turn znorm on for all the datasets. Will need to introduce a variable for each dataset
        # and differentiate each one later in the loop.
        if dataset_opt['znorm']:
            znorm = True

    # Create model
    model = create_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        need_hr = False
        test_results = {"psnr": [], "ssim": [], "psnr_y": [], "ssim_y": []}

        for data in test_loader:
            need_hr = test_loader.dataset.opt['dataroot_HR'] is not None

            model.feed_data(data, need_HR=need_hr)
            img_path = data['in_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            model.test()  # test
            visuals = model.get_current_visuals(need_HR=need_hr)

            # if znorm the image range is [-1,1], Default: Image range is [0,1]
            # testing, each "dataset" can have a different name (not train, val or other)
            top_img = tensor2np(visuals['top_fake'])  # uint8
            bot_img = tensor2np(visuals['bottom_fake'])  # uint8

            # save images
            save_img_path = os.path.join(dataset_dir, img_name + (opt['suffix'] if opt['suffix'] else ''))
            util.save_img(top_img, '{:s}_top.png'.format(save_img_path))
            util.save_img(bot_img, '{:s}_bot.png'.format(save_img_path))

            # TODO: update to use metrics functions
            # calculate PSNR and SSIM
            if need_hr:
                # if znorm the image range is [-1,1], Default: Image range is [0,1]
                # testing, each "dataset" can have a different name (not train, val or other)
                gt_img = tensor2np(visuals['HR'], denormalize=znorm)  # uint8
                gt_img = gt_img / 255.
                sr_img = sr_img / 255.

                crop_border = test_loader.dataset.opt['scale']
                cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

                psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                log_msg = '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}'.format(img_name, psnr, ssim)
                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                    cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                    psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                    log_msg += '; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}'.format(psnr_y, ssim_y)
                logger.info(log_msg)
            else:
                logger.info(img_name)

        # TODO: update to use metrics functions
        if need_hr:  # metrics
            # averages
            logger.info(
                '----Average PSNR/SSIM results for {}----\n'.format(test_set_name) +
                '\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
                    sum(test_results['psnr']) / len(test_results['psnr']),
                    sum(test_results['ssim']) / len(test_results['ssim'])
                )
            )
            if test_results['psnr_y'] and test_results['ssim_y']:
                logger.info(
                    '----Y channel, average PSNR/SSIM----\n'
                    '\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.format(
                        sum(test_results['psnr_y']) / len(test_results['psnr_y']),
                        sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                    )
                )


if __name__ == '__main__':
    main()
