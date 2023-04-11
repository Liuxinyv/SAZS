r""" Visualize model predictions """
import os

from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from . import utils


class Visualizer:

    @classmethod
    def initialize(cls, visualize):
        cls.visualize = visualize
        if not visualize:
            return

        cls.colors = {'red': (255, 50, 50), 'blue': (102, 140, 255)}
        for key, value in cls.colors.items():
            cls.colors[key] = tuple([c / 255 for c in cls.colors[key]])

        cls.mean_img = [0.485, 0.456, 0.406]
        cls.std_img = [0.229, 0.224, 0.225]
        cls.to_pil = transforms.ToPILImage()


    @classmethod

    def visualize_prediction_batch(cls, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b, batch_idx, name,vis_path,iou_b=None,fb_ious=None):
        # spt_img_b = utils.to_cpu(spt_img_b)
        # spt_mask_b = utils.to_cpu(spt_mask_b)
        # qry_img_b = utils.to_cpu(qry_img_b)#1,3,375,500
        # qry_mask_b = utils.to_cpu(qry_mask_b)#1,375,500
        # iou_v = utils.to_cpu(iou_b[-1])
        # cls_id_b = utils.to_cpu(cls_id_b)

        # new_vis_path = os.path.join(vis_path[0]+name[0][0][21:-4]+'_'+'%.2f' % (iou_b[-1].item())+'/')
        for sample_idx, (qry_img, qry_mask, pred_mask, cls_id) in \
                enumerate(zip(qry_img_b, qry_mask_b, pred_mask_b, cls_id_b)):
            qry_img = utils.to_cpu(qry_img)#1,3,375,500
            qry_mask = utils.to_cpu(qry_mask)#1,375,500
            # pred_mask = utils.to_cpu(pred_mask)
            cls_id = utils.to_cpu(cls_id)
            iou = iou_b[sample_idx] if iou_b is not None else None
            fb_iou = fb_ious[sample_idx] if fb_ious is not None else None
            print(len(qry_mask_b))
            if sample_idx==len(qry_mask_b)-1:
                cls.visualize_prediction(qry_img, qry_mask, pred_mask, cls_id, batch_idx, name[sample_idx],
                                         vis_path[sample_idx], True, iou, fb_iou,True)
            else:
                cls.visualize_prediction(qry_img, qry_mask, pred_mask, cls_id, batch_idx, name[sample_idx],vis_path[sample_idx], True, iou,fb_iou,False)
            # cls.visualize_prediction(qry_img, qry_mask, pred_mask, cls_id, batch_idx, name, vis_path, True)
        # for sample_idx, (spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id) in \
        #         enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b)):
        #     iou = iou_b[sample_idx] if iou_b is not None else None
        #     cls.visualize_prediction(spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sample_idx, True, iou)

    @classmethod
    def to_numpy(cls, tensor, type):
        if type == 'img':
            return np.array(cls.to_pil(cls.unnormalize(tensor))).astype(np.uint8)
        elif type == 'mask':
            return np.array(tensor).astype(np.uint8)
        else:
            raise Exception('Undefined tensor type: %s' % type)

    @classmethod
    # def visualize_prediction(cls, spt_imgs, spt_masks, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sample_idx, label, iou=None):

    # cls.visualize_prediction(qry_img, qry_mask, pred_mask, cls_id, batch_idx, name, vis_path, True, iou, fb_iou)
    def visualize_prediction(cls, qry_img, qry_mask, pred_mask, cls_id, batch_idx, name,vis_path,
                             label, iou=None,fb_iou=None,seen=False):

        spt_color = cls.colors['blue']
        # qry_color = cls.colors['red']
        qry_color = cls.colors['blue']
        pred_color = cls.colors['red']

        # # spt_imgs = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        # spt_pils = [cls.to_pil(spt_img) for spt_img in spt_imgs]
        # spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        # spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img, spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs, spt_masks)]
        qry_img = cls.to_numpy(qry_img.squeeze(), 'img')
        qry_mask = cls.to_numpy(qry_mask.squeeze(), 'mask')
        pred_mask = cls.to_numpy(pred_mask.squeeze(), 'mask')
        pred_masked_pil = Image.fromarray(
            cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))
        # pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        # qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))

        # merged_pil = cls.merge_image_pair(spt_masked_pils + [pred_masked_pil, qry_masked_pil])
        merged_pil = cls.merge_image_pair([pred_masked_pil, qry_masked_pil])

        iou = iou.item() if iou else 0.0
        # merged_pil.save(cls.vis_path + '%d_%d_class-%d_iou-%.2f' % (batch_idx, sample_idx, cls_id, iou) + '.jpg')
        # if seen is True:
        #     cls.vis_path = os.path.join(vis_path +
        #                                 '%s_^_iou-%.2f-class-%d_fb_iou-%.2f' % (
        #                                 name[0][21:-4], iou, cls_id, fb_iou) + '.jpg')
        # else:
        #     cls.vis_path = os.path.join(vis_path+
        #                                 '%s_iou-%.2f-class-%d_fb_iou-%.2f' % (name[0][21:-4],iou,cls_id, fb_iou) + '.jpg')
        # if not os.path.exists(vis_path): os.makedirs(vis_path)
        if seen is True:
            # cls.vis_path = os.path.join(vis_path +
            #                             'class-%d_%s_^_iou-%.2f_fb_iou-%.2f' % (cls_id, name[0], iou, fb_iou) + '.jpg')
            cls.vis_path = os.path.join(vis_path +
                                        'class-%d_^_iou-%.2f_fb_iou-%.2f_%s' % (cls_id,iou,fb_iou, name[0]) + '.jpg')
        else:
            cls.vis_path = os.path.join(vis_path +
                                        'class-%d_iou-%.2f_fb_iou-%.2f_%s' % (cls_id,iou, fb_iou,name[0]) + '.jpg')
        if not os.path.exists(vis_path): os.makedirs(vis_path)
        merged_pil.save(cls.vis_path)
        # merged_pil.save(
        #     cls.vis_path, 'class-%d_iou-%.2f_fb_iou-%.2f' % (cls_id, iou, fb_iou) + '.jpg')
        # merged_pil.save(
        #     cls.vis_path + 'class-%.2f-%s' % (cls_id, name[0][8:-4]) + '.jpg')

    @classmethod
    def merge_image_pair(cls, pil_imgs):
        r""" Horizontally aligns a pair of pytorch tensor images (3, H, W) and returns PIL object """

        canvas_width = sum([pil.size[0] for pil in pil_imgs])
        canvas_height = max([pil.size[1] for pil in pil_imgs])
        canvas = Image.new('RGB', (canvas_width, canvas_height))

        xpos = 0
        for pil in pil_imgs:
            canvas.paste(pil, (xpos, 0))
            xpos += pil.size[0]

        return canvas

    @classmethod
    def apply_mask(cls, image, mask, color, alpha=0.5):
    # def apply_mask(cls, image, mask, color, alpha=0):
        r""" Apply mask to the given image. """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
            # image[:, :, c] = np.where(mask == 1,
            #                           image[:, :, c] *
            #                           (1 - alpha) + alpha * color[c] * 255,
            #                           image[:, :, c])
        return image

    @classmethod
    def unnormalize(cls, img):
        img = img.clone()
        for im_channel, mean, std in zip(img, cls.mean_img, cls.std_img):
            im_channel.mul_(std).add_(mean)
        return img
