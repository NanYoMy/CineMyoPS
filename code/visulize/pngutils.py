from tools.tps_painter import save_image_with_tps_points
from tools.np_sitk_tools import clipseScaleSitkImage, clipseScaleSArray
from skimage.util.compare import compare_images
from skimage.exposure import rescale_intensity
import cv2
from medpy.metric import dc
from tools.dir import mkdir_if_not_exist
import numpy as np
import SimpleITK as sitk
from skimage import segmentation, color
from PIL import Image
from tools.dir import mk_or_cleardir, mkdir_if_not_exist
from visulize.color import *
from tools.np_sitk_tools import reindex_label_array_by_dict
import os
from PIL import Image, ImageFont, ImageDraw
import seaborn as sns
from matplotlib import pyplot as plt


class SaveNumpy2Png():

    def __init__(self, base_dir='../outputs/'):
        self.invoke = 0
        self.base_dir = base_dir

    def get_target_size(self, array_size):
        w, h = array_size

        # new_h=500
        # new_w = int(500/h*w)
        # return (new_h,new_w)
        new_x = 500
        new_y = int(500 / h * w)
        return (new_x, new_y)

    def save_diff_img(self, array1, array2, dir, name, parma=None):
        mkdir_if_not_exist(dir)
        array1 = np.squeeze(array1).astype(np.float32)
        array2 = np.squeeze(array2).astype(np.float32)

        target_size = self.get_target_size(parma)

        array1 = cv2.resize(array1, target_size)
        array2 = cv2.resize(array2, target_size)
        diff = compare_images(rescale_intensity(array1, out_range=(0, 1)),
                              rescale_intensity(array2, out_range=(0, 1)),
                              method='checkerboard', n_tiles=(5, 5))
        diff = (diff * 255).astype(np.uint8)
        # diff=cv2.flip(diff,0)#调整视角
        cv2.imwrite(f"{dir}/{name}", diff)

    def save_diff_img(self, array1, array2, dir, name):
        mkdir_if_not_exist(dir)
        array1 = np.squeeze(array1).astype(np.float32)
        array2 = np.squeeze(array2).astype(np.float32)
        array1 = clipseScaleSArray(array1, 0, 100).astype('uint8')
        array2 = clipseScaleSArray(array2, 0, 100).astype('uint8')
        diff = compare_images(rescale_intensity(array1, out_range=(0, 1)),
                              rescale_intensity(array2, out_range=(0, 1)),
                              method='checkerboard', n_tiles=(2, 2))
        diff = (diff * 255).astype(np.uint8)
        # diff=cv2.flip(diff,0)#调整视角
        diff = Image.fromarray(diff).convert('RGB')
        diff.save(f"{dir}/{name}")
        # cv2.imwrite(,diff )

    #
    def save_img_with_mask(self, img, lab, dir, name, colors=None, mash_alpha=0.001, param=None):

        mkdir_if_not_exist(dir)
        img = np.squeeze(img).astype(np.float32)
        lab = np.squeeze(lab).astype(np.float32)

        target_size = self.get_target_size(param)

        img = cv2.resize(img, target_size)
        lab = cv2.resize(lab, target_size, interpolation=cv2.INTER_NEAREST)
        img = clipseScaleSArray(img, 0, 100).astype('uint8')
        img = color.label2rgb(lab.astype(np.uint8), img, colors=colors, alpha=mash_alpha,
                              bg_label=0, bg_color=None)
        # img = cv2.flip(img, 0)
        cv2.imwrite(f"{dir}/{name}", img * 255)

        return f"{dir}/{name}"

    def save_img_with_mask_withoutparma(self, img, lab, dir, name, colors=None, mash_alpha=0.001):

        mkdir_if_not_exist(dir)
        img = np.squeeze(img).astype(np.float32)
        lab = np.squeeze(lab).astype(np.float32)

        img = clipseScaleSArray(img, 0, 100).astype('uint8')
        img = color.label2rgb(lab.astype(np.uint8), img, colors=colors, alpha=mash_alpha,
                              bg_label=0, bg_color=None)
        # img = cv2.flip(img, 0)
        cv2.imwrite(f"{dir}/{name}", img * 255)
        return f"{dir}/{name}"

    def save_mask(self, lab, dir, name, colors=None, mash_alpha=0.001, param=None):
        mkdir_if_not_exist(dir)

        lab = np.squeeze(lab).astype(np.float32)

        target_size = self.get_target_size(param)

        lab = cv2.resize(lab, target_size, cv2.INTER_NEAREST)
        img = np.ones_like(lab)
        img = color.label2rgb(lab.astype(np.uint8), img, colors=colors, alpha=mash_alpha,
                              bg_label=0, bg_color=None)
        # img = cv2.flip(img, 0)
        cv2.imwrite(f"{dir}/{name}", img * 255)

    def save_img_with_contorus(self, img, lab, dir, name, color=[(1, 0, 0)], param=None):
        mkdir_if_not_exist(dir)

        img = np.squeeze(img).astype(np.float32)
        lab = np.squeeze(lab).astype(np.float32)
        target_size = self.get_target_size(param)
        img = cv2.resize(img, target_size)
        lab = cv2.resize(lab, target_size, interpolation=cv2.INTER_NEAREST)
        img = clipseScaleSArray(img, 0, 100).astype('uint8')
        img = segmentation.mark_boundaries(img, lab.astype(np.uint8), color=color, mode='thick')

        # img=sitk.GetImageFromArray(img)
        # img=clipseScaleSitkImage(img,0,100)
        # img=sitk.GetArrayFromImage(img).astype('uint8')
        # contours = extract_semantic_contour(lab)
        # img = draw_coutours(np.tile(np.expand_dims(img, -1), (1, 1, 3)), contours,
        #                     my_color)
        # img = cv2.flip(img, 0)
        cv2.imwrite(f"{dir}/{name}", img * 255)

    def save_image_with_pred_gt_contous(self, img, pred_lab, gt_lab, dir, name, colora=[(0, 0, 1)], colorb=[(1, 0, 0)],
                                        param=None):
        mkdir_if_not_exist(dir)
        img = np.squeeze(img).astype(np.float32)
        pred_lab = np.squeeze(pred_lab).astype(np.float32)
        gt_lab = np.squeeze(gt_lab).astype(np.float32)
        target_size = self.get_target_size(param)

        pred_lab = cv2.resize(pred_lab, target_size, interpolation=cv2.INTER_NEAREST)
        gt_lab = cv2.resize(gt_lab, target_size, interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, target_size)
        img = clipseScaleSArray(img, 0, 100).astype(np.uint8)
        img = segmentation.mark_boundaries(img, pred_lab.astype(np.uint8), color=colora, mode='thick')
        img = segmentation.mark_boundaries(img, gt_lab.astype(np.uint8), color=colorb, mode='thick')
        # img = cv2.flip(img, 0)
        cv2.imwrite(f"{dir}/{name}", img * 255)

    def save_img_with_tps(self, img, control_points, dir, name, param=None):
        mkdir_if_not_exist(dir)
        source_array = np.squeeze(img)
        control_points = (control_points.data[0])
        tmp = sitk.GetImageFromArray(source_array)
        tmp = clipseScaleSitkImage(tmp, 0, 100)
        source_array = sitk.GetArrayFromImage(tmp).astype('uint8')

        save_image_with_tps_points(control_points, source_array, dir, name, self.args.grid_size, 500, 0)

    def save_img(self, img, dir, name, param=None):
        mkdir_if_not_exist(dir)
        tmp = np.squeeze(img)
        # tmp = sitk.GetImageFromArray(source_array)
        # tmp=np.squeeze(tmp)
        # target_size = self.get_target_size(param)
        # tmp = cv2.resize(tmp.astype(np.float32), target_size)
        tmp = clipseScaleSArray(tmp, 0, 100).astype('uint8')
        cv2.imwrite(f"{dir}/{name}", tmp)
        return f"{dir}/{name}"
        # source_array = sitk.GetArrayFromImage(tmp).astype('uint8')
        # save_img(source_array,dir,name,img_size=500,border=0)

        # tmp = Image.fromarray(source_array).convert('RGB').resize((500, 500))
        # source_points = (source_points + 1) / 2 * 128 + 64
        # source_image = Image.fromarray(img).convert('RGB').resize((128, 128))
        # img = ImageOps.flip(img)  # 调整正常的视角
        # tmp.save(f"{dir}/{name}")

    def save_imgs(self, imgs):
        pathes = []
        mk_or_cleardir(f'{self.base_dir}/tmp_{self.invoke}/')
        for img in imgs:

            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img))


            # edema_array = reindex_label_array_by_dict(edema_array, {1: [1220, 2221]})

            path = self.save_img(img_array,  f'{self.base_dir}/tmp_{self.invoke}/',
                                                        f"{os.path.basename(img).split('.')[0]}.png")


            pathes.append(path)

        self.invoke = self.invoke + 1
        return pathes


    def merge_imgs(self, imgs, edemaLabs, scarLabs, color=[colorT2[0], colorDe[0]]):
        pathes = []

        mk_or_cleardir(f'../tmp_{self.invoke}/')
        for img, edemaLab, scarLab in zip(imgs, edemaLabs, scarLabs):
            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img))
            edema_array = sitk.GetArrayFromImage(sitk.ReadImage(edemaLab))
            scar_array = sitk.GetArrayFromImage(sitk.ReadImage(scarLab))
            edema_array = reindex_label_array_by_dict(edema_array, {1: [1220]})
            scar_array = reindex_label_array_by_dict(scar_array, {1: [2221]})
            total = np.where(scar_array == 1, 2, edema_array)

            self.save_img_with_mask_withoutparma(img_array, total, f'../tmp_{self.invoke}/',
                                                 f"{os.path.basename(img).split('.')[0]}.png",
                                                 colors=color, mash_alpha=1)
            pathes.append(f"../tmp_{self.invoke}/{os.path.basename(img).split('.')[0]}.png")
        self.invoke = self.invoke + 1
        return pathes

    def merge_image_with_pred_gt_contous(self, imgs, predLabs, gtLabs, color=[colorT2[0], colorDe[0]]):
        pathes = []
        mk_or_cleardir(f'../tmp_{self.invoke}/')
        for p_img, p_pred_lab, p_gt_lab in zip(imgs, predLabs, gtLabs):
            img = sitk.GetArrayFromImage(sitk.ReadImage(p_img))
            pred_lab = sitk.GetArrayFromImage(sitk.ReadImage(p_pred_lab))
            pred_lab = reindex_label_array_by_dict(pred_lab, {1: [200, 1220, 2221]})

            gt_lab = sitk.GetArrayFromImage(sitk.ReadImage(p_gt_lab))
            gt_lab = reindex_label_array_by_dict(gt_lab, {2: [200, 1220, 2221]})

            img = np.squeeze(img).astype(np.float32)
            pred_lab = np.squeeze(pred_lab).astype(np.float32)
            gt_lab = np.squeeze(gt_lab).astype(np.float32)

            # target_size = self.get_target_size(param)
            # pred_lab=cv2.resize(pred_lab, target_size, interpolation=cv2.INTER_NEAREST)
            # gt_lab=cv2.resize(gt_lab, target_size, interpolation=cv2.INTER_NEAREST)
            # img=cv2.resize(img, target_size)

            img = clipseScaleSArray(img, 0, 100).astype(np.uint8)
            img = segmentation.mark_boundaries(img, pred_lab.astype(np.uint8), color=color[0], mode='thick')
            img = segmentation.mark_boundaries(img, gt_lab.astype(np.uint8), color=color[1], mode='thick')
            # img = cv2.flip(img, 0)
            cv2.imwrite(f"../tmp_{self.invoke}/{os.path.basename(p_img).split('.')[0]}.png", img * 255)
            pathes.append(f"../tmp_{self.invoke}/{os.path.basename(p_img).split('.')[0]}.png")
        self.invoke = self.invoke + 1
        return pathes

    def merge_img_with_contorus(self, imgs, labs, color=[(1, 0, 0)]):
        pathes = []
        mk_or_cleardir(f'../tmp_{self.invoke}/')
        for p_img, p_lab in zip(imgs, labs):
            img = sitk.GetArrayFromImage(sitk.ReadImage(p_img))
            lab = sitk.GetArrayFromImage(sitk.ReadImage(p_lab))
            lab = reindex_label_array_by_dict(lab, {1: [200, 1220, 2221, 1]})

            img = np.squeeze(img).astype(np.float32)
            pred_lab = np.squeeze(lab).astype(np.float32)

            # target_size = self.get_target_size(param)
            # pred_lab=cv2.resize(pred_lab, target_size, interpolation=cv2.INTER_NEAREST)
            # gt_lab=cv2.resize(gt_lab, target_size, interpolation=cv2.INTER_NEAREST)
            # img=cv2.resize(img, target_size)

            img = clipseScaleSArray(img, 0, 100).astype(np.uint8)

            # ret, thresh = cv2.threshold(pred_lab.astype(np.uint8), 0.5, 1, 0)
            # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # img = cv2.drawContours(np.tile(np.expand_dims(img,-1),[1,1,3]), contours, -1, (0, 255, 0), 1)  # img为三通道才能显示轮廓
            # #
            # cv2.imshow('drawimg2', img)
            #
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            img = segmentation.mark_boundaries(img, pred_lab.astype(np.uint8), color=color[0], mode='thick')
            # img = cv2.flip(img, 0)
            cv2.imwrite(f"../tmp_{self.invoke}/{os.path.basename(p_img).split('.')[0]}.png", img * 255)
            pathes.append(f"../tmp_{self.invoke}/{os.path.basename(p_img).split('.')[0]}.png")
        self.invoke = self.invoke + 1
        return pathes

    def save_heatmap(self, imgs):
        mk_or_cleardir(f'../tmp_{self.invoke}/')
        paths = []
        for pimg in imgs:
            img = sitk.GetArrayFromImage(sitk.ReadImage(pimg))
            img = np.squeeze(img).astype(np.float32)
            w, h = img.shape
            figure = plt.gcf()
            # dpi=80
            # plt.subplot(dpi=dpi)
            # figure.set_size_inches(5, 5)
            # fig.set_size_inches(magic_height*w/(h*dpi), magic_height/dpi)
            plt.axis('off')
            sns.heatmap(img, ax=None, cmap='viridis', cbar=False)

            plt.savefig(f'../tmp_{self.invoke}/{os.path.basename(pimg).split(".")[0]}.png', dpi=400,
                        bbox_inches='tight', pad_inches=0)

            tmp = f'../tmp_{self.invoke}/{os.path.basename(pimg).split(".")[0]}.png'
            larg_img = cv2.imread(tmp)
            small_img = cv2.resize(larg_img, (img.shape[1], img.shape[0]))
            cv2.imwrite(tmp, small_img)
            plt.cla()
            plt.clf()
            plt.close('all')  # fix bug
            paths.append(f'../tmp_{self.invoke}/{os.path.basename(pimg).split(".")[0]}.png')
        self.invoke = self.invoke + 1
        return paths

    def add_text(self, p, text, size=26):
        img = Image.open(p)
        draw = ImageDraw.Draw(img)
        # font=ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 19)
        font = ImageFont.truetype("./resource/times.ttf", size)
        draw.text((0, 0), text, (0, 255, 0), font=font)
        # draw.text((0,0),text,(0,255,0))
        img.save(p)

    def cal_dc(self, gd, pre):
        gd_img = sitk.GetArrayFromImage(sitk.ReadImage(gd))
        pre_img = sitk.GetArrayFromImage(sitk.ReadImage(pre))
        gd = reindex_label_array_by_dict(gd_img, {1: [1220, 2221, 1, 2]})
        pre = reindex_label_array_by_dict(pre_img, {1: [1220, 2221, 1, 2]})
        return dc(gd, pre)

    def cal_scar_size(self, gd, pre, intestting):
        gd_img = sitk.GetArrayFromImage(sitk.ReadImage(gd))
        pre_img = sitk.GetArrayFromImage(sitk.ReadImage(pre))
        gd = reindex_label_array_by_dict(gd_img, {1: [1, 3, 5]})
        pre = reindex_label_array_by_dict(pre_img, intestting)
        return np.sum(pre) / np.sum(gd)

    def merge_two_imgs(self, imgs, edemaLabs, color=[colorT2[0], colorDe[0]]):
        pathes = []
        mk_or_cleardir(f'../tmp_{self.invoke}/')
        for img, edemaLab in zip(imgs, edemaLabs):
            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img))
            edema_array = sitk.GetArrayFromImage(sitk.ReadImage(edemaLab))
            labs = np.unique(edema_array)

            # edema_array = reindex_label_array_by_dict(edema_array, {1: [1220, 2221]})

            path = self.save_img_with_mask(img_array, edema_array, f'../tmp_{self.invoke}/',
                                           f"{os.path.basename(img).split('.')[0]}.png", colors=color, mash_alpha=1)

            pathes.append(path)

        self.invoke = self.invoke + 1
        return pathes

    def merge_two_imgs_with_text(self, imgs, predLabs, gdLabs, color=[colorT2[0], colorDe[0],colorC0[0],colorYellow[0],colorGD[0]], text=False,labs_reindex={1:[4],2:[3]}):
        pathes = []
        mk_or_cleardir(f'{self.base_dir}/tmp_{self.invoke}/')
        for img, preLab, gdLab in zip(imgs, predLabs, gdLabs):

            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img))
            edema_array = sitk.GetArrayFromImage(sitk.ReadImage(preLab))
            print(np.unique(edema_array))
            edema_array=reindex_label_array_by_dict(edema_array,labs_reindex)
            
            labs = np.unique(edema_array)

            # edema_array = reindex_label_array_by_dict(edema_array, {1: [1220, 2221]})

            path = self.save_img_with_mask_withoutparma(img_array, edema_array, f'{self.base_dir}/tmp_{self.invoke}/',
                                                        f"{os.path.basename(img).split('.')[0]}.png", colors=color,
                                                        mash_alpha=1)

            if text:
                dice = round(self.cal_dc(gdLab, preLab) * 100, 2)
                self.add_text(path, f"{dice}%")

            pathes.append(path)

        self.invoke = self.invoke + 1
        return pathes

    def merge_two_imgs_with_text_FIXED_ERROR(self, imgs, predLabs, gdLabs, color=[colorT2[0], colorDe[0]], text=False):
        pathes = []
        mk_or_cleardir(f'{self.base_dir}/tmp_{self.invoke}/')
        for img, preLab, gdLab in zip(imgs, predLabs, gdLabs):

            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img))
            edema_array = sitk.GetArrayFromImage(sitk.ReadImage(preLab))
            edema_array = np.transpose(edema_array, [0, 2, 1])
            dice = round(self.cal_dc(gdLab, preLab) * 100, 2)
            labs = np.unique(edema_array)

            # edema_array = reindex_label_array_by_dict(edema_array, {1: [1220, 2221]})

            path = self.save_img_with_mask_withoutparma(img_array, edema_array, f'{self.base_dir}/tmp_{self.invoke}/',
                                                        f"{os.path.basename(img).split('.')[0]}.png", colors=color,
                                                        mash_alpha=1)

            if text:
                self.add_text(path, f"{dice}%")

            pathes.append(path)

        self.invoke = self.invoke + 1
        return pathes

    def merge_two_imgs_with_text_by_idx(self, imgs, predLabs, gdLabs, color=[colorT2[0], colorDe[0]],
                                        intersting={1: [2221]}, text=False):
        pathes = []
        mk_or_cleardir(f'{self.base_dir}/tmp_{self.invoke}/')
        for img, preLab, gdLab in zip(imgs, predLabs, gdLabs):

            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img))

            pathology_array = sitk.GetArrayFromImage(sitk.ReadImage(preLab))
            pathology_array = reindex_label_array_by_dict(pathology_array, intersting)
            dice = round(self.cal_dc(gdLab, preLab) * 100, 2)
            labs = np.unique(pathology_array)

            # edema_array = reindex_label_array_by_dict(edema_array, {1: [1220, 2221]})

            path = self.save_img_with_mask_withoutparma(img_array, pathology_array,
                                                        f'{self.base_dir}/tmp_{self.invoke}/',
                                                        f"{os.path.basename(img).split('.')[0]}.png", colors=color,
                                                        mash_alpha=1)

            if text:
                self.add_text(path, f"{dice}%")

            pathes.append(path)

        self.invoke = self.invoke + 1
        return pathes

    def merge_two_imgs_with_scar_size_text_by_idx(self, imgs, predLabs, gdLabs, color=[colorT2[0], colorDe[0]],
                                                  intersting={1: [2221]}, text=True):
        pathes = []
        mk_or_cleardir(f'{self.base_dir}/tmp_{self.invoke}/')

        for img, preLab, gdLab in zip(imgs, predLabs, gdLabs):

            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img))

            pathology_array = sitk.GetArrayFromImage(sitk.ReadImage(preLab))
            pathology_array = reindex_label_array_by_dict(pathology_array, intersting)
            dice = round(self.cal_scar_size(gdLab, preLab, intersting) * 100, 2)
            labs = np.unique(pathology_array)

            # edema_array = reindex_label_array_by_dict(edema_array, {1: [1220, 2221]})

            path = self.save_img_with_mask_withoutparma(img_array, pathology_array,
                                                        f'{self.base_dir}/tmp_{self.invoke}/',
                                                        f"{os.path.basename(img).split('.')[0]}.png", colors=color,
                                                        mash_alpha=1)

            if text:
                self.add_text(path, f"{dice}%")

            pathes.append(path)

        self.invoke = self.invoke + 1
        return pathes

    def compared_two_imgs(self, imgs, mv_imgs):
        pathes = []

        mk_or_cleardir(f'../tmp_{self.invoke}/')
        for img, mv_img in zip(imgs, mv_imgs):
            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img))
            mv_img_array = sitk.GetArrayFromImage(sitk.ReadImage(mv_img))
            self.save_diff_img(img_array, mv_img_array, f'../tmp_{self.invoke}/',
                               f"{os.path.basename(img).split('.')[0]}.png")
            pathes.append(f"../tmp_{self.invoke}/{os.path.basename(img).split('.')[0]}.png")
        self.invoke = self.invoke + 1
        return pathes

    def merge_lab(self, awsnet_de_scars, awsnet_de_edemas):
        pathes = []
        from tools.itkdatawriter import sitk_write_array_as_nii
        for awsnet_de_scar, awsnet_de_edema in zip(awsnet_de_scars, awsnet_de_edemas):
            scare = sitk.ReadImage(awsnet_de_scar)
            scare_array = sitk.GetArrayFromImage(sitk.ReadImage(awsnet_de_scar))
            edema = sitk.ReadImage(awsnet_de_edema)
            edema_arry = sitk.GetArrayFromImage(edema)
            new_arryA = np.where(edema_arry == 2, 1, 0)
            new_arryB = np.where(scare_array == 2, 2, 0)
            out = new_arryA + new_arryB
            out = np.where(out == 3, 2, out)
            sitk_write_array_as_nii(out, scare, dir=f'../tmp_{self.invoke}',
                                    name=f"{os.path.basename(awsnet_de_scar).split('.')[0]}.nii.gz")
            pathes.append(f"../tmp_{self.invoke}/{os.path.basename(awsnet_de_scar).split('.')[0]}.nii.gz")
        return pathes




