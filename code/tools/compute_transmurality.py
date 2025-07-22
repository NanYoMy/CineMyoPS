import SimpleITK as sitk
import cv2
import numpy as np
from skimage import draw
from skimage.morphology import disk, erosion, binary_dilation, skeletonize,binary_erosion

from tools.np_sitk_tools import reindex_label_array_by_dict


def make_sector(image, center=[70, 100], R=100, theta0= 20, theta1= 40):
    r0, c0 = center[0],center[1] # circle center (row, column)
    # R = 100  # circle radius
    #
    theta0 = np.deg2rad(theta0) # angle #1 for arc
    theta1 = np.deg2rad(theta1)  # angle #2 for arc

    # Above, I provide two angles, but you can also just give the two
    # coordinates below directly

    r1, c1 = r0 - 1.5 * R * np.sin(theta0), c0 + 1.5 * R * np.cos(theta0)  # arc coord #1
    r2, c2 = r0 - 1.5 * R * np.sin(theta1), c0 + 1.5 * R * np.cos(theta1)  # arc coord #2

    # --- mask calculation

    mask_circle = np.zeros(image.shape[:2], dtype=bool)
    mask_poly = np.zeros(image.shape[:2], dtype=bool)

    rr, cc = draw.ellipse(r0, c0, R, R, shape=mask_circle.shape)
    mask_circle[rr, cc] = 1

    rr, cc = draw.polygon([r0, r1, r2, r0],
                          [c0, c1, c2, c0], shape=mask_poly.shape)



    mask_poly[rr, cc] = 1

    mask = mask_circle & mask_poly

    return mask


def make_line(image, center=[70, 100], R=100, theta = 20):
    print(theta)
    r0, c0 = center[0], center[1]  # circle center (row, column)
    # R = 100  # circle radius
    #
    theta0 = np.deg2rad(theta - 3)  # angle #1 for arc
    theta1 = np.deg2rad(theta + 3)  # angle #2 for arc
    thetaline = np.deg2rad(theta)  # angle #2 for arc

    # Above, I provide two angles, but you can also just give the two
    # coordinates below directly

    r1, c1 = r0 - 1.5 * R * np.sin(theta0), c0 + 1.5 * R * np.cos(theta0)  # arc coord #1
    r2, c2 = r0 - 1.5 * R * np.sin(theta1), c0 + 1.5 * R * np.cos(theta1)  # arc coord #2
    l_r, l_c = r0 - 1.5 * R * np.sin(thetaline), c0 + 1.5 * R * np.cos(thetaline)  # arc coord #2

    # --- mask calculation

    mask_circle = np.zeros(image.shape[:2], dtype=bool)
    mask_line = np.zeros(image.shape[:2], dtype=bool)
    mask_poly = np.zeros(image.shape[:2], dtype=bool)

    rr, cc = draw.circle(r0, c0, R, shape=mask_circle.shape)
    mask_circle[rr, cc] = 1

    rr, cc = draw.polygon([r0, r1, r2, r0],
                          [c0, c1, c2, c0], shape=mask_poly.shape)

    mask_poly[rr, cc] = 1



    rr, cc,val = draw.line_aa(int(r0), int(c0),int(l_r), int(l_c))

    new_rr = []
    new_cc = []
    for i, j in zip(rr, cc):
        if i >= mask_circle.shape[0] or j >= mask_circle.shape[1] or i <0 or j<0:
            continue
        else:
            new_rr.append(i)
            new_cc.append(j)

    mask_line[new_rr, new_cc] = 1

    mask=mask_line&mask_circle&mask_poly


    return mask

#back up
def calculate_transmularity(predict_path, reference, show_detail=False, intersting={1:[2221]}):



    pred_arr = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(predict_path)))
    img=sitk.ReadImage(reference)
    ref_arr=sitk.GetArrayFromImage(img)
    ref_arr=np.squeeze(ref_arr)
    # img_arr=np.flipud(img_arr)
    print(f"{predict_path} {np.unique(pred_arr)}")

    # img_arr=np.flip(img_arr,[0])
    # img_arr=np.transpose(img_arr,[1,0])

    # blobs = data.binary_blobs(200, blob_size_fraction=.2,
    #                           volume_fraction=.35, seed=1)
    myo=reindex_label_array_by_dict(ref_arr,{1:[1,5,4]})
    footprint = disk(1)
    ero_myo=binary_erosion(myo,footprint)
    dali_myo=binary_dilation(ero_myo)
    skeleton_lee = skeletonize(dali_myo.astype(np.uint8), method='lee')
    index=np.argwhere(skeleton_lee==1)
    # index=np.argwhere(myo==1)

    sector=np.zeros_like(skeleton_lee)

    center=np.mean(index,axis=0)
    # myo[int(center[0]),int(center[1])]=1

    transmularity=[]
    for i in range(0,3600,36):
        sector=make_sector(sector, center, np.max(skeleton_lee.shape), i / 10, (i + 36) / 10)
        # sector=make_line(sector, center, np.max(skeleton_lee.shape), i / 10)
        myo_blobs=sector*myo
        myo_blobs=np.where(myo_blobs>0,1,0)

        blobs_scar=sector*pred_arr
        blobs_scar=reindex_label_array_by_dict(blobs_scar,intersting)
        # myo_blobs=np.where(myo_blobs>0,1,0)
        if np.sum(myo_blobs)==0:
            ratio=0
        else:
            ratio=np.sum(blobs_scar)/np.sum(myo_blobs)

        if ratio >=0 and ratio<0.25:
            tmp=0
        elif ratio>=0.25 and ratio<0.5:
            tmp=1
        elif ratio>=0.5 and ratio<0.75:
            tmp=2
        else:
            tmp=3
        # print(f"{tmp} {ratio}={np.sum(blobs_scar)}/{np.sum(myo_blobs)}")
        # transmularity = [i / 4 + 0.001 for i in transmularity]
        transmularity.append(ratio)
        # transmularity.append(ratio)

        if show_detail:
            cv2.imshow('IMG',ref_arr.astype(np.uint8))
            cv2.imshow('blobs_scar',blobs_scar.astype(np.uint8)*255)
            cv2.imshow('pred_arr',pred_arr.astype(np.uint8)*120)
            cv2.imshow('blobs_total',myo_blobs.astype(np.uint8)*255)
            blobs=reindex_label_array_by_dict(ref_arr,{255:[2221],100:[200]})
            cv2.imshow('Result',cv2.applyColorMap(blobs.astype(np.uint8),3))
            cv2.waitKey()
    # print(transmularity)

    transmularity.reverse()
    return transmularity
    # plt.imshow(blobs)
    # print(f"{i}")
    # plt.show()
    # plt.waitforbuttonpress()
    # plt.des

def calculate_transmularity_ratio(predict_path, reference, show_detail=False, intersting={1:[2221]}):



    pred_arr = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(predict_path)))
    img=sitk.ReadImage(reference)
    ref_arr=sitk.GetArrayFromImage(img)
    ref_arr=np.squeeze(ref_arr)
    # img_arr=np.flipud(img_arr)
    print(f"{predict_path} {np.unique(pred_arr)}")

    # img_arr=np.flip(img_arr,[0])
    # img_arr=np.transpose(img_arr,[1,0])

    # blobs = data.binary_blobs(200, blob_size_fraction=.2,
    #                           volume_fraction=.35, seed=1)
    myo=reindex_label_array_by_dict(ref_arr,{1:[200,1220,2221]})
    footprint = disk(1)
    ero_myo=binary_erosion(myo,footprint)
    dali_myo=binary_dilation(ero_myo)
    skeleton_lee = skeletonize(dali_myo.astype(np.uint8), method='lee')
    index=np.argwhere(skeleton_lee==1)
    # index=np.argwhere(myo==1)

    sector=np.zeros_like(skeleton_lee)

    center=np.mean(index,axis=0)
    # myo[int(center[0]),int(center[1])]=1

    transmularity=[]
    for i in range(0,3600,36):
        sector=make_sector(sector, center, np.max(skeleton_lee.shape), i / 10, (i + 36) / 10)
        # sector=make_line(sector, center, np.max(skeleton_lee.shape), i / 10)
        myo_blobs=sector*myo
        myo_blobs=np.where(myo_blobs>0,1,0)

        blobs_scar=pred_arr*myo_blobs
        blobs_scar=reindex_label_array_by_dict(blobs_scar,intersting)
        # myo_blobs=np.where(myo_blobs>0,1,0)
        if np.sum(myo_blobs)==0:
            ratio=0
        else:
            ratio=np.sum(blobs_scar)/np.sum(myo_blobs)

        if ratio<0.01:
            ratio=0

        # print(f"{tmp} {ratio}={np.sum(blobs_scar)}/{np.sum(myo_blobs)}")
        # transmularity = [i / 4 + 0.001 for i in transmularity]
        transmularity.append(ratio)
        # transmularity.append(ratio)

        if show_detail:
            cv2.imshow('IMG',ref_arr.astype(np.uint8))
            cv2.imshow('blobs_scar',blobs_scar.astype(np.uint8)*255)
            cv2.imshow('pred_arr',pred_arr.astype(np.uint8)*120)
            cv2.imshow('blobs_total',myo_blobs.astype(np.uint8)*255)
            blobs=reindex_label_array_by_dict(ref_arr,{255:[2221],100:[200]})
            cv2.imshow('Result',cv2.applyColorMap(blobs.astype(np.uint8),3))
            cv2.waitKey()
    # print(transmularity)

    transmularity.reverse()
    return transmularity
    # plt.imshow(blobs)
    # print(f"{i}")
    # plt.show()
    # plt.waitforbuttonpress()
    # plt.des

def calculate_transmularity_class(predict_path, reference, show_detail=False, intersting={1:[2221]}):



    pred_arr = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(predict_path)))
    img=sitk.ReadImage(reference)
    ref_arr=sitk.GetArrayFromImage(img)
    ref_arr=np.squeeze(ref_arr)
    # img_arr=np.flipud(img_arr)
    print(f"{predict_path} {np.unique(pred_arr)}")

    # img_arr=np.flip(img_arr,[0])
    # img_arr=np.transpose(img_arr,[1,0])

    # blobs = data.binary_blobs(200, blob_size_fraction=.2,
    #                           volume_fraction=.35, seed=1)
    myo=reindex_label_array_by_dict(ref_arr,{1:[1,4,5]})
    footprint = disk(1)
    ero_myo=binary_erosion(myo,footprint)
    dali_myo=binary_dilation(ero_myo)
    skeleton_lee = skeletonize(dali_myo.astype(np.uint8), method='lee')
    index=np.argwhere(skeleton_lee==1)
    # index=np.argwhere(myo==1)

    sector=np.zeros_like(skeleton_lee)

    center=np.mean(index,axis=0)
    # myo[int(center[0]),int(center[1])]=1

    transmularity=[]
    for i in range(0,3600,36):
        sector=make_sector(sector, center, np.max(skeleton_lee.shape), i / 10, (i + 36) / 10)
        # sector=make_line(sector, center, np.max(skeleton_lee.shape), i / 10)
        myo_blobs=sector*myo
        myo_blobs=np.where(myo_blobs>0,1,0)

        blobs_scar=sector*pred_arr
        blobs_scar=reindex_label_array_by_dict(blobs_scar,intersting)
        # myo_blobs=np.where(myo_blobs>0,1,0)
        if np.sum(myo_blobs)==0:
            ratio=0
        else:
            ratio=np.sum(blobs_scar)/np.sum(myo_blobs)

        if ratio<0.01:
            tmp=0
        elif ratio >= 0.01 and ratio < 0.25:
            tmp = 1
        elif ratio >= 0.25 and ratio < 0.5:
            tmp = 2
        elif ratio >= 0.5 and ratio < 0.75:
            tmp = 3
        else:
            tmp = 4


        # print(f"{tmp} {ratio}={np.sum(blobs_scar)}/{np.sum(myo_blobs)}")
        # transmularity = [i / 4 + 0.001 for i in transmularity]
        transmularity.append(tmp)

        # print(f"{tmp} {ratio}={np.sum(blobs_scar)}/{np.sum(myo_blobs)}")
        # transmularity = [i / 4 + 0.001 for i in transmularity]
        # transmularity.append(ratio)
        # transmularity.append(ratio)

        if show_detail:
            cv2.imshow('IMG',ref_arr.astype(np.uint8))
            cv2.imshow('blobs_scar',blobs_scar.astype(np.uint8)*255)
            cv2.imshow('pred_arr',pred_arr.astype(np.uint8)*120)
            cv2.imshow('blobs_total',myo_blobs.astype(np.uint8)*255)
            blobs=reindex_label_array_by_dict(ref_arr,{255:[2221],100:[200]})
            cv2.imshow('Result',cv2.applyColorMap(blobs.astype(np.uint8),3))
            cv2.waitKey()
    # print(transmularity)

    transmularity.reverse()
    return transmularity
    # plt.imshow(blobs)
    # print(f"{i}")
    # plt.show()
    # plt.waitforbuttonpress()
    # plt.des




from tools.np_sitk_tools import clipseScaleSArray


def calculate_infarct(predict_path, reference, show_detail=False, intersting={1:[2221]}):



    pred_arr = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(predict_path)))
    img=sitk.ReadImage(reference)
    ref_arr=sitk.GetArrayFromImage(img)
    ref_arr=np.squeeze(ref_arr)
    # img_arr=np.flipud(img_arr)
    print(f"{predict_path} {np.unique(pred_arr)}")

    # img_arr=np.flip(img_arr,[0])
    # img_arr=np.transpose(img_arr,[1,0])

    # blobs = data.binary_blobs(200, blob_size_fraction=.2,
    #                           volume_fraction=.35, seed=1)
    myo=reindex_label_array_by_dict(ref_arr,{1:[200,1220,2221]})
    footprint = disk(1)
    ero_myo=binary_erosion(myo,footprint)
    dali_myo=binary_dilation(ero_myo)
    skeleton_lee = skeletonize(dali_myo.astype(np.uint8), method='lee')
    index=np.argwhere(skeleton_lee==1)
    # index=np.argwhere(myo==1)

    sector=np.zeros_like(skeleton_lee)

    center=np.mean(index,axis=0)
    # myo[int(center[0]),int(center[1])]=1

    transmularity=[]
    for i in range(0,3600,36):
        sector=make_sector(sector, center, np.max(skeleton_lee.shape), i / 10, (i + 36) / 10)
        # sector=make_line(sector, center, np.max(skeleton_lee.shape), i / 10)
        myo_blobs=sector*myo
        myo_blobs=np.where(myo_blobs>0,1,0)

        blobs_scar=sector*pred_arr
        blobs_scar=reindex_label_array_by_dict(blobs_scar,intersting)
        myo_blobs=np.where(myo_blobs>0,1,0)
        if np.sum(myo_blobs)==0:
            ratio=0
        else:
            ratio=np.sum(blobs_scar)/np.sum(myo_blobs)

        if ratio >=0 and ratio<0.25:
            tmp=0
        elif ratio>=0.25 and ratio<0.5:
            tmp=1
        elif ratio>=0.5 and ratio<0.75:
            tmp=2
        else:
            tmp=3
        # print(f"{tmp} {ratio}={np.sum(blobs_scar)}/{np.sum(myo_blobs)}")
        # transmularity = [i / 4 + 0.001 for i in transmularity]
        transmularity.append(tmp/3)
        # transmularity.append(ratio)

        if show_detail:
            cv2.imshow('IMG',ref_arr.astype(np.uint8))
            cv2.imshow('blobs_scar',blobs_scar.astype(np.uint8)*255)
            cv2.imshow('pred_arr',pred_arr.astype(np.uint8)*120)
            cv2.imshow('blobs_total',myo_blobs.astype(np.uint8)*255)
            blobs=reindex_label_array_by_dict(ref_arr,{255:[2221],100:[200]})
            cv2.imshow('Result',cv2.applyColorMap(blobs.astype(np.uint8),3))
            cv2.waitKey()
    # print(transmularity)

    transmularity.reverse()
    return transmularity
    # plt.imshow(blobs)
    # print(f"{i}")
    # plt.show()
    # plt.waitforbuttonpress()
    # plt.des


def get_path_s(mask, img,show_detail=False,dialed=1):


    img=img.astype(np.int32)
    # pred_arr = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(predict_path)))
    # img=sitk.ReadImage(reference)
    # ref_arr=sitk.GetArrayFromImage(img)
    # ref_arr=np.squeeze(ref_arr)
    # # img_arr=np.flipud(img_arr)
    # print(f"{predict_path} {np.unique(pred_arr)}")

    # img_arr=np.flip(img_arr,[0])
    # img_arr=np.transpose(img_arr,[1,0])

    # blobs = data.binary_blobs(200, blob_size_fraction=.2,
    #                           volume_fraction=.35, seed=1)
    health_pixels=[]
    myo=reindex_label_array_by_dict(mask, {1:[200]})

    footprint = disk(dialed)


    ero_myo=erosion(myo,footprint)
    # dali_myo=binary_dilation(ero_myo)
    skeleton_lee = skeletonize(ero_myo.astype(np.uint8), method='lee')
    ind_a=np.argwhere(skeleton_lee==1)
    ind_S = ind_a[np.random.choice(ind_a.shape[0], 5), :]
    ind = tuple(map(tuple, np.transpose(ind_a)))
    health_pixels.append(img[ind])

    # a=0
    # patch=np.zeros_like(mask)
    # for a in ind_S:
    #     patch = patch+extract_path(a, mask)
    #
    # patch=patch>0
    # patch=patch.astype(np.uint8)

    # n=3
    # patch=extract_pathV2(ind_a, mask,n)
    # while np.sum(patch*ero_myo)<20:
    #     print(np.sum(patch*ero_myo))
    #     # a=a+1
    #     # ind = ind[np.random.choice(ind.shape[0], 1), :]
    #     patch=extract_pathV2(ind_a, mask,n)

    # patch=extract_path(ind, mask)
    # while np.sum(patch*myo)<35:
    #     a=a+1
    #     ind = ind[np.random.choice(ind.shape[0], 1), :]
    #     patch=extract_path(ind[a], mask)


    # min=np.median(img[ind])
    # for x,y in zip(ind[0],ind[1]):
    #     if img[x,y]==min:
    #         print(f"{x},{y}")
    #         break
    # patch=extract_path([x,y], mask)
    # while np.sum(patch*myo)<35:
    #     a=a+1
    #     ind = ind[np.random.choice(ind.shape[0], 1), :]
    #     patch=extract_path(ind[a], mask)

    datas=img[ind]
    datas=np.sort(datas)
    a=0

    x, y = finddata(img, ind, datas[a])

    patch=extract_path([x,y], mask,n=3)
    # while np.sum(patch*ero_myo)<20 :
    #     # tmp = np.random.randint(0,len(ind[0]))
    #     a=a+1
    #     if a ==len(ind):
    #         x, y = finddata(img, ind, datas[0])
    #         patch = extract_path((x, y), mask)
    #         break
    #
    #     x, y = finddata(img, ind, datas[a])
    #     patch=extract_path((x,y), mask,n=3)

    if show_detail:
        tmp=patch * ero_myo
        tmp=np.where(tmp==1,0,1)

        cv2.imshow('IMG', clipseScaleSArray(img,0,100).astype(np.uint8))
        cv2.imshow('IMG_patch', (tmp*clipseScaleSArray(img,0,100)).astype(np.uint8))
        cv2.imshow('blobs_scar', skeleton_lee.astype(np.uint8) * 255)
        # cv2.imshow('blobs_scar', skeleton_lee.astype(np.uint8) * 255)
        cv2.imshow('patch', (patch * ero_myo).astype(np.uint8) * 255)
        # cv2.imshow('patch_mask', (patch * myo*mask).astype(np.uint8) )
        # cv2.imshow('pred_arr', pred_arr.astype(np.uint8) * 120)
        # cv2.imshow('blobs_total', myo_blobs.astype(np.uint8) * 255)
        # blobs = reindex_label_array_by_dict(ref_arr, {255: [2221], 100: [200]})
        # cv2.imshow('Result', cv2.applyColorMap(blobs.astype(np.uint8), 3))
        cv2.waitKey()

    return patch* ero_myo


def finddata(img, ind, min):
    for x, y in zip(ind[0], ind[1]):
        if img[x, y] == min:
            print(f"{x},{y}")
            break
    return x, y


def get_path_sv2(mask, img,show_detail=True, intersting={1:[2221]}):


    img=img.astype(np.int32)
    # pred_arr = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(predict_path)))
    # img=sitk.ReadImage(reference)
    # ref_arr=sitk.GetArrayFromImage(img)
    # ref_arr=np.squeeze(ref_arr)
    # # img_arr=np.flipud(img_arr)
    # print(f"{predict_path} {np.unique(pred_arr)}")

    # img_arr=np.flip(img_arr,[0])
    # img_arr=np.transpose(img_arr,[1,0])

    # blobs = data.binary_blobs(200, blob_size_fraction=.2,
    #                           volume_fraction=.35, seed=1)
    health_pixels=[]
    myo=reindex_label_array_by_dict(mask, {1:[200]})
    footprint = disk(1)
    ero_myo=erosion(myo,footprint)
    # dali_myo=binary_dilation(ero_myo)
    skeleton_lee = skeletonize(ero_myo.astype(np.uint8), method='lee')
    ind=np.argwhere(skeleton_lee==1)
    ind = tuple(map(tuple, np.transpose(ind)))
    min=np.min(img[ind])
    health_pixels.append(img[ind])

    for x,y in zip(ind[0],ind[1]):
        if img[x,y]==min:
            print(f"{x},{y}")
            break


    # index=np.argwhere(myo==1)
    # # ind = np.argwhere(healthy_myo == 2)
    # ind = ind[np.random.choice(ind.shape[0], 100), :]
    # ind = tuple(map(tuple, np.transpose(ind)))
    # health_pixels = []
    # health_pixels.append(target_img[ind])
    # mean = np.mean(health_pixels)
    # sd = np.std(health_pixels)

    a=0
    patch=extract_path([x,y], mask)
    while np.sum(patch*myo)<35:
        a=a+1
        ind = ind[np.random.choice(ind.shape[0], 1), :]
        patch=extract_path(ind[a], mask)

    if show_detail:
        cv2.imshow('IMG', mask.astype(np.uint8))
        cv2.imshow('blobs_scar', skeleton_lee.astype(np.uint8) * 255)
        # cv2.imshow('blobs_scar', skeleton_lee.astype(np.uint8) * 255)
        cv2.imshow('patch', (patch * myo).astype(np.uint8) * 255)
        # cv2.imshow('patch_mask', (patch * myo*mask).astype(np.uint8) )
        # cv2.imshow('pred_arr', pred_arr.astype(np.uint8) * 120)
        # cv2.imshow('blobs_total', myo_blobs.astype(np.uint8) * 255)
        # blobs = reindex_label_array_by_dict(ref_arr, {255: [2221], 100: [200]})
        # cv2.imshow('Result', cv2.applyColorMap(blobs.astype(np.uint8), 3))
        cv2.waitKey()

    return patch* myo


def extract_path(ind, ref_arr,n=3):
    # ind = ind[np.random.choice(ind.shape[0], 1), :]
    path_img = np.zeros_like(ref_arr)
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            path_img[ind[0] + i, ind[1] + j] = 1

    return path_img

def extract_pathV2(ind, ref_arr,n=3):
    ind = ind[np.random.choice(ind.shape[0], 1), :]
    path_img = np.zeros_like(ref_arr)
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            path_img[ind[0][0] + i, ind[0][1] + j] = 1
    return path_img


