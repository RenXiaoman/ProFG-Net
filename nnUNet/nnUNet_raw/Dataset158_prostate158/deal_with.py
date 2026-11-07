from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
import numpy as np
import SimpleITK as sitk
import os
from dataclasses import dataclass
import numpy.typing as npt
from numpy.testing import assert_allclose
from scipy import ndimage
from reg_lib import register_spline


@dataclass
class PreprocessingSettings():
    """
    Preprocessing settings
    - matrix_size: number of voxels output image (z, y, x)
    - spacing: output voxel spacing in mm/voxel (z, y, x)
    - physical_size: size in mm of the target image (z, y, x)
    - crop_only: only crop to specified size (i.e., do not pad)
    - align_segmentation: whether to align the scans using the centroid of the provided segmentation
    - scan_interpolator: interpolation method for scans
    - lbl_interpolator: interpolation method for labels
    - seg_interpolator: interpolation method for segmentations
    """
    matrix_size: Optional[Iterable[int]] = None
    spacing: Optional[Iterable[float]] = None
    physical_size: Optional[Iterable[float]] = None
    crop_only: bool = False
    scan_interpolator: int = sitk.sitkBSpline
    lbl_interpolator: int = sitk.sitkNearestNeighbor
    seg_interpolator: int = sitk.sitkNearestNeighbor

    def __post_init__(self):
        if self.physical_size is None and self.spacing is not None and self.matrix_size is not None:
            # calculate physical size
            self.physical_size = [
                voxel_spacing * num_voxels
                for voxel_spacing, num_voxels in zip(
                    self.spacing,
                    self.matrix_size
                )
            ]

        if self.spacing is None and self.physical_size is not None and self.matrix_size is not None:
            # calculate spacing
            self.spacing = [
                size / num_voxels
                for size, num_voxels in zip(
                    self.physical_size,
                    self.matrix_size
                )
            ]

class Case:
    def __init__(self, 
                 t2w: Union[Path, sitk.Image], 
                 adc: Union[Path, sitk.Image], 
                 dwi: Union[Path, sitk.Image], 
                 lesion: Union[Path, sitk.Image],
                 settings: PreprocessingSettings,
                 name: Optional[str] = None,
                 ):
        self.t2w = self._ensure_image(t2w)
        self.adc = self._ensure_image(adc)
        self.dwi = self._ensure_image(dwi)
        self.lbl = self._ensure_image(lesion)

        if isinstance(name, str):
            self.name = name

        self.settings = settings

        self.scans = [self.t2w, self.adc, self.dwi, self.lbl]

        self.post_init()

    def post_init(self):
        if self.lbl is not None:
            # keep track of connected components
            lbl = sitk.GetArrayFromImage(self.lbl)
            _, num_gt_lesions = ndimage.label(lbl, structure=np.ones((3, 3, 3)))
            self.num_gt_lesions = num_gt_lesions



    def _ensure_image(self, value: Union[Path, sitk.Image]) -> sitk.Image:
        if isinstance(value, sitk.Image):
            return sitk.Cast(value, sitk.sitkFloat32)
        elif isinstance(value, Path):
            if not value.exists():
                raise FileNotFoundError(f"文件不存在: {value}")
            return sitk.Cast(sitk.ReadImage(value), sitk.sitkFloat32) 
        else:
            raise TypeError(f"只支持 Path 或 sitk.Image, 当前类型: {type(value)}")
        
    #################  resample_img  #################
    def resample_img(
            self,
            image: sitk.Image,
            out_spacing: Iterable[float] = (2.0, 2.0, 2.0),
            out_size: Optional[Iterable[int]] = None,
            is_label: bool = False,
            interpolation = None,
            pad_value: Optional[Union[float, int]] = 0.,
    ) -> sitk.Image:
        """
        Resample images to target resolution spacing
        Ref: SimpleITK
        """
        # get original spacing and size
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        # convert our z, y, x convention to SimpleITK's convention
        out_spacing = list(out_spacing)[::-1]

        if out_size is None:
            # calculate output size in voxels
            out_size = [
                int(np.round(
                    size * (spacing_in / spacing_out)
                ))
                for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
            ]

        # determine pad value
        if pad_value is None:
            pad_value = image.GetPixelIDValue()

        # set up resampler
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(list(out_spacing))
        resample.SetSize(out_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(pad_value)

        if interpolation is not None:
            resample.SetInterpolator(interpolation)
        elif is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)

        # perform resampling
        image = resample.Execute(image)

        return image

    
    def resample_spacing(self,
                         spacing: Optional[Iterable[float]] = None):
        """Resample scans and label to the target spacing"""
        if spacing is None:
            assert self.settings.spacing is not None
            spacing = self.settings.spacing

        self.scans = [
            self.resample_img(scan, out_spacing=self.settings.spacing, interpolation=self.settings.scan_interpolator)
            for scan in self.scans
        ]

        # resample annotation to target resolution
        if self.lbl is not None:
            self.lbl = self.resample_img(self.lbl, out_spacing=spacing, interpolation=self.settings.lbl_interpolator)

    #################  centre_crop_or_pad  #################
    def input_verification_crop_or_pad(self,
    image: "Union[sitk.Image, npt.NDArray[Any]]",
    size: Optional[Iterable[int]] = (20, 256, 256),
    physical_size: Optional[Iterable[float]] = None,
    ) -> Tuple[Iterable[int], Iterable[int]]:
        """
        Calculate target size for cropping and/or padding input image

        Parameters:
        - image: image to be resized (sitk.Image or numpy.ndarray)
        - size: target size in voxels (z, y, x)
        - physical_size: target size in mm (z, y, x)

        Either size or physical_size must be provided.

        Returns:
        - shape of original image (in convention of SimpleITK (x, y, z) or numpy (z, y, x))
        - size of target image (in convention of SimpleITK (x, y, z) or numpy (z, y, x))
        """
        # input conversion and verification
        if physical_size is not None:
            # convert physical size to voxel size (only supported for SimpleITK)
            if not isinstance(image, sitk.Image):
                raise ValueError("Crop/padding by physical size is only supported for SimpleITK images.")
            spacing_zyx = list(image.GetSpacing())[::-1]
            size_zyx = [length/spacing for length, spacing in zip(physical_size, spacing_zyx)]
            size_zyx = [int(np.round(x)) for x in size_zyx]

            if size is None:
                # use physical size
                size = size_zyx
            else:
                # verify size
                if list(size) != list(size_zyx):
                    raise ValueError(f"Size and physical size do not match. Size: {size}, physical size: "
                                    f"{physical_size}, spacing: {spacing_zyx}, size_zyx: {size_zyx}.")

        if isinstance(image, sitk.Image):
            # determine shape and convert convention of (z, y, x) to (x, y, z) for SimpleITK
            shape = image.GetSize()
            size = list(size)[::-1]
        else:
            # determine shape for numpy array
            assert isinstance(image, (np.ndarray, np.generic))
            shape = image.shape
            size = list(size)
        rank = len(size)
        assert rank <= len(shape) <= rank + 1, \
            f"Example size doesn't fit image size. Got shape={shape}, output size={size}"

        return shape, size


    def crop_or_pad(self,
        image: Union[sitk.Image, npt.NDArray[Any]],
        size: Optional[Iterable[int]] = (20, 256, 256),
        physical_size: Optional[Iterable[float]] = None,
        crop_only: bool = False,
        pad_only: bool = False,
        pad_value: Union[float, int] = 0,
        ) -> Union[sitk.Image, npt.NDArray[Any]]:
        """
        Resize image by cropping and/or padding

        Parameters:
        - image: image to be resized (sitk.Image or numpy.ndarray)
        - size: target size in voxels (z, y, x)
        - physical_size: target size in mm (z, y, x)

        Either size or physical_size must be provided.

        Returns:
        - resized image (same type as input)
        """
        # input conversion and verification
        shape, size = self.input_verification_crop_or_pad(image, size, physical_size)

        # set identity operations for cropping and padding
        rank = len(size)
        padding = [[0, 0] for _ in range(rank)]
        slicer = [slice(None) for _ in range(rank)]

        # for each dimension, determine process (cropping or padding)
        for i in range(rank):
            if shape[i] < size[i]:
                if crop_only:
                    continue

                # set padding settings
                padding[i][0] = (size[i] - shape[i]) // 2
                padding[i][1] = size[i] - shape[i] - padding[i][0]
            else:
                if pad_only:
                    continue

                # create slicer object to crop image
                idx_start = int(np.floor((shape[i] - size[i]) / 2.))
                idx_end = idx_start + size[i]
                slicer[i] = slice(idx_start, idx_end)

        # crop and/or pad image
        if isinstance(image, sitk.Image):
            pad_filter = sitk.ConstantPadImageFilter()
            pad_filter.SetPadLowerBound([pad[0] for pad in padding])
            pad_filter.SetPadUpperBound([pad[1] for pad in padding])
            pad_filter.SetConstant(pad_value)
            return pad_filter.Execute(image[tuple(slicer)])
        else:
            return np.pad(image[tuple(slicer)], padding, constant_values=pad_value)
    

    def centre_crop_or_pad(self):
        """Centre crop and/or pad scans and label"""
        kwargs = {
            "size": self.settings.matrix_size,
            "physical_size": self.settings.physical_size,
            "crop_only": self.settings.crop_only,
        }
        self.scans = [
            self.crop_or_pad(scan, **kwargs)
            for scan in self.scans
        ]

        if self.lbl is not None:
            self.lbl = self.crop_or_pad(self.lbl, **kwargs)

    #################  resample_to_first_scan  #################
    def resample_to_first_scan(self):
        """Resample scans and label to the first scan"""
        # set up resampler to resolution, field of view, etc. of first scan
        resampler = sitk.ResampleImageFilter()  # default linear
        resampler.SetReferenceImage(self.scans[0])
        resampler.SetInterpolator(self.settings.scan_interpolator)

        # resample other images
        self.scans[1:] = [resampler.Execute(scan) for scan in self.scans[1:]]

        # resample annotation and segmentation
        resampler.SetInterpolator(self.settings.lbl_interpolator)

        if self.lbl is not None:
            self.lbl = resampler.Execute(self.lbl)

    #################  align_physical_metadata  #################
    def align_physical_metadata(self, check_almost_equal=True):
        """Align the origin and direction of each scan, and label"""
        case_origin, case_direction, case_spacing = None, None, None
        for img in self.scans:
            # copy metadata of first scan (nnUNet and nnDetection require this to match exactly)
            if case_origin is None:
                case_origin = img.GetOrigin()
                case_direction = img.GetDirection()
                case_spacing = img.GetSpacing()
            else:
                if check_almost_equal:
                    # check if current scan's metadata is almost equal to the first scan
                    assert_allclose(img.GetOrigin(), case_origin)
                    assert_allclose(img.GetDirection(), case_direction)
                    assert_allclose(img.GetSpacing(), case_spacing)

                # copy over first scan's metadata to current scan
                img.SetOrigin(case_origin)
                img.SetDirection(case_direction)
                img.SetSpacing(case_spacing)

        if self.lbl is not None:
            assert case_origin is not None and case_direction is not None and case_spacing is not None
            self.lbl.SetOrigin(case_origin)
            self.lbl.SetDirection(case_direction)
            self.lbl.SetSpacing(case_spacing)

    
    ###############  Registration  #################
    def get_gradient_features(self, image: sitk.Image) -> sitk.Image:
        """Return the average gradient of the 3D image in the (x, y, z) direction。"""
        grad = sitk.GradientImageFilter().Execute(image)
        gx = sitk.VectorIndexSelectionCast(grad, 0)
        gy = sitk.VectorIndexSelectionCast(grad, 1)
        gz = sitk.VectorIndexSelectionCast(grad, 2)
        return 0.33 * (gx + gy + gz)
    
    def register_image_pair(self,
        fixed_img: sitk.Image | str | Path,
        moving_img: sitk.Image | str | Path,
        *,
        to_float32: bool = True,
        verbose: bool = False,
    ) -> tuple[sitk.Image, sitk.Transform, float]:
        """
        使用 B-Spline (register_spline) 将 moving 配准到 fixed。

        Parameters
        ----------
        fixed_img, moving_img : `SimpleITK.Image` or path-like
            固定影像 / 待配准影像 (NIfTI、MHA 等)。
        to_float32 : bool
            若为 True, 则强制将读入影像转换为 sitkFloat32。

        Returns
        -------
        registered_img : `SimpleITK.Image`
            按 transform 重采样后的 moving 影像 (在 fixed 空间)。
        transform : `SimpleITK.Transform`
            由 B-Spline 优化得到的变换，可持久化保存。
        metric : float
            最终 Mattes Mutual Information 指标值。
        """
        # ---------- Read / Check Type ----------
        if isinstance(fixed_img, (str, Path)):
            fixed_img = sitk.ReadImage(str(fixed_img), sitk.sitkFloat32 if to_float32 else None)
        if isinstance(moving_img, (str, Path)):
            moving_img = sitk.ReadImage(str(moving_img), sitk.sitkFloat32 if to_float32 else None)

        # if the input is a 'Image' rather than a path, explicit cast can also be performed
        if to_float32: 
            fixed_img  = sitk.Cast(fixed_img,  sitk.sitkFloat32)
            moving_img = sitk.Cast(moving_img, sitk.sitkFloat32)

        # ---------- 梯度特征 ----------
        fixed_grad   = self.get_gradient_features(fixed_img)
        moving_grad  = -self.get_gradient_features(moving_img)

        # ---------- B-Spline 配准 ----------
        transform, metric = register_spline(
            fixed_grad,
            moving_grad,
            verbose=verbose,
        )

        # ---------- 重采样 ----------
        registered_img = sitk.Resample(
            moving_img,            # 要重采样的影像
            fixed_img,             # 目标空间
            transform,             # 计算得到的变换
            sitk.sitkLinear,       # 插值方式
            0.0,                   # 默认空洞填充值
            # moving_img.GetPixelID()
            sitk.sitkFloat32
        )

        return registered_img, transform, metric

    def register(self):
        # 遍历 scans[1] 和 scans[2]，都配准到 scans[0]
        for i in range(1, len(self.scans) - 1):  # 只遍历1和2
            registered_image, transform, metric = self.register_image_pair(
                self.scans[0],           # fixed image
                self.scans[i],           # moving image
                to_float32=True,
                verbose=False
            )
            self.scans[i] = registered_image  # 覆盖原有的

    ################################################
    #################  preprocess  #################
    ################################################
    def preprocess(self):
        """Perform all preprocessing steps"""
        # resample scans to target resolution
        self.resample_spacing(self.settings.spacing)

        # perform centre crop and/or pad
        self.centre_crop_or_pad()

        # resample scans and label to first scan's spacing, field-of-view, etc.
        self.resample_to_first_scan()

        # copy physical metadata to align subvoxel differences between sequences
        self.align_physical_metadata()

        if self.lbl is not None:
            # check connected components of annotation
            lbl = sitk.GetArrayFromImage(self.lbl)
            _, num_gt_lesions = ndimage.label(lbl, structure=np.ones((3, 3, 3)))
            assert self.num_gt_lesions == num_gt_lesions, \
                f"Label has changed due to resampling/other errors for {self.name}! " \
                + f"Have {self.num_gt_lesions} -> {num_gt_lesions} isolated ground truth lesions"
        
        # register scans
        self.register()

    def get_t2w(self):
        """Return T2W scan"""
        return self.scans[0]
    
    def get_adc(self):
        """Return ADC scan"""
        return self.scans[1]
    
    def get_dwi(self):
        """Return DWI scan"""
        return self.scans[2]
    
    def get_lbl(self):
        """Return label scan"""
        return self.lbl

#########################
# def main(TASK_DIR: str):
#     TASK_DIR = Path(TASK_DIR)

#     settings = PreprocessingSettings(
#         matrix_size=[20, 256, 256],
#         spacing=[3.5, 0.5, 0.5],
#     )

#     # Test
#     t2w_path = TASK_DIR / "bao chang sheng/localizer_t2_tse_tra__4959.0.0.0__002/1.3.12.2.1107.5.2.30.27678.2021011909361362524024959.0.0.0.nii.gz"
#     adc_path = TASK_DIR / "bao chang sheng/a_b0_b800_p2_160_ADC__8107.0.0.0__003/1.3.12.2.1107.5.2.30.27678.2021011909545596836828107.0.0.0.nii.gz"
#     dwi_path = TASK_DIR / "bao chang sheng/f_tra_b0_b800_p2_160__4285990060__001/1.2.826.0.1.3680043.8.498.81731386843390242450033246404285990060.nii.gz"
#     lesion_path = TASK_DIR / "bao chang sheng/localizer_t2_tse_tra__4959.0.0.0__002/1.3.12.2.1107.5.2.30.27678.2021011909361362524024959.0.0.0.nii.gz"

#     start = time.time()
#     case = Case(t2w_path, adc_path, dwi_path, lesion_path, settings)
#     case.preprocess()
#     end = time.time()
#     print(f"main()函数运行时长: {end-start:.2f} 秒")

    # print(f"T2W Image Spacing:{case.t2w.GetSpacing()}, Size:{case.t2w.GetSize()}" )
    # print(f"Label Image Spacing:{case.lbl.GetSpacing()}, Size:{case.lbl.GetSize()}" )
    # print('After preprocessing:')
    # print(f"T2W Image Spacing:{case.scans[0].GetSpacing()}, Size:{case.scans[0].GetSize()}" )
    # print(f"Label Image Spacing:{case.lbl.GetSpacing()}, Size:{case.lbl.GetSize()}" )




# if __name__ == "__main__":
#     TASK_DIR = "src/chengda/二分类/二分类图像/train/"
#     print("1")

#     # print("工作目录:", os.getcwd())
#     main(TASK_DIR)