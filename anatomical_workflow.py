'''
export PATH=/data/u_steinj_software/conda/envs/preprocessing/bin/:$PATH
SCWRAP afni latest
SCWRAP freesurfer latest
SPM MATLAB --version 9.13 (MATLAB 2022b to run CAT12)
unset LD_PRELOAD (if runnning on remote Linux to disable graphics warnings)
'''

import os
import subprocess
import shutil
from pathlib import Path
import importlib.resources as pkg_resources
import layers_preprocessing

import pickle

import nipype
from nipype.interfaces import freesurfer
import nibabel as nib

import numpy as np
from scipy.ndimage import morphology, generate_binary_structure


class anatomical():

    def __init__(self, subjectID, runID, taskID, baseDir):

        self.baseDir = baseDir
        self.subjectID = subjectID
        self.runID = runID
        self.taskID = taskID
        self.bidsDir = f"{baseDir}/sub-{subjectID}"
        self.derivativesDir = f"{baseDir}/derivatives/sub-{subjectID}"
        self.catPath = f"{baseDir}/derivatives/sub-{subjectID}/cat_out"
        self.freesurfPath = f"{baseDir}/derivatives/sub-{subjectID}/freesurfer_out"
        self.inv2Path = f"{baseDir}/sub-{subjectID}/anat/sub-{subjectID}_task-{taskID}_acq-inv2_T1w.nii.gz"
        self.uniPath = f"{baseDir}/sub-{subjectID}/anat/sub-{subjectID}_task-{taskID}_acq-uni_T1w.nii.gz"

        self.mprageisePath = os.path.join('/data/u_steinj_software/', 'MPRAGEise-2.0/')
        self.laynii = os.path.join('/data/u_steinj_software/', 'LayNii_v2.9.0')
        self.matlabScriptPath = Path(pkg_resources.files(layers_preprocessing))

        self.mnis = ['mni_nlin_2009_c_sym_1mm','mni_nlin_2009_c_asym_1mm_brain','mni_nlin_2009_b_sym_05mm']
        self.mniPath = Path(pkg_resources.files(layers_preprocessing) / "roi" / "mni")
        self.juelich = Path(pkg_resources.files(layers_preprocessing) / "roi" / "juelich_auditory_cortex")
        self.juelich_maxprob = Path(pkg_resources.files(layers_preprocessing) / "roi" / "juelich_maxprob")
        self.thalamus = Path(pkg_resources.files(layers_preprocessing) / "roi" / "thalamus_probseg_freesurfer_atlas")
        self.subcortical = Path(pkg_resources.files(layers_preprocessing) / "roi" / "sitek_auditory_subcortical")

        self.rois_juelich = ['Te-1.0_l', 'Te-1.1_l', 'Te-1.2_l','Te-2.1_l','Te-2.2_l', 'Te-1.0_r', 'Te-1.1_r', 'Te-1.2_r','Te-2.1_r','Te-2.2_r']
        self.rois_juelich_maxprob_l = [['Te-1.0_l_max', 'Te-1.1_l_max', 'Te-1.2_l_max','Te-2.1_l_max','Te-2.2_l_max'], [10, 99, 161, 167, 204]] 
        self.rois_juelich_maxprob_r = [['Te-1.0_r_max', 'Te-1.1_r_max', 'Te-1.2_r_max','Te-2.1_r_max','Te-2.2_r_max'], [10, 99, 161, 167, 204]] 
        self.rois_thalamus = [['MGB_L', 'MGB_R', 'TRN_L','TRN_R'],[4, 3, 19, 39]] # NOTE: TRN labels wrong in atlas file by freesurfer?
        self.rois_subcortical = ['CN_L_sit', 'CN_R_sit','SOC_L_sit','SOC_R_sit','IC_L_sit','IC_R_sit','MGB_L_sit', 'MGB_R_sit']

        self.derivativesDirPath = Path(self.derivativesDir)
        self.pklFile = self.derivativesDirPath / f'{self.subjectID}.anat_pkl'

        if self.pklFile.exists():
            loaded = self.load(self.pklFile)
            for key, val in self.__dict__.items():
                loaded.__dict__[key] = val
            self.__dict__.update(loaded.__dict__)
            print(f"Loaded existing SubjectData from {self.pklFile}")
        else:
            self.baseDir = baseDir
            self.subjectID = subjectID
            self.runID = runID
            self.taskID = taskID
            self.bidsDir = f"{baseDir}/sub-{subjectID}"
            self.derivativesDir = f"{baseDir}/derivatives/sub-{subjectID}"
            self.catPath = f"{baseDir}/derivatives/sub-{subjectID}/cat_out"
            self.freesurfPath = f"{baseDir}/derivatives/sub-{subjectID}/freesurfer_out"
            self.inv2Path = f"{baseDir}/sub-{subjectID}/anat/sub-{subjectID}_task-{taskID}_acq-inv2_T1w.nii.gz"
            self.uniPath = f"{baseDir}/sub-{subjectID}/anat/sub-{subjectID}_task-{taskID}_acq-uni_T1w.nii.gz"

            self.mprageisePath = os.path.join('/data/u_steinj_software/', 'MPRAGEise-2.0/')
            self.laynii = os.path.join('/data/u_steinj_software/', 'LayNii_v2.9.0')
            self.matlabScriptPath = Path(pkg_resources.files(layers_preprocessing))

            self.mnis = mnis = ['mni_nlin_2009_c_sym_1mm','mni_nlin_2009_c_asym_1mm_brain','mni_nlin_2009_b_sym_05mm']
            self.mniPath = Path(pkg_resources.files(layers_preprocessing) / "roi" / "mni")
            self.juelich = Path(pkg_resources.files(layers_preprocessing) / "roi" / "juelich_auditory_cortex")
            self.juelich_maxprob = Path(pkg_resources.files(layers_preprocessing) / "roi" / "juelich_maxprob")
            self.thalamus = Path(pkg_resources.files(layers_preprocessing) / "roi" / "thalamus_probseg_freesurfer_atlas")
            self.subcortical = Path(pkg_resources.files(layers_preprocessing) / "roi" / "sitek_autitory_subcortical")

            self.rois_juelich = self.rois_juelich = ['Te-1.0_l', 'Te-1.1_l', 'Te-1.2_l','Te-2.1_l','Te-2.2_l', 'Te-1.0_r', 'Te-1.1_r', 'Te-1.2_r','Te-2.1_r','Te-2.2_r']
            self.rois_juelich_maxprob_l = [['Te-1.0_l_max', 'Te-1.1_l_max', 'Te-1.2_l_max','Te-2.1_l_max','Te-2.2_l_max'], [10, 99, 161, 167, 204]] 
            self.rois_juelich_maxprob_r = [['Te-1.0_r_max', 'Te-1.1_r_max', 'Te-1.2_r_max','Te-2.1_r_max','Te-2.2_r_max'], [10, 99, 161, 167, 204]] 
            self.rois_thalamus = [['MGB_L', 'MGB_R', 'TRN_L','TRN_R'],[4, 3, 19, 39]] # NOTE: TRN labels wrong in atlas file by freesurfer?
            self.rois_subcortical = ['CN_L_sit', 'CN_R_sit','SOC_L_sit','SOC_R_sit','IC_L_sit','IC_R_sit','MGB_L_sit', 'MGB_R_sit']
            
            print(f"Created new SubjectData for subject {subjectID}")

    @classmethod
    def load(cls, filename):
        '''load'''
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj
    
    def createOutputDirs(self):
        ''' create the necessary output directories'''
        os.makedirs(self.baseDir, exist_ok=True)
        os.makedirs(self.derivativesDir, exist_ok=True)
        os.makedirs(self.catPath, exist_ok=True)
        os.makedirs(self.freesurfPath, exist_ok=True)
    
    def addFile(self, name, path):
        '''add file to pickle'''
        setattr(self, name, path) 
        self.save()

    def save(self):
        '''save pickle'''
        with open(self.pklFile, "wb") as f:
            pickle.dump(self, f)

    def cleanEntry(self, key, delete_file=True):
        '''remove entry from pickle'''
        if not hasattr(self, key):
            print(f"[INFO] No attribute '{key}' found — nothing to clean.")
            return

        path = getattr(self, key)
        if delete_file and isinstance(path, str) and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"[WARN] Could not delete {path}: {e}")
        delattr(self, key)
        self.save()

    def useMPRAGEise(self):
        '''
        use MPRAGEISE module to prepare MP2RAGE image

        input:
        self.inv2Path: path to inv2 image of mp2rage
        self.uniPath: path to uni image of mp2rage

        output:
        self.mprageised: path to anatomical image used for recon-all
        '''

        print('=== running MPRAGEISE ===')

        self.cleanEntry("mprageised") 

        cmd = [
            "python",
            f'{self.mprageisePath}/MPRAGEise.py',
            "-i", self.inv2Path,
            "-u", self.uniPath,
            "-o", f'{self.catPath}/mprageise_out'
        ]
        print(cmd)
        subprocess.run(cmd)
        mprageised = f'{self.catPath}/mprageise_out/sub-{self.subjectID}_task-{self.taskID}_acq-uni_T1w_unbiased_clean.nii.gz'

        out_file = self.addFile("mprageised", mprageised)
        self.save()

    def runautorecon1(self):
        #TODO: check -cm flag (does it do anything in addition to -hires?)

        '''
        runs autorecon 1 on mprageised image
        '''

        print('=== running autorecon 1 ===')

        os.environ["SUBJECTS_DIR"] = self.freesurfPath

        os.system("sc freesurfer latest recon-all" + \
            " -i " + self.mprageised + \
            " -hires" + \
            " -autorecon1" + \
            " -noskullstrip" + \
            " -sd " + self.freesurfPath + \
            " -s " + f'sub-{self.subjectID}' + \
            " -parallel")

    def meow(self):
        '''
        runs external MATLAB script for using CAT12 for high quality segmentation

        input:
        self.mprageised: mprageised image

        output:
        self.cat_wm: path to cat12 white matter mask
        self.cat_gm: path to cat12 grey matter mask
        '''

        print('=== running cat12 segmentation ===')

        self.cleanEntry("cat_wm") 
        self.cleanEntry("cat_gm") 

        print(self.mprageised)

        matlab_cmd = (
            f"cat12_seg('{self.mprageised}','{self.catPath}');"
            "exit;"
        )

        subprocess.run([
            "spm",
            "-nodisplay",
            "-nodesktop",
            "-r",
            matlab_cmd
        ], cwd = self.matlabScriptPath, check=True)

        wm_file = self.addFile("cat_wm", f'{self.catPath}/mprageise_out/mri/p2sub-{self.subjectID}_task-{self.taskID}_acq-uni_T1w_unbiased_clean.nii')
        self.save()
        gm_file = self.addFile("cat_gm", f'{self.catPath}/mprageise_out/mri/p1sub-{self.subjectID}_task-{self.taskID}_acq-uni_T1w_unbiased_clean.nii')
        self.save()

    def createMask(self):
        # TODO: check: gdc file from scanner --> used by Denis for gradient distortion correction

        '''
        creates brain mask from cat12 white and grey matter

        output:
        self.cat_mask: cat12 brain mask from wm and gm
        self.cat_brain: cat12 mask applied to mprageised
        '''

        print('=== creating cat mask ===')

        self.cleanEntry("cat_mask") 
        self.cleanEntry("cat_brain") 

        wm_nii = nib.load(self.cat_wm)
        gm_nii = nib.load(self.cat_gm)
        uni_mprageised_nii = nib.load(self.mprageised)

        wm_data = wm_nii.get_fdata()
        gm_data = gm_nii.get_fdata()
        uni_mprageised_data = uni_mprageised_nii.get_fdata()

        brainmask_file = os.path.join(self.catPath, 'mprageise_out', 'brainmask_cat.nii')
        uni_mprageised_brain_file = os.path.join(self.catPath, 'mprageise_out', 'brain_cat.nii')

        brainmask_data = np.array(((wm_data > 0) | (gm_data > 0)),dtype=int)
        brainmask_nii = nib.Nifti1Image(brainmask_data,
                                        uni_mprageised_nii.affine,
                                        uni_mprageised_nii.header)
        nib.save(brainmask_nii,brainmask_file)
        
        uni_mprageised_brain_data = brainmask_data * uni_mprageised_data
        uni_mprageised_brain_nii = nib.Nifti1Image(uni_mprageised_brain_data,
                                                uni_mprageised_nii.affine,
                                                uni_mprageised_nii.header)
        nib.save(uni_mprageised_brain_nii, uni_mprageised_brain_file)

        brainmask_file = self.addFile("cat_mask", f'{self.catPath}/mprageise_out/brainmask_cat.nii')
        self.save()
        brain_file = self.addFile("cat_brain", f'{self.catPath}/mprageise_out/brain_cat.nii')
        self.save()

    def transformNapplyMask(self):

        '''
        transform cat mask to orig.mgz using neares neighbor interpolation and apply mask to autorecon1 T1 image & add
        the files where Freesurfer expects them to be
        '''

        print('=== transforming and applying cat mask ===')

        transmask = freesurfer.ApplyVolTransform()
        transmask.inputs.source_file = self.cat_mask
        transmask.inputs.target_file = os.path.join(self.freesurfPath, f'sub-{self.subjectID}', 'mri', 'orig.mgz')
        transmask.inputs.reg_header = True
        transmask.inputs.interp = "nearest"
        transmask.inputs.transformed_file = os.path.join(self.freesurfPath, f'sub-{self.subjectID}', 'mri', 'brainmask_mask.mgz')
        transmask.inputs.args = "--no-save-reg"
        transmask.run(cwd=self.freesurfPath)

        applymask = freesurfer.ApplyMask()
        applymask.inputs.in_file = os.path.join(self.freesurfPath, f'sub-{self.subjectID}','mri','T1.mgz')
        applymask.inputs.mask_file = os.path.join(self.freesurfPath, f'sub-{self.subjectID}', 'mri', 'brainmask_mask.mgz')
        applymask.inputs.out_file =  os.path.join(self.freesurfPath, f'sub-{self.subjectID}', 'mri', 'brainmask.mgz')
        applymask.run(cwd=self.freesurfPath)

        shutil.copy2(os.path.join(self.freesurfPath, f'sub-{self.subjectID}', 'mri', 'brainmask.mgz'),
                    os.path.join(self.freesurfPath, f'sub-{self.subjectID}', 'mri','brainmask.auto.mgz'))

    def runautorecon2N3(self):

        '''
        run autorecon 2 and 3
        '''

        print('=== running autorecon 2 and 3 ===')

        with open(os.path.join(self.catPath, 'mprageise_out', 'expert.opts'), 'w') as text_file:
            text_file.write('mris_inflate -n 100\n')

        os.environ["SUBJECTS_DIR"] = self.freesurfPath

        os.system("sc freesurfer latest recon-all" + \
            " -hires" + \
            " -autorecon2" + " -autorecon3" + \
            " -sd " + self.freesurfPath + \
            " -s " + f'sub-{self.subjectID}' + \
            " -expert " + os.path.join(self.catPath, 'mprageise_out', 'expert.opts') + \
            " -xopts-overwrite" + \
            " -parallel")

    def fsBrain2Nii(self):

        '''
        convert Freesurfer output brain to nifti
        '''

        self.cleanEntry("fs_brain") 

        os.system(f'sc freesurfer latest mri_convert {self.freesurfPath}/sub-{self.subjectID}/mri/brain.mgz {self.freesurfPath}/sub-{self.subjectID}/mri/fs_brain.nii.gz')  
        fs_brain = self.addFile("fs_brain", f'{self.freesurfPath}/sub-{self.subjectID}/mri/fs_brain.nii.gz')
        self.save()

    def createRim(self):
        '''
        created rim file from freesurfer aseg
        credit: https://github.com/ofgulban/LAYNII_extras/blob/38e607edbd6601b5893e519af3f791059fbe190d/demo-freesurfer_segmentation_to_rim/freesurfer_segmentation_to_rim.py 

        input:
        self.freesurfPath: path to freesurfer output directory

        output:
        self.rim_anat: path to created rim in anatomical space
        '''

        print("==== Creating rim ===")

        self.cleanEntry("rim_anat") 

        os.system(f'mkdir {self.derivativesDir}/layers')

        fs_aseg_mgz = self.addFile("fs_aseg_mgz", f'{self.freesurfPath}/sub-{self.subjectID}/mri/aseg.mgz')
        self.save()

        os.system(f'sc freesurfer latest mri_convert {self.fs_aseg_mgz} {self.freesurfPath}/sub-{self.subjectID}/mri/aseg.nii.gz')

        fs_aseg_nii = self.addFile("fs_aseg_nii", f'{self.freesurfPath}/sub-{self.subjectID}/mri/aseg.nii.gz')
        self.save()

        input = self.fs_aseg_nii
        print(input)

        wm_labels = [2, 41]
        gm_labels = [3, 42]

        # =============================================================================
        nii = nib.load(input)
        data = np.asarray(nii.dataobj)

        # -----------------------------------------------------------------------------
        # Fill in white matter but only as a border
        rim_wm = np.zeros(data.shape, dtype="int32")
        for i in wm_labels:
            rim_wm[data == i] = 1
        struct = generate_binary_structure(3, 3)
        rim_inner = morphology.binary_erosion(rim_wm, structure=struct, iterations=2)
        rim_inner = rim_inner - rim_wm

        # -----------------------------------------------------------------------------
        # Fill in gray matter
        rim_gm = np.zeros(data.shape, dtype="int32")
        for i in gm_labels:
            rim_gm[data == i] = 3

        # -----------------------------------------------------------------------------
        # Generate an outer gray matter border
        rim_out = morphology.binary_dilation(rim_gm, structure=struct, iterations=2)

        # -----------------------------------------------------------------------------
        # Collate all three labels into one in order
        rim = np.zeros(data.shape, dtype="int32")
        rim[rim_out != 0] = 1
        rim[rim_inner != 0] = 2
        rim[rim_gm != 0] = 3

        out_dir = f'{self.derivativesDir}/layers/'
        basename, ext = nii.get_filename().split(os.extsep, 1)
        out = nib.Nifti1Image(rim, header=nii.header, affine=nii.affine)
        out_path = os.path.join(out_dir, f"{basename}_rim{ext}")
        nib.save(out, "{}{}.{}".format(out_dir, "aseg_rim", ext))
        rimAnat = os.path.join(self.derivativesDir, 'layers', 'aseg_rim.nii.gz')

        rim = self.addFile("rim_anat", f'{self.derivativesDir}/layers/aseg_rim.nii.gz')
        self.save()

    def upsampleRim(self, factor = 5, method = 0):
        '''
        function to upsample rim for smoother layer estimation

        input:
        self.rim_anat: rim in anatomical space
        factor: upscaling factor (default = 5)
        method: interpolation method (default = 0, linear)

        output: 
        self.rim_anat_upsampled = path to upsampled rim in anatomical space
        '''
    
        print("==== Upsampling rim ===")

        self.cleanEntry("rim_anat_upsampled") 

        R1_LABELS = [1, 1]
        R2_LABELS = [2, 2]
        R3_LABELS = [3, 3]

        struct = generate_binary_structure(3, 3)
        nii = nib.load(self.rim_anat)
        data = np.asarray(nii.dataobj)

        r1 = np.zeros(data.shape, dtype="int32")
        for i in R1_LABELS:
            r1[data == i] = 1

        r2 = np.zeros(data.shape, dtype="int32")
        for i in R2_LABELS:
            r2[data == i] = 1

        r3 = np.zeros(data.shape, dtype="int32")
        for i in R3_LABELS:
            r3[data == i] = 1

        out = nib.Nifti1Image(r1, header=nii.header, affine=nii.affine)
        nib.save(out, f"{self.derivativesDir}/layers/r1.nii.gz")

        out = nib.Nifti1Image(r2, header=nii.header, affine=nii.affine)
        nib.save(out, f"{self.derivativesDir}/layers/r2.nii.gz")

        out = nib.Nifti1Image(r3, header=nii.header, affine=nii.affine)
        nib.save(out, f"{self.derivativesDir}/layers/r3.nii.gz")

        r1_file = f"{self.derivativesDir}/layers/r1.nii.gz"
        r2_file = f"{self.derivativesDir}/layers/r2.nii.gz"
        r3_file = f"{self.derivativesDir}/layers/r3.nii.gz"

        nii = nib.load(r1_file)
        
        voxel_sizes = nii.header.get_zooms()[:3]
        delta_x, delta_y, delta_z = voxel_sizes
        sdelta_x = delta_x / factor
        sdelta_y = delta_y / factor
        sdelta_z = delta_z / factor

        cmd = f"sc ants latest ResampleImage 3 {r1_file} {self.derivativesDir}/layers/upsampled_r1.nii.gz {str(sdelta_x)}x{str(sdelta_y)}x{str(sdelta_z)}"
        os.system(cmd)

        nii = nib.load(r2_file)
        cmd = f"sc ants latest ResampleImage 3 {r2_file} {self.derivativesDir}/layers/upsampled_r2.nii.gz {str(sdelta_x)}x{str(sdelta_y)}x{str(sdelta_z)}"
        os.system(cmd)

        nii = nib.load(r3_file)
        cmd = f"sc ants latest ResampleImage 3 {r3_file} {self.derivativesDir}/layers/upsampled_r3.nii.gz {str(sdelta_x)}x{str(sdelta_y)}x{str(sdelta_z)}"
        os.system(cmd)

        # add files
        r1_upsampled = f"{self.derivativesDir}/layers/upsampled_r1.nii.gz"
        r2_upsampled = f"{self.derivativesDir}/layers/upsampled_r2.nii.gz"
        r3_upsampled = f"{self.derivativesDir}/layers/upsampled_r3.nii.gz"

        r1 = nib.load(r1_upsampled)
        r2 = nib.load(r2_upsampled)
        r3 = nib.load(r3_upsampled)

        data1 = r1.get_fdata()
        data2 = r2.get_fdata()
        data3 = r3.get_fdata()

        stacked = np.stack([data1, data2, data3], axis=-1)
        argmax = np.argmax(stacked, axis=-1)
        all_zero_mask = np.all(stacked == 0, axis=-1)
        labels = argmax + 1
        labels[all_zero_mask] = 0
        label_img = nib.Nifti1Image(labels.astype(np.int16), r1.affine, r1.header)
        nib.save(label_img, f"{self.derivativesDir}/layers/aseg_rim_upsampled.nii.gz")

        rim_upsampled = self.addFile("rim_anat_upsampled", f'{self.derivativesDir}/layers/aseg_rim_upsampled.nii.gz')
        self.save()

    def layerifyRim(self, N = 3):

        '''uses LayNii to create N layers from rim files

        input:
        - self.rim_anat_upsampled: upsampled rim file in anatomical space
        - N: number of layers (default = 3)

        output:
        - self.layers_anat_upsampled = path to upsampled layers file
        '''

        print("==== Creating layers ===")

        self.cleanEntry("layers_anat_upsampled") 

        os.system(f"{self.laynii}/LN2_LAYERS -rim {self.rim_anat_upsampled} -nr_layers {N} -thickness -equal_counts -output {self.derivativesDir}/layers/{N}_layers_upsampled_anat")

        layers_upsampled = self.addFile("layers_anat_upsampled", f'{self.derivativesDir}/layers/{N}_layers_upsampled_anat_layers_equidist.nii')
        self.save()

    def prepMni(self, mni_idx, ext = 'nii.gz'):
        '''
        get skull stripped mnis and brain masks to improve registration to fs_brain
        '''

        mnix = self.mnis[mni_idx]
        template = f'{self.mniPath}/{self.mnis[mni_idx]}.{ext}'

        self.cleanEntry(f'mni_brain_{mnix}')
        self.cleanEntry(f'mni_brain_mask_{mnix}')

        if mni_idx == 0:
            os.system(f'sc fsl latest fslmaths {template} -mas {self.mniPath}/mni_nlin_2009_c_sym_1mm_brain_mask.nii.gz {self.mniPath}/{self.mnis[mni_idx]}_brain.{ext}')

        elif mni_idx == 1:
            done = True

        elif mni_idx == 2: 
            os.system(f'sc freesurfer latest mri_synthstrip -i {template} -o {self.mniPath}/{self.mnis[mni_idx]}_brain.{ext} --no-csf')
            os.system(f'sc fsl latest fslmaths {self.mniPath}/{self.mnis[mni_idx]}_brain.{ext} -thr 0 -bin {self.mniPath}/{self.mnis[mni_idx]}_brain_mask.{ext}')

        
        mni_brain = self.addFile(f'mni_brain_{mnix}', f'{self.mniPath}/{self.mnis[mni_idx]}_brain.{ext}')
        self.save()

        mni_brain = self.addFile(f'mni_brain_mask_{mnix}', f'{self.mniPath}/{self.mnis[mni_idx]}_brain_mask.nii.gz')
        self.save()

   
    def mni2anat(self, mni_idx, ext = 'nii.gz'):

        '''
        compute transforms between different MNI templated and anatomical

        input:
        mni_idx: mni index (0-2) indicating which template

        output:
        self.mni2anat_{mnix}_ + different ants outputs
        '''

        self.cleanEntry("fs_brain_bin")

        os.system(f'mkdir {self.derivativesDir}/mni2anat')
        os.system("export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8")

        mnix = self.mnis[mni_idx]

        print(f'=== registering {mnix} to anat ===')

        self.cleanEntry(f"{mnix}_2_anat_warped")
        self.cleanEntry(f"{mnix}_2_anat_inv_warped")
        self.cleanEntry(f"{mnix}_2_anat_1warp")
        self.cleanEntry(f"{mnix}_2_anat_1invwarp")
        self.cleanEntry(f"{mnix}_2_anat_0genericaffine")

        os.system(f" sc fsl latest fslmaths {self.fs_brain} -thr 0.1 -bin {self.freesurfPath}/sub-{self.subjectID}/mri/fs_brain_bin_mask.nii.gz")

        fs_brain_bin = self.addFile("fs_brain_bin", f'{self.freesurfPath}/sub-{self.subjectID}/mri/fs_brain_bin_mask.nii.gz')
        self.save()

        template = f'{self.mniPath}/{self.mnis[mni_idx]}_brain.{ext}'
        mask_mni = getattr(self, f'mni_brain_mask_{mnix}')

        '''
        os.system(
            f"sc ants latest antsApplyTransforms "
            f"-d 3 "
            f"-i {template_pre} "
            f"-r {self.fs_brain} "
            f"-o {self.mniPath}/{mnix}_reoriented.{ext} "
            f"--transform identity"
        )

        template = f'{self.mniPath}/{self.mnis[mni_idx]}_reoriented.{ext}'

        '''

        os.system("sc ants latest antsRegistration" + \
            " --verbose 1" + \
            " --dimensionality 3" + \
            " --float 1" + \
            " --output " + f'[{self.derivativesDir}/mni2anat/mni2anat_{mnix}_,{self.derivativesDir}/mni2anat/mni2anat_{mnix}_Warped.nii.gz, {self.derivativesDir}/mni2anat/mni2anat_{mnix}_InverseWarped.nii.gz]' + \
            " --interpolation Linear" + \
            " --use-histogram-matching 0" + \
            " --winsorize-image-intensities [0.005,0.995]" + \
            " --transform Rigid[0.05]" + \
            " --metric CC" + f'[{self.fs_brain},{template},0.7,32,Regular,0.1]' + \
            " --convergence [1000x500,1e-6,10]" + \
            " --shrink-factors 2x1" + \
            " --smoothing-sigmas 1x0vox" + \
            " --transform Affine[0.1]" + \
            " --metric MI" + f'[{self.fs_brain},{template},0.7,32,Regular,0.1]' + \
            " --convergence [1000x500,1e-6,10]" + \
            " --shrink-factors 2x1" + \
            " --smoothing-sigmas 1x0vox" + \
            " --transform SyN[0.1,2,0]" + \
            " --metric CC" + f'[{self.fs_brain},{template},1,2]' \
            " --convergence [500x100,1e-6,10]" + \
            " --shrink-factors 2x1" + \
            " --smoothing-sigmas 1x0vox" + \
            " -x " + f'[{self.fs_brain_bin}, {mask_mni}]')    

        warped = self.addFile(f"{mnix}_2_anat_warped", f'{self.derivativesDir}/mni2anat/mni2anat_{mnix}_Warped.nii.gz')
        self.save()
        inverseWarped = self.addFile(f"{mnix}_2_anat_inv_warped", f'{self.derivativesDir}/mni2anat/mni2anat_{mnix}_InverseWarped.nii.gz')
        self.save()
        warp1 = self.addFile(f"{mnix}_2_anat_1warp", f'{self.derivativesDir}/mni2anat/mni2anat_{mnix}_1Warp.nii.gz')
        self.save()
        inverseWarp1 = self.addFile(f"{mnix}_2_anat_1invwarp", f'{self.derivativesDir}/mni2anat/mni2anat_{mnix}_1InverseWarp.nii.gz')
        self.save()
        genericAffine = self.addFile(f"{mnix}_2_anat_0genericaffine", f'{self.derivativesDir}/mni2anat/mni2anat_{mnix}_0GenericAffine.mat')
        self.save()

    def applyMni2Anat(self, mni_idx, ext):

        mnix = self.mnis[mni_idx]

        #template = f'{self.mniPath}/{self.mnis[mni_idx]}_reoriented.{ext}'
        template = f'{self.mniPath}/{self.mnis[mni_idx]}.{ext}'

        os.system("sc ants latest antsApplyTransforms" + \
            " --interpolation BSpline[5]" + \
            " -d 3" + \
            " -i " + f"{template}" + \
            " -r " + f"{self.fs_brain}" + \
            " -t " + f"{self.derivativesDir}/mni2anat/mni2anat_{mnix}_1Warp.nii.gz" + \
            " -t " + f"{self.derivativesDir}/mni2anat/mni2anat_{mnix}_0GenericAffine.mat" + \
            " -o " + f"{self.derivativesDir}/mni2anat/{mnix}_anat.nii.gz")

        mni_func = self.addFile(f"{mnix}_2_anat", f'{self.derivativesDir}/mni2anat/{mnix}_anat.nii.gz')
        self.save() 

    def rois2Anat(self,roi): 

        '''
        register rois from MNI to anatomical space

        input:
        - roi: roi name

        output:
        - {roi}_2_anat: path to roi in anatomical space
        '''

        print(f'=== registering {roi} to anat ===')

        self.cleanEntry(f"{roi}_2_anat") 
        
        if roi in self.rois_thalamus[0]:
            mni_idx = 0
            mnix = self.mnis[mni_idx]

            method = 'Linear'
            
            rois_thal_flat = np.array(self.rois_thalamus).flatten()
            index_roi = np.where(rois_thal_flat == roi)[0][0]
            index_vol = self.rois_thalamus[1][index_roi]

            atlas = nib.load(os.path.join(self.thalamus, 'ThalamusProbs.MNIsymSpace.nii.gz'))
            atlas_data = atlas.get_fdata()

            roi_data = atlas_data[:, :, :, index_vol] 
            roi_img = nib.Nifti1Image(roi_data, atlas.affine, atlas.header)
            nib.save(roi_img, f'{self.thalamus}/{roi}.nii.gz')

            roi_path = f'{self.thalamus}/{roi}.nii.gz'

        elif any(roi.lower() == r.lower() for r in self.rois_juelich):
            mni_idx = 1
            mnix = self.mnis[mni_idx]
            method = 'Linear'

            roi_path = f'{self.juelich}/{roi}.nii.gz'

        elif roi in self.rois_subcortical:
            mni_idx = 2
            mnix = self.mnis[mni_idx]
            method = 'NearestNeighbor'

            rois_subcortical_flat = np.array(self.rois_subcortical).flatten()
            index_roi = np.where(rois_subcortical_flat == roi)[0][0]

            atlas = nib.load(os.path.join(self.subcortical, 'sub-invivo_mni_rois_resampled.nii.gz'))
            atlas_data = atlas.get_fdata()

            roi_data = (atlas_data == index_roi + 1).astype(np.uint8)
            roi_img = nib.Nifti1Image(roi_data, atlas.affine, atlas.header)
            nib.save(roi_img, f'{self.subcortical}/{roi}.nii.gz')

            roi_path = f'{self.subcortical}/{roi}.nii.gz'

        elif any(roi.lower() == r.lower() for r in self.rois_juelich_maxprob_l[0]):
            mni_idx = 1
            mnix = self.mnis[mni_idx]
            method = 'NearestNeighbor'

            rois_juel_flat = np.array(self.rois_juelich_maxprob_l).flatten()
            index_roi = np.where(rois_juel_flat == roi)[0][0]
            index_vol = self.rois_juelich_maxprob_l[1][index_roi]

            atlas = nib.load(os.path.join(self.juelich_maxprob, 'JulichBrainAtlas_3.1_207areas_MPM_lh_MNI152.nii.gz'))
            atlas_data = atlas.get_fdata()

            roi_data = (atlas_data == index_vol).astype(np.uint8)
            roi_img = nib.Nifti1Image(roi_data, atlas.affine, atlas.header)
            nib.save(roi_img, f'{self.juelich_maxprob}/{roi}.nii.gz')

            roi_path = f'{self.juelich_maxprob}/{roi}.nii.gz'

        elif any(roi.lower() == r.lower() for r in self.rois_juelich_maxprob_r[0]):
            mni_idx = 1
            mnix = self.mnis[mni_idx]
            method = 'NearestNeighbor'

            rois_juel_flat = np.array(self.rois_juelich_maxprob_r).flatten()
            index_roi = np.where(rois_juel_flat == roi)[0][0]
            index_vol = self.rois_juelich_maxprob_r[1][index_roi]

            atlas = nib.load(os.path.join(self.juelich_maxprob, 'JulichBrainAtlas_3.1_207areas_MPM_rh_MNI152.nii.gz'))
            atlas_data = atlas.get_fdata()

            roi_data = (atlas_data == index_vol).astype(np.uint8)
            roi_img = nib.Nifti1Image(roi_data, atlas.affine, atlas.header)
            nib.save(roi_img, f'{self.juelich_maxprob}/{roi}.nii.gz')

            roi_path = f'{self.juelich_maxprob}/{roi}.nii.gz'


        os.system("sc ants latest antsApplyTransforms" + \
            " --interpolation " + f"{method}" + \
            " -d 3" + \
            " -i " + f"{roi_path}" + \
            " -r " + f"{self.fs_brain}" + \
            " -t " + f"{self.derivativesDir}/mni2anat/mni2anat_{mnix}_1Warp.nii.gz" + \
            " -t " + f"{self.derivativesDir}/mni2anat/mni2anat_{mnix}_0GenericAffine.mat" + \
            " -o " + f"{self.derivativesDir}/mni2anat/{roi}_anat.nii.gz")

        roi_func = self.addFile(f"{roi}_2_anat", f'{self.derivativesDir}/mni2anat/{roi}_anat.nii.gz')
        self.save() 
