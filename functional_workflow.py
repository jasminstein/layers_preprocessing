'''
export PATH=/data/u_steinj_software/conda/envs/preprocessing/bin/:$PATH
SCWRAP afni latest
SCWRAP freesurfer latest
C3D
'''

import os
import subprocess
import glob
import pickle

from pathlib import Path
import importlib.resources as pkg_resources

import nibabel as nib
import numpy as np
from scipy.ndimage import morphology, generate_binary_structure

import layers_preprocessing

import nipype
from nipype.interfaces import afni
from nipype.interfaces import freesurfer
from nipype.interfaces.freesurfer import BBRegister
from nipype.interfaces.c3 import C3dAffineTool

class functional():

    def __init__(self, subjectID, runID, taskID, baseDir, TR, acqID = None):

        self.baseDir = baseDir
        self.subjectID = subjectID
        self.runID = runID
        self.taskID = taskID
        self.bidsDir = f"{baseDir}/sub-{subjectID}"
        self.derivativesDir = f"{baseDir}/derivatives/sub-{subjectID}"

        self.funcDir = f"{baseDir}/derivatives/sub-{subjectID}/functional"
        self.stcDir = f"{baseDir}/derivatives/sub-{subjectID}/functional/stc"
        self.realDir = f"{baseDir}/derivatives/sub-{subjectID}/functional/realign"
        self.registerDir = f"{baseDir}/derivatives/sub-{subjectID}/functional/register"
        self.freesurfPath = f"{baseDir}/derivatives/sub-{subjectID}/freesurfer_out"

        self.TR = TR

        if acqID is not None:
            self.bold = f"{self.bidsDir}/func/sub-{subjectID}_task-{taskID}_run-{runID}_acq-{acqID}_bold.nii.gz"
            self.bold_json = f"{self.bidsDir}/func/sub-{subjectID}_task-{taskID}_run-{runID}_acq-{acqID}_bold.json"
        else:
            self.bold = f"{self.bidsDir}/func/sub-{subjectID}_task-{taskID}_run-{runID}_bold.nii.gz"
            self.bold_json = f"{self.bidsDir}/func/sub-{subjectID}_task-{taskID}_run-{runID}_bold.json"

        self.preprocAnat = f"{baseDir}/derivatives/sub-{subjectID}/freesurfer_out/mri/brain.mgz"
        #self.freesurfAseg = f"{baseDir}/derivatives/sub-{subjectID}/freesurfer_out/mri/aseg.mgz"
        #self.freesurfRibbon = f"{baseDir}/derivatives/sub-{subjectID}/freesurfer_out/mri/ribbon.mgz"

        #self.laynii = os.path.join('/data/u_steinj_software/', 'laynii/laynii/')

        self.derivativesDirPath = Path(self.derivativesDir)
        self.pklFile = self.derivativesDirPath / f'{self.subjectID}.func_pkl'

        if self.pklFile.exists():
            loaded = self.load(self.pklFile)
            self.__dict__.update(loaded.__dict__)
            print(f"Loaded existing SubjectData from {self.pklFile}")
        else:
            self.baseDir = baseDir
            self.subjectID = subjectID
            self.runID = runID
            self.taskID = taskID
            self.bidsDir = f"{baseDir}/sub-{subjectID}"
            self.derivativesDir = f"{baseDir}/derivatives/sub-{subjectID}"
            self.funcDir = f"{baseDir}/derivatives/sub-{subjectID}/functional"
            self.TR = TR

            if acqID is not None:
                self.bold = f"{self.bidsDir}//func/sub-{subjectID}_task-{taskID}_run-{runID}_acq-{acqID}_bold.nii.gz"
                self.bold_json = f"{self.bidsDir}/func/sub-{subjectID}_task-{taskID}_run-{runID}_acq-{acqID}_bold.json"
            else:
                self.bold = f"{self.bidsDir}/func/sub-{subjectID}_task-{taskID}_run-{runID}_bold.nii.gz"
                self.bold_json = f"{self.bidsDir}/func/sub-{subjectID}_task-{taskID}_run-{runID}_bold.json"

            #self.preprocAnat = f"{baseDir}/derivatives/sub-{subjectID}/freesurfer_out/mri/brain.mgz"
            #self.freesurfAseg = f"{baseDir}/derivatives/sub-{subjectID}/freesurfer_out/mri/aseg.mgz"
            #self.freesurfRibbon = f"{baseDir}/derivatives/sub-{subjectID}/freesurfer_out/mri/ribbon.mgz"

            #self.laynii = os.path.join('/data/u_steinj_software/', 'laynii/laynii/')

    @classmethod
    def load(cls, filename):
        '''load'''
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj
    
    def createOutputDirs(self):
        '''
        create the necessary output directories
        '''    
        os.makedirs(self.funcDir, exist_ok=True)
        os.makedirs(self.stcDir, exist_ok=True)
        os.makedirs(self.realDir, exist_ok=True)
        os.makedirs(self.registerDir, exist_ok=True)
        os.makedirs(f"{self.derivativesDir}/layers", exist_ok=True)
        os.makedirs(f"{self.freesurfPath}/sub-{self.subjectID}/mri", exist_ok=True)
    
    def addFile(self, name, path):
        '''
        add file to pickle
        '''
        setattr(self, name, path) 
        self.save()

    def save(self):
        '''
        save pickle
        '''
        with open(self.pklFile, "wb") as f:
            pickle.dump(self, f)

    def cleanEntry(self, key, delete_file=True):
        '''
        remove entry from pickle
        '''
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

    def writeNCorrectSliceTime(self):
        # TODO: necessary? --> no slice time correction in Degutis et al. or Denis' repositories
        
        '''
        function to write out slice timing from json file and slice time correct

        necessary input: 
        self.bold: path to bold image
        self.bold_json: path to json sidecar

        output: self.stcFile: path to slice time corrected bold file
        '''
        
        print("==== Running slice time correction ===")

        self.cleanEntry("stc_bold") 

        os.system(f"jq -r '.SliceTiming[]' {self.bold_json} > {self.stcDir}/slice_times.txt")
        
        sliceTimes = f'{self.stcDir}/slice_times.txt'
        stcFile = f'{self.stcDir}/stc_bold.nii.gz'

        tshift = afni.TShift()
        tshift.inputs.in_file = self.bold
        tshift.inputs.tzero = 0.5 # correct to middle slice
        tshift.inputs.tr = '%.1fs' % self.TR
        tshift.inputs.slice_timing = sliceTimes
        tshift.inputs.out_file = stcFile
        tshift.run(cwd=f'{self.stcDir}')

        stc_file = self.addFile("stc_bold", f'{stcFile}')
        self.save()  


    def realign(self):
        #TODO: maybe remove the first files (esp. if non-steady state) from outlier count

        '''
        function to compute the volume with the lowest fraction of outliers and realign
        to this volume

        input: 
        self.ctFile: path to slice time corrected bold timeseries
        
        output: 
        self.real_bold: path to realigned bold timeseries
        self.motion_pars: path to txt file containing volume-wise outlier fractions
        '''

        print("==== Running realignment ===")

        self.cleanEntry("real_bold") 
        self.cleanEntry("motion_params") 

        toutcount = afni.OutlierCount()
        toutcount.inputs.in_file = self.stc_bold
        toutcount.inputs.automask = True
        toutcount.inputs.fraction = True
        toutcount.inputs.legendre = True
        toutcount.inputs.polort = 5
        #toutcount.inputs.save_outliers = True
        toutcount.inputs.out_file = 'outlier_fraction.txt'
        outs = toutcount.run(cwd=f'{self.realDir}')

        outliers = outs.outputs.out_file

        with open(outliers) as outlier_file:
            values = [val.rstrip() for val in outlier_file]

        values = np.array(values)
        minInd = np.argmin(values)
        print(f'file index for realignment: {minInd}')    

        os.system("sc afni latest 3dvolreg" + \
            " -prefix " + os.path.join(f'{self.realDir}','realigned_bold.nii.gz') + \
            " -Fourier" + \
            " -float" + \
            " -base " + os.path.join(f'{self.stc_bold}')+f'\"[${minInd}]\"' + \
            " -dfile "  + os.path.join(f'{self.realDir}','motion.par') + \
            " -1Dfile " + os.path.join(f'{self.realDir}','motion.1D') + \
            " -maxdisp1D " + os.path.join(f'{self.realDir}','max_disp.1D') + \
            f" {self.stc_bold}"
                )

        for f in glob.glob(os.path.join(f'{self.realDir}', "*.nipype*")):
            os.remove(f)

        realFile = os.path.join(f'{self.realDir}','realigned_bold.nii.gz')
        movementPars = os.path.join(f'{self.realDir}','motion.1D')   

        real_file = self.addFile("real_bold", f'{realFile}')
        self.save()
        motion = self.addFile("motion_params", f'{movementPars}')
        self.save()

    def plotMovement(self):
        '''
        plot movement parameters (translations and rotations along 3 axes)

        input: s
        self.motion_pars: path to txt file containing movement parameters
        '''

        os.system(f'sc afni latest 1dplot {self.motion_params}') 

    def averageBold(self):
        '''
        function to create average bold image from bold time series for registration

        input: 
        self.real_bold: path to realigned bold timeseries

        output: 
        self.mean_bold: path to averaged bold image
        '''

        print("==== Averaging BOLD ===")

        self.cleanEntry("mean_bold") 

        os.system(f'3dTstat -mean -prefix {self.registerDir}/mean_bold_ref.nii.gz {self.realDir}/realigned_bold.nii.gz')
        meanBold = os.path.join(self.registerDir,'mean_bold_ref.nii.gz')
        
        mean_bold = self.addFile("mean_bold", f'{meanBold}')
        self.save()

    def registerAnat2FuncAnts(self, anat):
        '''
        function to register preprocessed anatomical to functional image
        after bias field correcting the mean bold image and creating an automatic brain mask for bold

        input:
        self.mean_bold: path to averaged bold image
        anat: object of class anatomical
        '''
        print("==== Running registration using ANTs only ===") 

        # bias field correct the bold mean (necessary?)
        os.system(f'sc ants latest N4BiasFieldCorrection -i {self.mean_bold} -o {self.registerDir}/n4_mean_bold_ref.nii.gz')
        print('bias corrected')

        # auto brain mask on functional --> low quality? alternatives? better than nothing?
        os.system(f'sc afni latest 3dAutomask -prefix {self.registerDir}/mean_bold_ref_mask.nii.gz -apply_prefix {self.registerDir}/masked_n4_mean_bold_ref.nii.gz {self.registerDir}/n4_mean_bold_ref.nii.gz')
        print('mask created, starting registration...')

        os.system("export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8")
        os.system(f'mkdir {self.registerDir}/ants')
        os.system("sc ants latest antsRegistration" + \
            " --verbose 1" + \
            " --dimensionality 3" + \
            " --float 1" + \
            " --output " + f"[{self.registerDir}/ants/anat2func_,{self.registerDir}/ants/anat2func_Warped.nii.gz,{self.registerDir}/ants/anat2func_InverseWarped.nii.gz]" + \
            " --interpolation Linear" + \
            " --use-histogram-matching 0" + \
            " --winsorize-image-intensities [0.005,0.995]" + \
            " --transform Rigid[0.05]" + \
            " --metric CC" + f"[{self.registerDir}/masked_n4_mean_bold_ref.nii.gz,{anat.fs_brain},0.7,32,Regular,0.1]" + \
            " --convergence [1000x500,1e-6,10]" + \
            " --shrink-factors 2x1" + \
            " --smoothing-sigmas 1x0vox" + \
            " --transform Affine[0.1]" + \
            " --metric MI" + f"[{self.registerDir}/masked_n4_mean_bold_ref.nii.gz,{anat.fs_brain},0.7,32,Regular,0.1]" + \
            " --convergence [1000x500,1e-6,10]" + \
            " --shrink-factors 2x1" + \
            " --smoothing-sigmas 1x0vox" + \
            " --transform SyN[0.1,2,0]" + \
            " --metric CC" + f"[{self.registerDir}/masked_n4_mean_bold_ref.nii.gz,{anat.fs_brain},1,2]" \
            " --convergence [500x100,1e-6,10]" + \
            " --shrink-factors 2x1" + \
            " --smoothing-sigmas 1x0vox" + \
            " -x " + f"{self.registerDir}/mean_bold_ref_mask.nii.gz")   

        '''
        NOTE Parameters Denis (test at some point whether this makes a difference TODO):

        antsRegistration \
        --verbose 1 \
        --dimensionality 3 \
        --float 0 \
        --output [fs_to_func_,fs_to_func_Warped.nii,fs_to_func_InverseWarped.nii]  \
        --use-histogram-matching 0 \
        --winsorize-image-intensities [0.005,0.995] \
        --initial-moving-transform init.txt \
        --transform Rigid[0.1] \
        --metric MI[${bold_file},fs_brain.nii,1,32,Regular,0.25] \
        --convergence [1000x500x250,1e-6,10] \
        --shrink-factors 4x2x1 \
        --smoothing-sigmas 2x1x0vox \
        --transform Affine[0.1] \
        --metric MI[${bold_file},fs_brain.nii,1,32,Regular,0.25] \
        --convergence [1000x500x250x100,1e-6,10] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox \
        --transform SyN[0.1,3,0] \
        --metric CC[${bold_file},fs_brain.nii,1,4] \
        --convergence [100x100x100x100,1e-6,10] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox \
        -x mask.nii
        '''   

    def registerAnat2FuncBBRegAnts(self, anat):
        '''
        alternative registration: use bbreg to compute initial transfor, then run ants SyN

        input
        self.mean_bold: mean bold time series used for registration
        anat: object of class anatomical
        '''

        print("==== Running registration using Freesurfer and ANTs ===") 

        # bias field correct the bold mean (necessary?)
        os.system(f'sc ants latest N4BiasFieldCorrection -i {self.mean_bold} -o {self.registerDir}/n4_mean_bold_ref.nii.gz')
        print('bias corrected')

        # auto brain mask on functional --> low quality? alternatives? better than nothing?
        os.system(f'sc afni latest 3dAutomask -prefix {self.registerDir}/mean_bold_ref_mask.nii.gz -apply_prefix {self.registerDir}/masked_n4_mean_bold_ref.nii.gz {self.registerDir}/n4_mean_bold_ref.nii.gz')
        print('mask created, starting registration...')

        os.system(f'mkdir {self.registerDir}/fs_ants')

        os.environ["SUBJECTS_DIR"] = anat.freesurfPath

        bbreg = BBRegister()
        bbreg.inputs.subject_id = f'sub-{self.subjectID}'
        bbreg.inputs.source_file = f'{self.registerDir}/masked_n4_mean_bold_ref.nii.gz'
        bbreg.inputs.init = 'fsl'
        bbreg.inputs.contrast_type = 'bold'
        bbreg.inputs.out_reg_file = os.path.join(f'{self.registerDir}/fs_ants', "mean_bold_ref_bbreg_01.dat")
        #bbreg.inputs.out_fsl_file = os.path.join(f'{self.registerDir}/fs_ants', "mean_bold_ref_bbreg_01.mat")
        bbreg.inputs.out_lta_file = os.path.join(f'{self.registerDir}/fs_ants', "mean_bold_ref_bbreg_01.lta")
        bbreg.inputs.dof = 12
        register = bbreg.run()

        os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

        subprocess.run([
            "lta_convert",
            "--inlta", f"{self.registerDir}/fs_ants/mean_bold_ref_bbreg_01.lta",
            "--outfsl", f"{self.registerDir}/fs_ants/mean_bold_ref_bbreg_01.mat"
        ])

        # convert FSL convention to ANTs usable
        c3 = C3dAffineTool()
        c3.inputs.transform_file = os.path.join(f'{self.registerDir}/fs_ants', "mean_bold_ref_bbreg_01.mat")
        c3.inputs.source_file = f'{self.registerDir}/masked_n4_mean_bold_ref.nii.gz'
        c3.inputs.reference_file = anat.fs_brain
        c3.inputs.itk_transform = os.path.join(f'{self.registerDir}/fs_ants', "mean_bold_ref_bbreg_01_ants.txt")
        c3.inputs.fsl2ras = True
        os.system(c3.cmdline) 

        # remove temp dirs
        for d in glob.glob(os.path.join(f'{self.registerDir}/fs_anat', "tmp*")):
            if os.path.isdir(d):
                shutil.rmtree(d)

        # run SyN only in ANTs additionally
        os.system("export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8")

        os.system("sc ants latest antsRegistration" + \
            " --verbose 1" + \
            " --dimensionality 3" + \
            " --float 1" + \
            " --output " + f"[{self.registerDir}/fs_ants/anat2func_,{self.registerDir}/fs_ants/anat2func_Warped.nii.gz,{self.registerDir}/fs_ants/anat2func_InverseWarped.nii.gz]" + \
            " --interpolation Linear" + \
            " --use-histogram-matching 0" + \
            " --winsorize-image-intensities [0.005,0.995]" + \
            " --initial-moving-transform " + f"[{self.registerDir}/fs_ants/mean_bold_ref_bbreg_01_ants.txt,1]" + \
            " --transform SyN[0.1,2,0]" + \
            " --metric CC" + f"[{self.registerDir}/masked_n4_mean_bold_ref.nii.gz,{anat.fs_brain},1,2]" \
            " --convergence [500x100,1e-6,10]" + \
            " --shrink-factors 2x1" + \
            " --smoothing-sigmas 1x0vox" + \
            " -x " + f"{self.registerDir}/mean_bold_ref_mask.nii.gz")

    
    def applyTransform(self, anat, prefix, method):
        '''
        function to apply computed registration parameters/warp field

        input: 
        anat: object of class anatomical
        prefix: prefix for output file
        method: ants or fs_ants (depending on which transform to apply)

        output:
        self.anat_2_bold_{method} = anatomical transformed to bold
        '''

        print("==== Running registration: applying transformation ===")

        self.cleanEntry(f"anat_2_bold_{method}") 

        os.system("sc ants latest antsApplyTransforms" + \
            " --interpolation BSpline[5]" + \
            " -d 3" + \
            " -i " + f"{anat.fs_brain}" + \
            " -r " + f"{self.registerDir}/masked_n4_mean_bold_ref.nii.gz" + \
            " -t " + f"{self.registerDir}/{method}/{prefix}_1Warp.nii.gz" + \
            " -t " + f"{self.registerDir}/{method}/{prefix}_0GenericAffine.mat" + \
            " -o " + f"{self.registerDir}/{method}/{prefix}_registered_2func.nii.gz")

        registered2func = os.path.join(f"{self.registerDir}",f"{method}",f"{prefix}_registered_2func.nii.gz")
        
        anat2func = self.addFile(f"anat_2_bold_{method}", f'{registered2func}')
        self.save()

    def upsampleBoldRef(self, factor = 5, method = "3['l']"):
        '''
        function to upsample boldref to aid registration between upsampled layers and bold

        input: 
        factor: upsampling factor (default = 5)
        method: interpolation method (default = 3['l'])

        output:
        self.upsampled_mean_bold: path to upsampled bold average
        '''

        print("==== Upsampling boldref ===")

        self.cleanEntry("upsampled_mean_bold") 

        nii = nib.load(f'{self.registerDir}/masked_n4_mean_bold_ref.nii.gz')
        voxel_sizes = nii.header.get_zooms()[:3]
        delta_x, delta_y, delta_z = voxel_sizes

        # Compute new voxel spacing
        sdelta_x = delta_x / factor
        sdelta_y = delta_y / factor
        sdelta_z = delta_z / factor

        cmd = f"sc ants latest ResampleImage 3 {self.registerDir}/masked_n4_mean_bold_ref.nii.gz {self.derivativesDir}/layers/upsampled_masked_n4_mean_bold_ref.nii.gz {str(sdelta_x)}x{str(sdelta_y)}x{str(sdelta_z)} 0 {method} 6"
        os.system(cmd)    

        upsampled_boldref = self.addFile("upsampled_mean_bold", f'{self.derivativesDir}/layers/upsampled_masked_n4_mean_bold_ref.nii.gz')
        self.save() 

    def layers2func(self, anat, prefix_trans, prefix_out, method):
        '''
        registers layers to upsampled bold average by applying the transforms computed before

        input:
        - anat.layers_anat_upsampled: path to layers file
        - prefix_trans: prefix chosen previously for transformation files 'anat2func'
        - prefix_out: output filename
        - method: ants or fs_ants

        output:
        - self.layers_2_func_{method}: path to upsampled layers in functional space
        '''

        print("==== Register layers ===")

        self.cleanEntry(f"layers_2_func_{method}") 

        os.system("sc ants latest antsApplyTransforms" + \
            " --interpolation NearestNeighbor" + \
            " -d 3" + \
            " -i " + f"{anat.layers_anat_upsampled}" + \
            " -r " + f"{self.upsampled_mean_bold}" + \
            " -t " + f"{self.derivativesDir}/functional/register/{method}/{prefix_trans}_1Warp.nii.gz" + \
            " -t " + f"{self.derivativesDir}/functional/register/{method}/{prefix_trans}_0GenericAffine.mat" + \
            " -o " + f"{self.derivativesDir}/layers/{prefix_out}_{method}.nii.gz")

        layers_func = self.addFile(f"layers_2_func_{method}", f'{self.derivativesDir}/layers/{prefix_out}_{method}.nii.gz')
        self.save() 


    def downsampleLayers(self, prefix_out, method):
        '''
        downsample layers to native resolution of the functional images

        input:
        - self.layers_2_func_{method}: path to upsampled layers in functional space
        - prefix_out: output file name
        - method: ants or fs_ants

        output:
        - self.layers_2_func_native_res; path to file containing the downsampled layers

        '''

        print("==== downsampling layers ===")

        self.cleanEntry("layers_2_func_native_res") 

        layers_in = getattr(self, f"layers_2_func_{method}")

        os.system("sc ants latest antsApplyTransforms" + \
            " --interpolation NearestNeighbor" + \
            " -d 3" + \
            " -i " + layers_in + \
            " -r " + f"{self.registerDir}/masked_n4_mean_bold_ref.nii.gz" + \
            " -t " + "identity" + \
            " -o " + f"{self.derivativesDir}/layers/{prefix_out}_{method}.nii.gz")

        layers_func_nat = self.addFile("layers_2_func_native_res", f'{self.derivativesDir}/layers/{prefix_out}_{method}.nii.gz')
        self.save()

    def registerRois2Func(self, anat, roi, reg_method, prefix):
        '''
        register anatomical ROIs from anatomical to functional space

        input:
        anat: object of class anatomical
        roi = roi name
        reg_method: ants or fs_ants
        prefix: previously used prefix (anat2func)

        output
        self.{roi}_2_func_{method}: roi in functional space
        '''

        print(f'=== registering {roi} to func ===')
        
        if roi in anat.rois_thalamus[0]:
            method = 'Linear'
            roi_path = getattr(anat, f"{roi}_2_anat")

        elif any(roi.lower() == r.lower() for r in anat.rois_juelich):
            method = 'Linear'
            roi_path = getattr(anat, f"{roi}_2_anat")

        elif roi in anat.rois_subcortical:
            method = 'NearestNeighbor'
            roi_path = getattr(anat, f"{roi}_2_anat")

        elif any(roi.lower() == r.lower() for r in anat.rois_juelich_maxprob_l[0]):
            method = 'NearestNeighbor'
            roi_path = getattr(anat, f"{roi}_2_anat")
            print(method)

        elif any(roi.lower() == r.lower() for r in anat.rois_juelich_maxprob_r[0]):
            method = 'NearestNeighbor'
            roi_path = getattr(anat, f"{roi}_2_anat")

        self.cleanEntry(f"{roi}_2_func_{method}") 
        
        os.system("sc ants latest antsApplyTransforms" + \
            " --interpolation " + f"{method}" + \
            " -d 3" + \
            " -i " + f"{roi_path}" + \
            " -r " + f"{self.registerDir}/masked_n4_mean_bold_ref.nii.gz" + \
            " -t " + f"{self.registerDir}/{reg_method}/{prefix}_1Warp.nii.gz" + \
            " -t " + f"{self.registerDir}/{reg_method}/{prefix}_0GenericAffine.mat" + \
            " -o " + f"{self.registerDir}/{roi}_{reg_method}_func.nii.gz")

        roi_func = self.addFile(f"{roi}_2_func_{method}", f"{self.registerDir}/{roi}_{reg_method}_func.nii.gz")
        self.save()

    def getWmCsfRegs(self, anat, reg_method = 'ants', prefix = 'anat2func'):
        '''
        generate nuisance regressors for WM and CSF signal

        input:
        anat: object of class anatomical
        reg_method: ants or fs_ants
        prefix: previously used prefix (anat2func)

        output:
        self.csf_func: path to csf mask in functional space
        self.wm_func path to wm mask in functional space
        self.wm_csf_regs: path to txt file containing regressors for mean wm/csf
        '''

        print('=== registering WM and CSF to func ===') 
        
        self.cleanEntry("csf_anat")
        self.cleanEntry("wm_anat")
        self.cleanEntry("csf_func")
        self.cleanEntry("wm_func") 
        self.cleanEntry("wm_csf_regs") 

        input = anat.fs_aseg_nii

        wm_labels = [2, 41]
        csf_labels = [4, 14, 15, 24, 43, 72, 217, 122, 257]

        aseg = nib.load(input)
        aseg_data = aseg.get_fdata()

        csf_mask = np.isin(aseg_data, csf_labels)
        wm_mask = np.isin(aseg_data, wm_labels)

        nib.save(nib.Nifti1Image(csf_mask.astype(np.uint8), aseg.affine),f'{self.freesurfPath}/sub-{self.subjectID}/mri/csf_mask.nii.gz')
        nib.save(nib.Nifti1Image(wm_mask.astype(np.uint8), aseg.affine),f'{self.freesurfPath}/sub-{self.subjectID}/mri/wm_mask.nii.gz')

        roi_func = self.addFile(f"csf_anat", f'{self.freesurfPath}/sub-{self.subjectID}/mri/csf_mask.nii.gz')
        self.save()
        roi_func = self.addFile(f"wm_anat", f'{self.freesurfPath}/sub-{self.subjectID}/mri/wm_mask.nii.gz')
        self.save()

        os.system("sc ants latest antsApplyTransforms" + \
            " --interpolation NearestNeighbor" + \
            " -d 3" + \
            " -i " + f"{self.csf_anat}" + \
            " -r " + f"{self.registerDir}/masked_n4_mean_bold_ref.nii.gz" + \
            " -t " + f"{self.registerDir}/{reg_method}/{prefix}_1Warp.nii.gz" + \
            " -t " + f"{self.registerDir}/{reg_method}/{prefix}_0GenericAffine.mat" + \
            " -o " + f"{self.registerDir}/csf_{reg_method}_func.nii.gz")

        
        os.system("sc ants latest antsApplyTransforms" + \
            " --interpolation NearestNeighbor" + \
            " -d 3" + \
            " -i " + f"{self.wm_anat}" + \
            " -r " + f"{self.registerDir}/masked_n4_mean_bold_ref.nii.gz" + \
            " -t " + f"{self.registerDir}/{reg_method}/{prefix}_1Warp.nii.gz" + \
            " -t " + f"{self.registerDir}/{reg_method}/{prefix}_0GenericAffine.mat" + \
            " -o " + f"{self.registerDir}/wm_{reg_method}_func.nii.gz")

        roi_func = self.addFile(f"csf_func", f'{self.registerDir}/csf_{reg_method}_func.nii.gz')
        self.save()
        roi_func = self.addFile(f"wm_func", f'{self.registerDir}/wm_{reg_method}_func.nii.gz')
        self.save()
        
        func = self.real_bold
        funcy = nib.load(func)
        funcy_data = funcy.get_fdata()     

        wm = nib.load(self.csf_func).get_fdata() > 0
        csf = nib.load(self.wm_func).get_fdata() > 0   
        wm_nuis = funcy_data[wm].mean(axis=0)
        csf_nuis = funcy_data[csf].mean(axis=0)

        nuis_regs = np.column_stack([wm_nuis, csf_nuis])
        np.savetxt(f'{self.realDir}/wm_csf.txt', nuis_regs, fmt='%.6f', delimiter='\t')

        roi_func = self.addFile(f"wm_csf_regs", f'{self.realDir}/wm_csf.txt')
        self.save()


    def registerAudit2func(self, anat, reg_method = 'ants', prefix = 'anat2func'):
        '''

        '''

        print('=== auditory rois to func ===') 
        
        os.system(f'sc freesurfer latest mri_convert {anat.freesurfPath}/sub-{self.subjectID}/mri/aparc.a2009s+aseg.mgz {self.freesurfPath}/sub-{self.subjectID}/mri/aparc_a2009s_aseg.mgz.nii.gz')  
        fs_seg_full = self.addFile("aparc_a2009s_aseg", f'{self.freesurfPath}/sub-{self.subjectID}/mri/aparc_a2009s_aseg.mgz.nii.gz')
        self.save()

        audit_labels = [11133, 12133, 11136, 12136, 11135, 12135, 11134, 12134]

        aseg = nib.load(self.aparc_a2009s_aseg)
        aseg_data = aseg.get_fdata()

        for roi in audit_labels:
            mask = np.isin(aseg_data, roi)

            nib.save(nib.Nifti1Image(mask.astype(np.uint8), aseg.affine),f'{self.freesurfPath}/sub-{self.subjectID}/mri/{roi}_mask.nii.gz')

            roi_anat = self.addFile(f"{roi}_anat", f'{self.freesurfPath}/sub-{self.subjectID}/mri/{roi}_mask.nii.gz')
            self.save()

            roi_path = getattr(self, f"{roi}_anat")

            os.system("sc ants latest antsApplyTransforms" + \
                " --interpolation NearestNeighbor" + \
                " -d 3" + \
                " -i " + f"{roi_path}" + \
                " -r " + f"{self.registerDir}/masked_n4_mean_bold_ref.nii.gz" + \
                " -t " + f"{self.registerDir}/{reg_method}/{prefix}_1Warp.nii.gz" + \
                " -t " + f"{self.registerDir}/{reg_method}/{prefix}_0GenericAffine.mat" + \
                " -o " + f"{self.registerDir}/{roi}_{reg_method}_func.nii.gz")


            roi_func = self.addFile(f"{roi}_func", f'{self.registerDir}/{roi}_{reg_method}_func.nii.gz')
            self.save()
        


     



