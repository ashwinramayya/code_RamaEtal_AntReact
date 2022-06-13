"""
ccdt_func (color change detection task)

These functions  analyzing behavioral and iEEG data fromf patients performing the CCDT task. All data was collected at Hospital of University of 
Pennsylvania (HUP). Raw data is stored on redcap and iEEG portal.

Ashwin G. Ramayya (ashwinramayya@gmail.com)

03/24/2020
"""

# import packages
import numpy as np
import pandas as pd
import mne
import json as json
from scipy.io import loadmat # to load matlab
from scipy import stats,ndimage,signal,optimize, cluster
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg
import random as rd
import pickle
import os
import subprocess
import pycircstat as circ
import fooof as ff
import tensorpac as tp
from sklearn.metrics import r2_score,silhouette_score,calinski_harabasz_score, davies_bouldin_score,pairwise_distances
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import nibabel as nib
from nilearn import image,plotting,surface,datasets



from pylab import * # plotting

# Create Subject class that contains library of functions used to process behavioral and ephys data from a subject
class Subject:

    # constructor
    def __init__(self,subj,paramsDict = None, do_init=True):
        #Input
        # subj Id
        # paramsDict = Optional. dictionary of params to update. Will overwrite default param values

        # subect
        self.subj = subj


        #params dictionary
        # default params
        # all times in ms
        # wave_number of cycles is linearly spaced values from 7 (for lowest freq) to 3 (for highest freq)
        #wave_number_of_freq = 10
        # saveTF... flag of whether or not to save TF data (largest file size, most flexible)
        # savePow.... flag to save down sampled power data (smaller file size, specified a time window to bin)
        # savePhs ... flag to save phase data for a particular condition
        # saveAmp ... flag to save amp data for a particular condition
        # overwriteFlag .... flag to overwrite saved data


        self.params = {'error_trial_thresh':(0,1000), # error trials are RTs< 200 and > 1000
                       'fastResponse_thresh':250,
                       'montage':'bipolar',
                       'tmin_ms':-1000, 'tmax_ms':4000,
                       'eeg_notch_filter':False,
                       'eeg_filt_l_freq':62,
                       'eeg_filt_h_freq':58,
                       'buffer_ms':1000,
                       'reject_z_thresh':10,
                       'frange_HFA':(70,200),
                       'frange_LFA':(2,40),
                       'frange_thetaL':(3,6),
                       'frange_thetaH':(6,8),
                       'frange_theta':(3,8),
                       'frange_alpha':(8,12),
                       'frange_beta':(12,30),
                       'frange_gamma':(30,50),
                       'wave_frange':(70,200),
                       'wave_number_of_freq':5,
                       'trange_periEvent':500,
                       'trange_periEventShort':250,
                       'trange_periEventLong':1000,
                       'trange_prebuildup_ms':250,
                       'saveTF': False,
                       'savePow': False,
                       'savePhs': False,
                       'saveAmp': False,
                        'overwriteFlag': False}

        # add wave number of cycles
        self.params['wave_number_of_cycles'] = 3

        # this is how you set wavenumbers to decrease as a function of freq so that you optimize freq resolution at low freqs and temporal resolution at high freqs
        #self.params['wave_number_of_cycles'] = np.flip(np.linspace(3,7,self.params['wave_number_of_freq']))

        # update this dict if dictionary values are provided
        if paramsDict != None:
            self.params.update(paramsDict)

        # load snr_info JSON file to get directories and subjInfor
        with open("ccdt_info.json", "r") as read_file:
            ccdt_info = json.load(read_file)

        # find index of current subject
        s_idx = None
        for s in arange(0,np.shape(ccdt_info['subjects'])[0]):
            if subj==(ccdt_info["subjects"][s]['subj']):
                s_idx = s

        self.subjInfo = ccdt_info['subjects'][s_idx]
        self.subjDir = ccdt_info['dirs']['data']+ccdt_info['dirs']['eegDir']+self.subj+'/'
        self.eegDir = self.subjDir+'eeg.noreref'+'/' # always grab no-reref data, can do custom montaging later
        self.talDir = self.subjDir+'tal'+'/'
        # two tal files:
        # 1) talFile_bipolar is mni coords
        # 2) talFile_bipolar_native is native T1 space with patient specific parcellation
        self.talFile_bipolar = self.subj+'_electrodelabels_and_coordinates_mni_mid.csv'
        self.talFile_bipolar_native = self.subj+'_electrodenames_coordinates_mid.csv'
        self.talFile_bipolar_wm_labels = self.subj+'_wm_labels.csv'
        self.eventsDir =  ccdt_info['dirs']['data']+ccdt_info['dirs']['eventsDir']

        self.eventsFile = self.eventsDir+self.subj+'_events.mat'


        # Initialize with basic functions:
        # ( ) update this as you go
        if do_init == True:
            self.mkDirs()
            self.loadEvents()
            self.loadJacksheetAndParams()
            self.loadTalData()
    # update params dictionary
    def updateParamsDict(self,key,val):
        # update params dictionary
        self.params[key] = val

    def mkDirs(self):
        # This function makes directories that will be used in subsequent functions

        # create scratch dir if it doesnt exist
        self.scratch_dir = os.getcwd()+'/scratch/'
        if os.path.exists(self.scratch_dir)==False:
            print('tring to make dir')
            os.mkdir(self.scratch_dir)

        # create fig dir
        self.fig_dir = os.getcwd()+'/figs/'
        if os.path.exists(self.fig_dir)==False:
            os.mkdir(self.fig_dir)

        # params lbl
        self.params_lbl = self.params['montage']+'-'+str(self.params['tmin_ms'])+'-'+str(self.params['tmax_ms'])+'-'+str(self.params['reject_z_thresh'])+'-'+str(self.params['eeg_notch_filter'])+'-'+str(self.params['buffer_ms'])+'-'+str(self.params['wave_frange'][0])+'-'+str(self.params['wave_frange'][1])+'-'+str(self.params['wave_number_of_freq'])+'/'

        # create params dir  if it doesnt exist
        self.params_dir = self.scratch_dir+self.params_lbl
        if os.path.exists(self.params_dir)==False:
            os.mkdir(self.params_dir)

        # create paramsFig dir  if it doesnt exist
        self.paramsFig_dir = self.fig_dir+self.params_lbl
        if os.path.exists(self.paramsFig_dir)==False:
            os.mkdir(self.paramsFig_dir)


    def loadJacksheetAndParams(self):
        # This function will load a list of electrodes from the jacksheet.txt file and sample rate from the params file

        # jacksheet filename
        self.eNum_js = []
        self.eLbl_js = []
        fname_jacksheet = self.eegDir+'jacksheet.txt'
        f = open(fname_jacksheet, 'r')
        for line in f:
            x = line.split()
            self.eNum_js = self.eNum_js + [int(x[0])]
            self.eLbl_js = self.eLbl_js+[x[1]]
        f.close()

        # read params
        fname_params = self.eegDir+'params.txt'
        params= pd.read_csv(fname_params,delimiter=' ',header = None)
        f = open(fname_params, 'r')
        while x != []:
            x = f.readline().split()
            if x == []:
                f.close()
            elif x[0] in ['samplerate','gain']:
                setattr(self,x[0],float(x[1]))
            elif x[0] == 'dataformat':
                setattr(self,x[0],x[1])

    # this function generates a csv file containing white matter labels for each bipolar electrode pair
    def mni2atlasLabel_wrapper(self):
        # Load bipolar channels and mni coordinates
        if self.params['montage'] == 'bipolar':
            talpath = self.talDir+self.talFile_bipolar

        # get data frame for all electrodes localized in mni space
        tal_df = pd.read_csv(talpath,usecols=[0,1,2,3,4],
                                    names=['eLbl',
                                           'anat',
                                           'x',
                                          'y',
                                          'z'])

        # function to transform an mni coordinate to white matter label
        def mni2wmLabel(mni_coords,atlas_name='XTRACT HCP Probabilistic Tract Atlases'):
            output = subprocess.getoutput('atlasquery -a "'+str(atlas_name)+'" -c '+str(mni_coords[0])+','+str(mni_coords[1])+','+str(mni_coords[2]))
            # parse output
            anat_str = output.split('<br>')[1]

            if anat_str == 'No label found!':

                # get lobe information
                output = subprocess.getoutput('atlasquery -a "'+str('MNI Structural Atlas')+'" -c '+str(mni_coords[0])+','+str(mni_coords[1])+','+str(mni_coords[2]))

                lobe_str = output.split('<br>')[1]

                if lobe_str == 'No label found!':
                    wm_lbl = 'unlabelled'
                elif ',' in lobe_str:
                    wm_lbl = lobe_str.split('% ')[1].split(',')[0]+'_wm'
                else:
                    wm_lbl = lobe_str.split('% ')[1]+'_wm'

            else:
                if ',' in anat_str:
                    wm_lbl = anat_str.split('% ')[1].split(',')[0]
                else:
                    wm_lbl = anat_str.split('% ')[1]
            return wm_lbl

        # empty list container
        wm_dict_list = []

        # loop through electrodes and get whitematter label
        for e in np.arange(0,len(tal_df)):

            # init dict
            wm_dict = {}

            # get wm label
            wm_dict['anat_wm'] = mni2wmLabel(mni_coords=(tal_df.iloc[e]['x'],tal_df.iloc[e]['y'],tal_df.iloc[e]['z']))

            # append list
            wm_dict_list.append(wm_dict)

            # print
            print(e,'/',len(tal_df))

        # create df
        tal_df_wm = pd.DataFrame(wm_dict_list,index = tal_df['eLbl'].to_numpy())

        # save as csv
        tal_df_wm.to_csv(path_or_buf = self.talDir+self.subj+'_wm_labels.csv',header=False)

    #   this function load taliarach localization data from disk
    def loadTalData(self,overwrite_wm_flag=False):
        # expects that tal folder contains 3 csv files:
        #1)  SUBJ_electrodenames_coordinates_mid.csv for electrode coordinates in native T1 space
        #2) SUBJ_electrodelabels_and_coordinates_mni_mid.csv for MNI coords
        #3) SUBJ_wm_labels.csv that assigns a label for each electrode using white matter atlases. If missing, it generates it from the SUBJ_electrodelabels_and_coordinates_mni_mid.csv file and mni2wmlabel function

        # Load bipolar channels and coordinates
        if self.params['montage'] == 'bipolar':
            # To get MNI coords
            talpath = self.talDir+self.talFile_bipolar
            # to get subject specific labels
            talpath_native = self.talDir+self.talFile_bipolar_native
            # to get 
            talpath_wm = self.talDir+self.talFile_bipolar_wm_labels


        # Load monopolar channels and coordinates
        #if self.params['montage'] == 'monopolar':
        #    talpath = self.talDir+self.talFile_monopolar

        # read MNI file to data frame
        self.tal_df = pd.read_csv(talpath,usecols=[0,1,2,3,4],
                                    names=['eLbl',
                                           'anat',
                                           'x',
                                          'y',
                                          'z'])
        # read native loc dataframe
        tal_df_native = pd.read_csv(talpath_native,usecols=[0,1],\
            names = ['eLbl','anat_native'])

        # insert label from native dataframe to self.tal_df
        self.tal_df.insert(loc=len(self.tal_df.columns),\
            column = 'anat_native',value=tal_df_native['anat_native'])

        # load wm labels
        # write wm label csv if we havent created it yet
        if (overwrite_wm_flag == True)|(os.path.exists(talpath_wm)==False):
            self.mni2atlasLabel_wrapper()

        tal_df_wm = pd.read_csv(talpath_wm,usecols=[0,1],names=['eLbl','anat_wm'])

        # insert label from native dataframe to self.tal_df
        self.tal_df.insert(loc=len(self.tal_df.columns),\
            column = 'anat_wm',value=tal_df_wm['anat_wm'])

        # Remove entries from tal data frame for electrodes not included in jacksheet
        # index to drop
        idx_to_drop = np.array([])
        uElbl_list = []

        # loop through tal df
        for e in self.tal_df['eLbl']:

            # split lbl in to e1 and 2
            e1 = e.split(' - ')[0]
            e2 = e.split(' - ')[1]
            uElbl_list.append(self.subj+'-'+e1+'-'+e2)

            # identify rows where both lbls are not in jacksheet
            if ((e1 in self.eLbl_js)==False) | ((e2 in self.eLbl_js)==False):
                idx_to_drop = np.append(idx_to_drop,self.tal_df.query('eLbl==@e').index)


        # add a column with uElbl
        self.tal_df.insert(loc = len(self.tal_df.columns),
                           column='uElbl',value=uElbl_list)

        # drop indices of electrodes not in jacksheet
        self.tal_df.drop(labels=idx_to_drop,axis=0,inplace=True)
    def parse_anat_lbl(self,anat,anat_wm):
        """takes 'anat' which is a string indicating assigined anatomical label and 'anat_wm' which is a string indicating white matter label and returns 'anat_clean' which is a string that can be used by self.anat2roi (drops laterality and assigns whitematter label when appropriate)"""
        # key patterns
        l = 'Left'
        r = 'Right'
        m = '/'
        hipp = 'Hippocampus'
        wm = 'White Matter'
        anat_clean = anat
        
        #  ROI if hippocampal, relabel
        if hipp in anat_clean:
            anat_clean = 'Hippocampus'
            
        if wm in anat_clean:
            anat_clean = anat_wm#'White Matter'    

        # parse unlabeled (assign wm label)
        if (anat_clean==' ') | (anat_clean=='nan'):
            anat_clean = anat_wm        
        
        # remove '/' (assumes it is the last index)
        if m in anat_clean:
            anat_clean =anat_clean[:-1]     
        
        # remove 'Left'
        if l in anat_clean:
            idx = int((np.where(l in anat_clean)[0][0])+6)
            anat_clean =anat_clean[idx:]
        
        # remove 'Right'
        if r in anat_clean:
            idx = int((np.where(r in anat_clean)[0][0])+7)
            anat_clean =anat_clean[idx:]  

        # if still empty, 
        if anat_clean == ' ':
            anat_clean = 'unlabelled'
            
        return anat_clean



    def anat2roi(self,anat):
        """Takes 'anat' a string indicating anatomical label and returns 'roi' a string indicating reigon of interest"""
        # occipital
        if anat in ['MOG middle occipital gyrus', \
                    'SOG superior occipital gyrus', \
                    'OFuG occipital fusiform gyrus', \
                    'IOG inferior occipital gyrus',\
                    'LiG lingual gyrus','Occipital Lobe_wm','Optic Radiation L', 'Optic Radiation R','Forceps Major','Vertical Occipital Fasciculus L','Vertical Occipital Fasciculus R']:
            roi = 'Occipital'

        # parietal
        elif anat in ['AnG angular gyrus','PCu precuneus',\
                      'SPL superior parietal lobule',\
                      'SMG supramarginal gyrus',\
                      'PO parietal operculum','Parietal Lobe_wm']:
            roi = 'Parietal'
        # temporal
        elif anat in ['FuG fusiform gyrus', 'ITG inferior temporal gyrus',\
                     'PT planum temporale', 'MTG middle temporal gyrus',\
                      'FuG fusiform gyrus/"BA36',\
                      'PHG parahippocampal gyrus',\
                      'STG superior temporal gyrus',\
                      'TTG transverse temporal gyrus',\
                      'PP planum polare','TMP temporal pole','Temporal Lobe_wm','Acoustic Radiation L','Acoustic Radiation R']:
            roi = 'Temporal'

        # CINGULATE
        elif anat in ['ACgG anterior cingulate gyrus',\
                      'PCgG posterior cingulate gyrus',\
                     'MCgG middle cingulate gyrus','Cingulum subsection: Dorsal L', 'Cingulum subsection: Dorsal R',
           'Cingulum subsection: Peri-genual L',
           'Cingulum subsection: Peri-genual R',
           'Cingulum subsection: Temporal L',
           'Cingulum subsection: Temporal R']:
            roi = 'Cingulate'

        # MTL
        elif anat in ['Hippocampus','Amygdala','Ent entorhinal area','Fornix L', 'Fornix R']:
            roi = 'MTL'

        # PERIROLANDIC     
        elif anat in ['PoG postcentral gyrus',\
                     'CO central operculum',\
                      'PrG precentral gyrus','Corticospinal Tract L',
           'Corticospinal Tract R']:
            roi = 'Perirolandic-CST'

        # PREFRONTAL (LATERAL)
        # elif anat in ['FO frontal operculum',\
        #               'OpIFG opercular part of the inferior frontal gyrus',\
        #               'OrIFG orbital part of the inferior frontal gyrus',\
        #               'MFG middle frontal gyrus',\
        #               'TrIFG triangular part of the inferior frontal gyrus']:
        #     roi = 'Lateral Prefrontal'  

        # # PREFRONTAL (MEDIAL)
        # elif anat in ['SMC supplementary motor cortex', 
        #               'POrG posterior orbital gyrus', 'SFG superior frontal gyrus',\
        #               'GRe gyrus rectus',\
        #               'MOrG medial orbital gyrus', \
        #               'MFC medial frontal cortex', \
        #               'MSFG superior frontal gyrus medial segment', \
        #               'AOrG anterior orbital gyrus', 'FRP frontal pole']:
        #     roi = 'Medial Prefrontal' 
        # elif anat in ['Forceps Minor','Anterior Commissure','Frontal Lobe_wm']:
        #     roi = 'Frontal'


        # PREFRONTAL 
        elif anat in ['FO frontal operculum',\
                      'OpIFG opercular part of the inferior frontal gyrus',\
                      'OrIFG orbital part of the inferior frontal gyrus',\
                      'MFG middle frontal gyrus',\
                      'TrIFG triangular part of the inferior frontal gyrus','SMC supplementary motor cortex', 'POrG posterior orbital gyrus', 'SFG superior frontal gyrus','GRe gyrus rectus','MOrG medial orbital gyrus','MFC medial frontal cortex','MSFG superior frontal gyrus medial segment','AOrG anterior orbital gyrus', 'FRP frontal pole','Forceps Minor','Anterior Commissure','Frontal Lobe_wm']:
 
            roi = 'Prefrontal'


        # INSULA    
        elif anat in ['PIns posterior insula','AIns anterior insula','Insula_wm']:
            roi = 'Insula'
        #WM tracts 
        elif anat in ['Anterior Thalamic Radiation R','Anterior Thalamic Radiation L',\
                      'Superior Thalamic Radiation L', 'Superior Thalamic Radiation R','Frontal Aslant Tract L','Frontal Aslant Tract R']:#
            roi = 'Thalamocortical WM'
        elif anat in ['Arcuate Fasciculus L','Arcuate Fasciculus R','Uncinate Fasciculus L',
           'Uncinate Fasciculus R']:
            roi = 'Arc/Unc Fasiculus'
        elif anat in ['Inferior Fronto-Occipital Fasciculus L',\
                      'Inferior Fronto-Occipital Fasciculus R']:
            roi = 'IFOF WM'
        elif anat in ['Inferior Longitudinal Fasciculus L',
           'Inferior Longitudinal Fasciculus R','Middle Longitudinal Fasciculus L','Middle Longitudinal Fasciculus R']:
            roi = 'ILF-MLF WM'
        elif anat in ['Superior Longitudinal Fasciculus 1 L',
           'Superior Longitudinal Fasciculus 1 R',
           'Superior Longitudinal Fasciculus 2 L',
           'Superior Longitudinal Fasciculus 2 R',
           'Superior Longitudinal Fasciculus 3 L',
           'Superior Longitudinal Fasciculus 3 R']:
            roi = 'SLF WM'
        elif anat in ['Putamen','Caudate_wm',
                     'Caudate','Putamen_wm']:
            roi = 'Striatum'

        # OTHER
        elif anat in ['Cerebellum Exterior',\
                     'Inf Lat Vent','Ventral DC',\
                     'Lateral Ventricle',\
                     'unlabelled']:
            roi = 'unlabelled'
        else:
            roi = anat
        return roi
    def loadAtlas_yeo(self):
        """generates an atlas_dictionary the yeo buckner functional connectivity atlas. defaults to 7 network version with thick (liberal) parcellation 

        returns atlas_dict"""
        #from nilearn import image,datasets

        # fetch Yeo dataset
        yeo = datasets.fetch_atlas_yeo_2011()

        # load image object
        img = nib.load(yeo['thick_7'])

        #load image data 4D array. Freesurfer uses LIA orientation.  Dimension 0 is x-axis (+left, -right), 1 is z-axis (+ inf, - sup)), 2 is y-axis (+ant, -post). 
        yeo_img = img.get_fdata()

        # get affine (vox -> mni)
        yeo_aff =img.get_affine()

        # compute inverse of affine (mni -> vox)
        yeo_aff_inv = np.linalg.inv(yeo_aff)


        # atlas dict
        atlas_dict = {}
        atlas_dict['img'] = yeo_img
        atlas_dict['affine_vox2mni'] = yeo_aff
        atlas_dict['affine_mni2vox'] = yeo_aff_inv

        #below fields are populated from the atlas readme
        atlas_dict['roi_lbls'] = ['unlabelled','Visual','Somatomotor','DorsalAttention','Salience','Limbic','Frontoparietal','Default']

        def rgb2rgba(rgb_aslist,alpha = 1):
            """scales rgb values from 0 to 1 and adds transperency"""
            return np.array(list(np.array(rgb_aslist)/255)+[1])

        atlas_dict['roi_rgba'] = {}
        atlas_dict['roi_rgba'][''] = rgb2rgba([0,0,0])
        atlas_dict['roi_rgba']['Visual'] = rgb2rgba([120,18,134])
        atlas_dict['roi_rgba']['Somatomotor'] = rgb2rgba([70,130,180])
        atlas_dict['roi_rgba']['DorsalAttention'] = rgb2rgba([0,118,14])
        atlas_dict['roi_rgba']['Salience'] = rgb2rgba([196,58,250])
        atlas_dict['roi_rgba']['Limbic'] = rgb2rgba([220,248,164])
        atlas_dict['roi_rgba']['Frontoparietal'] = rgb2rgba([230,148,34])
        atlas_dict['roi_rgba']['Default'] = rgb2rgba([205,62,78])

        return atlas_dict

    def mni2roi(self,atlas_dict,mni_x,mni_y,mni_z):
        """ This function takes in an atlas_dict and mni coordinates and returns aroi label 

        inputs:
        atlas_dict ... dictionary containing vox2mni affine transofrm, image data, and look up tables for roi labels  (see loadAtlas_yeo)
        mni_x .. x coordinate ()
        mni_y .. y coord
        mni_z .. z coord


        returns
        roi ... string assigning an antomical label """

        #import nibabel as nib


        # transform mni coordinates to voxel coordinates (in atlas space)
        mni_xyz= np.array([mni_x,mni_y,mni_z])
        vox_xyz = nib.affines.apply_affine(atlas_dict[
            'affine_mni2vox'],mni_xyz).astype('int')

        # get value of voxel by looking up this coordinate 
        roi_idx = int(atlas_dict['img'][vox_xyz[0],vox_xyz[1],vox_xyz[2]])

        # get atlas label by looking up this value in atlas_dict
        roi = atlas_dict['roi_lbls'][roi_idx]

        return roi


    def parse_unabelled(self,roi_list,mni_coords):
        """takes an list of ROI and associated MNI coordinates. It replaces every 'unlabelled' electrode with the label associated with the nearest match """

        good_idx = np.array(roi_list) !='unlabelled' 

        for i in range(0,len(roi_list)):
            if roi_list[i] == 'unlabelled':
              D = pairwise_distances(np.array(mni_coords[i,:][np.newaxis]),mni_coords[good_idx,:]) 

              # closest match 
              roi_list[i] = roi_list[np.nonzero(good_idx)[0][np.argmin(D)]]

              #print(roi_list[i])

        return roi_list


    # load behavior
    def loadEvents(self):
        # This function loads events structure and convers it to a dataframe

        # load events structure mat file
        self.evStruct = loadmat(self.eventsFile)['events']

        # extract event data into a dictionary
        self.ev_dict = {}
        self.ev_dict['subj'] =  np.concatenate(np.squeeze(self.evStruct['subject'][0]))
        self.ev_dict['session'] =  np.concatenate(np.squeeze(self.evStruct['session'][0]))
        self.ev_dict['set'] = np.concatenate(np.concatenate(self.evStruct['set'][0]))
        self.ev_dict['trial'] = np.concatenate(np.concatenate(self.evStruct['trial'][0]))
        self.ev_dict['type'] =  np.concatenate(np.squeeze(self.evStruct['type']))
        self.ev_dict['RT'] = np.concatenate(np.concatenate(self.evStruct['rt'][0]))
        self.ev_dict['delay'] = np.concatenate(np.concatenate(self.evStruct['delay'][0]))
        self.ev_dict['mstime'] = np.concatenate(np.concatenate(self.evStruct['mstime'][0]))
        self.ev_dict['eegfile'] = np.concatenate(np.squeeze(self.evStruct['eegfile']))
        self.ev_dict['eegoffset'] = np.concatenate(np.concatenate(self.evStruct['eegoffset'][0]))
        self.ev_dict['targetLoc_x'] = np.concatenate(np.concatenate(self.evStruct['targetLoc_x'][0]))
        self.ev_dict['targetLoc_y'] = np.concatenate(np.concatenate(self.evStruct['targetLoc_y'][0]))
        self.ev_dict['targetLoc_lbl'] = np.concatenate(np.concatenate(np.concatenate(self.evStruct['targetLoc_lbl'][0])))


        # replace eeg file with just the filename (can append self.eegDir later when loading eeg data)
        for i in arange(0,len(self.ev_dict['eegfile'])):
            self.ev_dict['eegfile'][i]=self.ev_dict['eegfile'][i].split('/')[-1]

        # convert to dataframe
        self.ev_df = pd.DataFrame.from_dict(self.ev_dict)

        # insert additional columns to dataframe
        # error trials - RT <= 0 or >= 1000 (set in params)
        self.ev_df.insert(loc = len(self.ev_df.columns),column='error',value=(self.ev_df['RT'].to_numpy()<=self.params['error_trial_thresh'][0]) | (self.ev_df['RT'].to_numpy()>=self.params['error_trial_thresh'][1]))

        # fast response trials - RT <= 200 (set in params. eventually fit for each subject)
        self.ev_df.insert(loc = len(self.ev_df.columns),column='fastResponse',value=(self.ev_df['RT'].to_numpy()<=self.params['fastResponse_thresh'])&(self.ev_df['RT'].to_numpy()>0))


        # target locked RT (measures RT from target onset)
        self.ev_df.insert(loc = len(self.ev_df.columns),column='RT_targ',value=(self.ev_df['RT'].to_numpy()+self.ev_df['delay'].to_numpy()))



        # lag - 1 trial (trial_lagmin1)
        self.ev_df.insert(loc = len(self.ev_df.columns),column='trial_lagmin1',value=(self.ev_df.trial.to_numpy().astype('int') - 1))

        # [ ] lag-1 delay
        # [ ] lag-1 error

        # create a master list of ev_df (in case we filter ev_df etc in methods)
        self.ev_df_master = self.ev_df.copy()

        # track how events have been filtered.
        #Update these to see how events_df have been filtered
        self.ev_sessFilt = None
        self.ev_evQuery = None

        # get session list
        self.sess_list = np.unique(self.ev_df['session'].to_numpy())

        # initialize misc. tracking variables
        self.thisDelay = None

    # To aggregate data across multiple sessions,two options
        #[X] create an instance of Session and use subclass routines.
        #[ ] convert Session to a method of Subject that takes sess_idx as an input rather than a subclass

    # fcn to apply a filter remove error trials
    def filterEvents(self,evQuery = 'error==0'):
        #evQuery='type=="RESPONSE"'
        #evQuery='type=="FIX_ON"'
        #evQuery='type=="CC"'
        self.ev_df = self.ev_df.query(evQuery)
        self.ev_evQuery = evQuery
    # this function reverts trials back to master list
    def revertEvents(self):
        self.ev_df = self.ev_df_master.copy()
        self.ev_evQuery = None

    def save_pickle(self,obj, fpath, verbose=True):
        """Save object as a pickle file. Written by Dan Schonhaut. 11/2018"""
        with open(fpath, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        if verbose:
            print('Saved {}'.format(fpath))

    def load_pickle(self,fpath):
        """Return object. Written by Dan Schonhaut. 11/2018"""
        with open(fpath, 'rb') as f:
            obj = pickle.load(f)
        return obj

    # Fisher Z-transform of pearson r 
    def fisher_z(self,r):
        z = 0.5*(np.log((1+r)/(1-r)))
        return z

    # Get prior trial memories
    def getPriorTrialMemory(self,ev_df,tau = 1,  memory_buffer = 10):
        # This function creates a trial-by-trial representation of events that happened on preceeding trials. 
        # Inputs:
        # ev_df ... events dataframe. Can be filtered. Will use self.ev_df_master to get complete trial history prior to filtering
        # trialFeature ... what feature of past trials to capture (e.g., error, short delay, long delay, prior RTs, spatial location)
        #tau (defaults to .1). ~decay_parameter sets the rate at which these representations decay over time. The higher the value, the more strongly an event propagates in time. s   (0.1 ~ an effect only affects the subsequent trial) 1 ~ exponential decay over the next memory_buffer trials; 100 ~ almost perfect propagation over next 10 trials). currently, only has a single memory parameter for all events
        # memory_buffer ... number of trials over which to propagate an event. default to 10.

        #returns mem_dict which is a trial by trial representation of memories of past trials


        # initialize container
        priorHx_dict = {}

        # get master ev_df for a single trial type
        ev_df_full = self.ev_df_master.query('type=="RESPONSE"')


        def feat2mem(feat_vec, tau, memory_buffer):
            # this sub function converts a feature vector to a memory representation

            # create an exponential window that propagates a representation forward in time
            # memory buffer sets the length of the window
            # tau sets the exponential decay
            window = signal.windows.exponential(M=memory_buffer,center=0,tau=tau,sym=False)

            # convolve w exponential function (return full so we can clip it manually below)
            mem_vec_full = signal.convolve(in1 = feat_vec,\
                in2 = window,mode='full')

            # clip the end 
            mem_vec_full = mem_vec_full[:-(memory_buffer-1)]

            # shift forward by 1 trial so that your feature vec (plot to make sure it is working)
            mem_vec =np.concatenate((np.zeros(shape=1),mem_vec_full[:-1]))



            # TEMP (PLOT)
            #plt.figure()
            #plt.plot(feat_vec)
            #plt.plot(mem_vec,color = 'C1',alpha=.5)

            return mem_vec

        # get event_dict based on continguous event
        def get_mem_dict(ev_df_full,tau,memory_buffer):
            # This subfunction extracts key events from a set of continuous events, it runs an exponenetial decay filter based on tau and memory buffer params and shifts index forward by one trial (so we have a memory of past trials, rather than a record of what happened on the current trial). 

            # initialize event dict
            mem_dict = {}

            # errors (too fast)
            errors_fast = ev_df_full.eval('RT<0').to_numpy().astype('int')
            mem_dict['errorMemFast'] = feat2mem(feat_vec=errors_fast,tau=tau,memory_buffer=memory_buffer)

            # errors (too slow)
            errors_slow = ev_df_full.eval('RT>1000').to_numpy().astype('int')
            mem_dict['errorMemSlow'] = feat2mem(feat_vec=errors_slow,tau=tau,memory_buffer=memory_buffer)

            # fast responses (~ pos feedback)
            fastResponse = ev_df_full['fastResponse'].to_numpy().astype('int')
            mem_dict['fastResponseMem'] = feat2mem(feat_vec=fastResponse,tau=tau,memory_buffer=memory_buffer)

            # slow responses (~ weak, neg feedback)
            slowResponse = ev_df_full.eval('(fastResponse==0)&(error==0)&RT>600').to_numpy().astype('int')
            mem_dict['slowResponseMem'] = feat2mem(feat_vec=slowResponse,tau=tau,memory_buffer=memory_buffer)

            # med responses (~ neutral feedback)
            medResponse = ev_df_full.eval('(RT>300)&(RT<600)').to_numpy().astype('int')
            mem_dict['medResponseMem'] = feat2mem(feat_vec=medResponse,tau=tau,memory_buffer=memory_buffer)


            # short delay
            shortDelay = ev_df_full.eval('delay==500').to_numpy().astype('int')
            mem_dict['shortDelayMem'] = feat2mem(feat_vec=shortDelay,tau=tau,memory_buffer=memory_buffer)


            # long delay
            longDelay = ev_df_full.eval('(delay==1500)|(delay==1000)').to_numpy().astype('int')
            mem_dict['longDelayMem'] = feat2mem(feat_vec=longDelay,tau=tau,memory_buffer=memory_buffer)

            return mem_dict
        #
        def find_matching_trials(mem_dict,trialNums_to_include):
            #This subfunction filters memory dict by a subset of trials. Assumes filtered by session

            # convert trial nums into a trial indec
            t_idx = trialNums_to_include-1

            key_list = mem_dict.keys()

            for k in key_list:
                mem_dict[k] = mem_dict[k][t_idx]

            return mem_dict

        def cat_mem_dict(mem_dict,mem_dict_ss):
            # This subfunction concatenates this session's data into the running count for the entire subject

            # if mem_dict is empty, just assign the session dict
            if len(mem_dict.keys()) == 0:
                mem_dict = mem_dict_ss
            else:
                key_list = mem_dict.keys()
                

                # loop through and concatenate 
                for k in key_list:
                    mem_dict[k] = np.concatenate((mem_dict[k],mem_dict_ss[k]))

            return mem_dict

        def zscore_mem_dict(mem_dict):
            # zscore all arrays (Only do this within session, and only for trials that have been queried (ev_df, not ev_df_full). This is the same way that zrrt is calculated)

            key_list = mem_dict.keys()

            for k in key_list:

                if np.std(mem_dict_ss[k]) !=0:# this is to avoid div by 0
                    mem_dict[k] = stats.zscore(mem_dict[k])

            return mem_dict
            

        # loop through sessions
        sess_list = np.unique(ev_df_full['session']) 

        # container for full feat vec
        mem_dict = {}

        for ss in sess_list:

            # get feature vector 
            mem_dict_ss = get_mem_dict(ev_df_full.query('session=="'+ss+'"'),tau=tau,memory_buffer=memory_buffer)

            # only return matching trials for the events that are selected such that each field in mem_dict should have the same length as ev_df
            mem_dict_ss = find_matching_trials(mem_dict_ss,trialNums_to_include=ev_df.query('session=="'+ss+'"')['trial'].to_numpy())


            # calc delay conflict before z-scoring
            delay_conf = 1-np.abs(mem_dict_ss['longDelayMem']-mem_dict_ss['shortDelayMem'])

            # zscore within session (only queried trials)
            mem_dict_ss = zscore_mem_dict(mem_dict_ss)

            # include delay conflict (but not z-scored)
            mem_dict_ss['delayMem_conflict'] = delay_conf



            # add delay condition, trial number and zrrt (zscore of -1/RT) for regresion. Use ev_df here (not ev_df full so we have matching trials. Also note that we are z-scoring only queried RTs within session)
            mem_dict_ss['delayCondIsLong'] = ev_df.query('session=="'+ss+'"').eval('delay>500').to_numpy().astype('int')

            mem_dict_ss['trialNum'] = ev_df.query('session=="'+ss+'"')['trial'].to_numpy().astype('int')

            mem_dict_ss['zrrt'] = stats.zscore(-1/ev_df.query('session=="'+ss+'"')['RT'].to_numpy())

            # get feature vectors that were used to come up with memory representations (for illustrative purposes)
            mem_dict_ss['errorEv'] = ev_df.query('session=="'+ss+'"')['error'].to_numpy().astype('int')

            mem_dict_ss['fastResponseEv'] = ev_df.query('session=="'+ss+'"')['fastResponse'].to_numpy().astype('int')


            # concatenate with full mem_dict representation
            mem_dict = cat_mem_dict(mem_dict,mem_dict_ss)


        # return
        return mem_dict


    # Regress RT 
    def fitMemoryRegression(self,ev_df = [],evQuery = 'error==0&fastResponse==0',decay_model = 'best',print_results=False,slow_drift_sigma=5):
        # This function runs a regression to identify the factors contirbuting to stochastic variability in RTs (any effects beyond delay condition). It also returns residual z-scored rts that reflect endogenous fluctuations in RT that are not task related. 'slow_drift_sigma' is a free parameter that sets the width of the gaussian kernel (in trials)used to identify slow drifts in RT variability


        # clear ev_df_to_
        if hasattr(self,'ev_df_to_regress'):
            ev_df_to_regress = []


        # if events are not provided, query events
        if type(ev_df)==list:
            if ev_df == []: 
                ev_df = self.ev_df.query('type=="RESPONSE"')

                if evQuery!=None:
                    ev_df = ev_df.query(evQuery)

        # fit LATER model to get measures of systemic variability in RTs
        # 500 delay vs >500 delay (combines 1000 and 1500 in few subjects that have it)
        self.fitLATER_byCondition(rts_A =ev_df.query('delay==500')['RT'].to_numpy(), rts_B = ev_df.query('delay>500')['RT'].to_numpy(),model_type = 'best')

        # initialize vectors for each model parameter. (and update them with model parameter)
        delayCondIsLong_later_rise = np.zeros(len(ev_df))
        delayCondIsLong_later_bound = np.zeros(len(ev_df))
        delayCondIsLong_later_std  = np.zeros(len(ev_df))

        # update long delay trials with teh model parameters (reference point is the short delay RT distribution which is 0)
        delayCondIsLong_later_rise[ev_df.eval('delay>500').to_numpy()] = self.laterByCond_dict['paramsDiff_rise_rate']
        delayCondIsLong_later_bound[ev_df.eval('delay>500').to_numpy()] = self.laterByCond_dict['paramsDiff_rise_bound']
        delayCondIsLong_later_std[ev_df.eval('delay>500').to_numpy()] = self.laterByCond_dict['paramsDiff_rise_std']

        # cache these events in self
        self.ev_df_to_regress = ev_df.copy()

        # update self.ev_df_to_regress with these data (so that they can be used in regression below)
        self.ev_df_to_regress.insert(loc = len(self.ev_df_to_regress.columns),column='delayCondIsLong_later_rise',value=delayCondIsLong_later_rise)
        self.ev_df_to_regress.insert(loc = len(self.ev_df_to_regress.columns),column='delayCondIsLong_later_bound',value=delayCondIsLong_later_bound)
        self.ev_df_to_regress.insert(loc = len(self.ev_df_to_regress.columns),column='delayCondIsLong_later_std',value=delayCondIsLong_later_std)


        # add column to ev_df with parameters. We are changing the sign of the parameters so that they 

        # subfunction to retun err_ associated with a particular tau
        def getMemRegressErr(tau,storeInSelf = False,print_results = False,slow_drift_sigma=5):
            # assumes fixed memory_buffer as 10 trials
            memory_buffer = 10

            # get mem_dict
            mem_dict = self.getPriorTrialMemory(ev_df=self.ev_df_to_regress,tau = tau,memory_buffer=memory_buffer)

            # updaet mem_dict with LATER model fits
            mem_dict['delayCondIsLong_later_rise'] = self.ev_df_to_regress['delayCondIsLong_later_rise'].to_numpy()
            mem_dict['delayCondIsLong_later_bound'] = self.ev_df_to_regress['delayCondIsLong_later_bound'].to_numpy()
            mem_dict['delayCondIsLong_later_std'] = self.ev_df_to_regress['delayCondIsLong_later_std'].to_numpy()

            # parse best fitting LATER model
            if self.laterByCond_dict['model_type_B'] == 'mean_bound':
                later_mod_str = '+ delayCondIsLong_later_rise + delayCondIsLong_later_bound'
            elif self.laterByCond_dict['model_type_B'] == 'mean_std':
                later_mod_str = '+ delayCondIsLong_later_rise + delayCondIsLong_later_std'
            elif self.laterByCond_dict['model_type_B'] == 'mean':
                later_mod_str = '+ delayCondIsLong_later_rise'
            elif self.laterByCond_dict['model_type_B'] == 'bound':
                later_mod_str = '+ delayCondIsLong_later_bound'
            elif self.laterByCond_dict['model_type_B'] == 'std':
                later_mod_str = '+ delayCondIsLong_later_std'
            elif self.laterByCond_dict['model_type_B'] == 'null':
                later_mod_str = ''


            # run regression 
            mem_reg = smf.ols('zrrt ~delayCondIsLong + trialNum +  shortDelayMem ',data = mem_dict).fit()

            # other parameters (not including any more):
            #errorMemFast + errorMemSlow + fastResponseMem + delayMem_conflict

            # not including later fit here
            #+str(later_mod_str),
      
            # other parameter that I am not including: 
            #+ longDelay, delay_conflict (these are highly correlated with short Delay)
            #+ slowResponse + medResponse (concern for correlation)

            # return error function 
            #err_ = mem_reg.bic
            # neg log.liklehood
            err_ = (-1*mem_reg.llf)

            #store in self (optional)
            if storeInSelf == True:
                self.memReg_dict = {}
                self.memReg_dict.update(mem_dict)
                # store residuals
                self.memReg_dict['zrrt_resid'] = mem_reg.resid.to_numpy()

                # decompose into slow drifts and fast fluctuations
                # gaussian smoothing with sigma of 5 trials
                self.memReg_dict['zrrt_resid_slow'] = ndimage.gaussian_filter1d(self.memReg_dict['zrrt_resid'],sigma=slow_drift_sigma)
                self.memReg_dict['zrrt_resid_fast'] = self.memReg_dict['zrrt_resid']-self.memReg_dict['zrrt_resid_slow']



                #predicted values
                self.memReg_dict['zrrt_pred'] = mem_reg.model.predict(mem_reg.params)   
                # model performance
                self.memReg_dict['LLE'] = mem_reg.llf
                self.memReg_dict['bic'] = mem_reg.bic
                self.memReg_dict['rsquared'] = mem_reg.rsquared
                self.memReg_dict['rsquared_adj'] = mem_reg.rsquared_adj
                # model params
                self.memReg_dict['tau'] = tau
                self.memReg_dict['memory_buffer'] = memory_buffer


                var_list = mem_reg.model.exog_names[1:]
                self.memReg_dict['var_list'] = var_list
                for v in var_list:
                    self.memReg_dict[v+'_tstat'] = mem_reg.tvalues[v]
                    self.memReg_dict[v+'_pvalues'] = mem_reg.pvalues[v]
                    self.memReg_dict[v+'_beta'] = mem_reg.params[v]
                # add additional later model vars (for convenience)
                if ('delayCondIsLong_later_rise' in var_list)==False:
                    v = 'delayCondIsLong_later_rise'
                    self.memReg_dict[v+'_tstat'] = np.nan
                    self.memReg_dict[v+'_pvalues'] = np.nan
                    self.memReg_dict[v+'_beta'] = np.nan
                elif ('delayCondIsLong_later_bound' in var_list)==False:
                    v = 'delayCondIsLong_later_bound'
                    self.memReg_dict[v+'_tstat'] = np.nan
                    self.memReg_dict[v+'_pvalues'] = np.nan
                    self.memReg_dict[v+'_beta'] = np.nan
                elif ('delayCondIsLong_later_std' in var_list)==False:
                    v = 'delayCondIsLong_later_std'
                    self.memReg_dict[v+'_tstat'] = np.nan
                    self.memReg_dict[v+'_pvalues'] = np.nan
                    self.memReg_dict[v+'_beta'] = np.nan
            if print_results == True:
                print('tau: ', tau)
                print(mem_reg.summary())

            return err_

        # parse decay model option
        if decay_model=='best':
            # Minimize error function
            res = optimize.minimize_scalar(getMemRegressErr,method='Bounded',bounds=(0.1,10))
            tau_best = res.x

        elif decay_model=='default':
            tau_best = 0.1

        # create memReg_dict
        getMemRegressErr(tau = tau_best,storeInSelf = True,print_results=print_results)
        self.memReg_dict['decay_model'] = decay_model

        # return dict
        return self.memReg_dict



    # plot trial by trial data - Subject version.
    # Loops through sessions and plots trial by trial data
    def plot_TrialByTrial(self,evQuery = None, ax=None,
    figsize =(7,7), markersize=6,legend_fsize = 6):
        # input ax must have length sess_List

        #default fig_params
        fig_params = {'figsize':figsize,
                      'markersize':markersize,
                         'legend_fontsize':legend_fsize}

        if ax == None:
            fig = figure(figsize=(fig_params['figsize'][0]*(len(self.sess_list)+1),fig_params['figsize'][1]))


        #loop through sesslist
        for sess_idx in arange(0,len(self.sess_list)):

            # create Sess obj
            ss = Session(self.subj,sess_idx = sess_idx, paramsDict = self.params)

            # create ax
            if ax == None:
                thisAx = fig.add_subplot(1,len(self.sess_list)+1,sess_idx+1)
            else:
                thisAx = ax[sess_idx]

            # plot trial by trial
            ss.plot_TrialByTrial(evQuery = evQuery,ax=thisAx,fig_params_dict = fig_params)

    # basic function of violin plot
    def vplot(self,x,y,ylabel,cond_lbl,do_paired=False):
        f = plt.figure()
        ax = plt.subplot(111)
        vp = ax.violinplot((x,y))
        for i in np.arange(0,len(vp['bodies'])):
            vp['bodies'][i].set_facecolor('0.5')
            vp['bodies'][i].set_edgecolor('k')
            vp['bodies'][i].set_alpha(0.5)
        vp['cmins'].set_color('k')
        vp['cbars'].set_color('k')
        vp['cmaxes'].set_color('k')


        #paired ttest
        if do_paired == False:
            tstat,pval = stats.ttest_ind(x,y,equal_var=False)
        else:
            tstat,pval = stats.ttest_1samp(x-y,popmean=0)
        fstat,pval_anov = stats.f_oneway(x,y)
        ax.set_ylabel(ylabel)
        ax.set_xticks([1,2])
        ax.set_xticklabels(cond_lbl)
        ax.set_title('indepedent t stat:'+str(np.round(tstat,2))+\
                     '; pval:'+str(np.round(pval,2))+\
                     '\n f stat:'+str(np.round(fstat,2))+\
                     '; pval:'+str(np.round(pval_anov,2)))

    # basic function to make a scatter plot and plot a line of best fit
    def plot_scatter(self,x,y,ax = None,color = '0.5',plotLine=True,polyfit=False,pThresh = 0.05,use_spearman = False,remove_zeros = False,s=None,alpha=0.5,text_lbls = None,fsize_text=12,cmap='viridis',figsize = (7,5),text_offset_x=0, text_offset_y = 0,edgecolor = 'k'):
        # general plotting function to plot a scatter plot for two variables and fit a line. Returns x and y


        #remove nans
        rm_bool = (np.isnan(x)) | (np.isnan(y)) | (np.isinf(x)) | (np.isinf(y)) 
        x = x[rm_bool==False]
        y = y[rm_bool==False]

        if remove_zeros == True:
            z_bool = (x==0) | (y==0)
            x = x[z_bool==False]
            y = y[z_bool==False]

        # x vs. y
        if ax == None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)


        plt.scatter(x,y,c=color,edgecolor =edgecolor,alpha = alpha,s = s,cmap=cmap)

        # if we have a list
        if (text_lbls is None)==False:
            # remove labels assoc w bad values
            text_lbls = text_lbls[rm_bool==False]

            if remove_zeros == True:
                text_lbls=text_lbls[z_bool==False]

            for i in range(0,len(text_lbls)):
                ax.text(x[i]+text_offset_x,y[i]+text_offset_y,text_lbls[i],fontsize = fsize_text,alpha=0.5,color='k')

        if plotLine==True:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

            if use_spearman == True:
                r_value, p_value = stats.spearmanr(x,y)
                #print('spearman r',r_value)

            if polyfit == True:
                polymod= np.poly1d(np.polyfit(x, y, 3))
                x_mod = np.linspace(np.min(x),np.max(x),20)
                if p_value < pThresh:
                   plt.plot(x_mod,polymod(x_mod),'r', linestyle='dashed',alpha=0.5)
            else:
                if p_value < pThresh:
                    x_plot = np.linspace(np.min(x),np.max(x),2)
                    plt.plot(x_plot, intercept + (slope*np.array(x_plot)), 'r', linestyle='dashed',alpha=0.5)
            plt.title('r = '+str(np.round(r_value,2))+' p ='+str(np.round(p_value,3)))

            
            return x,y
    def plotScat(self,x,y,
                 xlbl = None,
                 ylbl = None,
                 ax = None,msize=5,alpha=.5,
                 mcolor=None,marker='o',
                 linewidths=1.5,edgecolors=None,
                 fit_linestyle ='--',
                 fit_linewidth = 2,
                 fit_linecolor = 'r',
                 fit_alpha = 0.4,
                 plotLegend=True,
                 lbl_fsize = 10,
                 leg_fsize = 6):
        # makes a scatter plot, plots a line of best fit
        # Inputs:
        # x = var 1 (array)
        # y = var 2 (array)
        # ax = axis to plot data
        # msize = marker size (scalar)
        # alpha = transparency
        # mcolor = color of marker
        # marker = marker to plot
        # linewidths = width of edge
        # edgecolor = default is no edge color, specify a color to plot an edge

        if ax == None:
            fig = figure(figsize=(5,5))
            ax = subplot(111)

        # plot scatter
        ax.scatter(x,y,s=msize,c=mcolor,marker=marker,alpha=alpha,
                   linewidths=linewidths,edgecolors=edgecolors)

        # find line of best fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

        # plot line of best fit
        ax.plot(x,((slope*x)+intercept),
                linestyle = fit_linestyle,
                linewidth = fit_linewidth,
                color = fit_linecolor,
                alpha = fit_alpha,
                label = 'r = '+str(np.round(r_value,3))+', p = '+str(np.round(p_value,3)))

        # set x and ylble
        if xlbl!=None:
            ax.set_xlabel(xlbl,fontsize=lbl_fsize)
        if ylbl!=None:
            ax.set_ylabel(ylbl,fontsize=lbl_fsize)

        # plot legend
        if plotLegend==True:
            ax.legend(fontsize=leg_fsize)

        ax.axis('tight')

    # getRTs
    def getRTs(self, evQuery = None, rt_dist_type = 'standard'):

        # This function returns rt values for all trials from a subject. It has the option to filter out particular trials, and apply various transforms on the rt distribution

        #evQuery .. how to filter events

        #rt_dist_type
        #            'standard' ... no transform       
        #            'reciprocal'... -1/rt
        #            'zrrt'...z-score (-1/RT) (as used in later analyses,
        #             no need to invert axes as high RTs are on the right
        #            'reciprobit'...cum probability vs. 1/rt

        # filter by choice ev
        choiceEv = self.ev_df.query('type=="RESPONSE"')


        # additional filter (e.g., error trials==0)
        if evQuery!=None:
            choiceEv = choiceEv.query(evQuery)


        # parse rt dist type
        if rt_dist_type == 'standard':
            rts = choiceEv['RT'].to_numpy()
        elif rt_dist_type == 'reciprocal':
            rts = -1./choiceEv['RT'].to_numpy().astype('float')
        elif rt_dist_type == 'zrrt':
            rts = stats.zscore(-1./choiceEv['RT'].to_numpy().astype('float'))


        return rts

    # getReciprobitFits
    def getReciprobitFits(self,evQuery = None):

        # This function performs fits to a reciprobit transform on RT data (see Carpenter et al) and returns parameters (slope, median, intercept)

        # filter by choice ev
        choiceEv = self.ev_df.query('type=="RESPONSE"')


        # additional filter (e.g., error trials==0)
        if evQuery!=None:
            choiceEv = choiceEv.query(evQuery)

        # x values are sorted RT data
        rt_sort = np.sort(choiceEv['RT'].to_numpy().astype('float'))

        # y values are cumulative probabilities (cumulative sum of x values that are normalized)
        cum_prob = np.cumsum(rt_sort)/np.sum(rt_sort)

        # convert cum_prob to probit scale (inverse of CDF)
        cum_prob_probit = stats.norm.ppf(cum_prob);

        # fit line to cumulative probabilities
        # [-1] indexing to ignore the value associated with inf (cum_prob = 1)
        # -1/rt_sort ensures that longer RTs are associated with higher values, so the slope should have a positive value
        slope, intercept, r_value, p_value, std_err = stats.linregress(-1/rt_sort[:-1],cum_prob_probit[:-1])
        r_sq = r_value**2

        # calculate rt median - rt value associated with median of transformed distribution
        rt_median = np.median(choiceEv['RT'].to_numpy())

        return slope, intercept,rt_median, std_err, r_sq

    # LATER model Fits
    def fitLATER(self,rts=[],evQuery = 'error==0&fastResponse==0', model_type = 'mean_std', B0 = [], M0 = [], S0 = []):
        # This function obtains LATER fits to RT distributions. Modelled based on code from Josh Gold https://github.com/TheGoldLab/Lab_Matlab_Utilities/blob/master/reciprobit/reciprobit_fit.m. Updated to include more flexibility in fitting Gaussian parameters and competing LATER units

        # inputs:
        # rts ... distribution of RTs to model (if None will grab RTs from choice Events)
        # evQuery (optional)...to filter events. Is only applied if rts == None
        # model_type.. how to fit the model. 
            #'mean'... one free param (mean)
            #'std'... one free param (std only)
            #'bound'... one free param (bound..yolked changes in mean and std)
            #'mean_std'...two free parameters (mean and std vary indepednetly)
            # 'mean_bound'...two free parameters (mean and bound)

        # returns
        #rate_of_rise
        #bound
        #error (negative log likelhood)
        # num_free_params


        # if no RTs are given, get RTs from choice Events
        if len(rts)==0:
            # filter by choice ev
            choiceEv = self.ev_df.query('type=="RESPONSE"')


            # additional filter (e.g., error trials==0)
            if evQuery!=None:
                choiceEv = choiceEv.query(evQuery)

            rts = choiceEv['RT'].to_numpy().astype('float')

        # transform to reciprocal rts in seconds (no negative sign here)
        rrts_to_fit = 1000./rts

        # hold this in self so you can access it in the error function
        self.rrts_to_fit = rrts_to_fit
        self.rts_to_fit_raw = rts

        # guess starting parameters based on rrt distribution
        #M0...starting value of rate of rise param
        #B0...starting value for bound param (distance)
        #S0... starting value for std param
        if B0==[]:
            # make initial guess based on STD if we are fitting mean_bound model
            if model_type == 'mean_bound':
                B0 = 1/np.std(self.rrts_to_fit)
            else:
                B0 = 1
            self.LATER_B0 = B0 

        if M0==[]:
            M0 = np.mean(self.rrts_to_fit)*B0
            self.LATER_M0 = M0 

        if S0==[]:
            # set initial STD to 1 if we are fitting mean_bound model
            if model_type == 'mean_bound':
                S0 = 1
            else:
                S0 = np.std(self.rrts_to_fit)
            self.LATER_S0 = S0 
        # define model (error) functions. Separte model for each model type
        def laterModel_null():
            #params... no additional parameters. Just returns the fit associated with the params stored in self

            # it fits the cached rrts in self as "self.rrts_to_fit"

            # initialize normal distribution for these parameters
            norm = stats.norm(loc = self.LATER_M0/self.LATER_B0,scale = self.LATER_S0/self.LATER_B0)

            # error function to calculate negative log likelhood for this distribution
            # err_ = -sum(log(normpdf(rrts, fits(1)/fits(2), 1/fits(2))));
            err_ = -np.sum(np.log(norm.pdf(self.rrts_to_fit)))

            if np.isinf(err_):
                err_ = 10**6



            return err_            

        # define model (error) functions. Separte model for each model type
        def laterModel_mean(mean_rise):
            #params... tuple repesenting model parameters. Expects one free parameter
            #1.... mean of rate of rise
            #2.... distance (~bound height or starting point)

            # it fits the cached rrts in self as "self.rrts_to_fit"

            # initialize normal distribution for these parameters
            
            norm = stats.norm(loc = mean_rise/self.LATER_B0,scale = self.LATER_S0/self.LATER_B0)


            # error function to calculate negative log likelhood for this distribution
            # err_ = -sum(log(normpdf(rrts, fits(1)/fits(2), 1/fits(2))));
            err_ = -np.sum(np.log(norm.pdf(self.rrts_to_fit)))

            if np.isinf(err_):
                err_ = 10**6



            return err_

        # define model (error) functions. Separte model for each model type
        def laterModel_bound(bound):
            #params... tuple repesenting model parameters. Expects one free parameter
            #1.... distance (~bound height or starting point)

            # it fits the cached rrts in self as "self.rrts_to_fit"
            # skip this iteration if bound = 0 (bc division by 0)
            if bound ==0:
                err_ = 10**6
            else:
                # initialize normal distribution for these parameters
                norm = stats.norm(loc = self.LATER_M0/bound,scale = self.LATER_S0/bound)

                # error function to calculate negative log likelhood for this distribution
                # err_ = -sum(log(normpdf(rrts, fits(1)/fits(2), 1/fits(2))));
                err_ = -np.sum(np.log(norm.pdf(self.rrts_to_fit)))

                if np.isinf(err_):
                    err_ = 10**6
            return err_

        # define model (error) functions. Separte model for each model type
        def laterModel_std(std_rise):
            #params... tuple repesenting model parameters. Expects one free parameter
            #1.... std of rate of rise

            # it fits the cached rrts in self as "self.rrts_to_fit"

            # initialize normal distribution for these parameters
            norm = stats.norm(loc = self.LATER_M0/self.LATER_B0,scale = std_rise/self.LATER_B0)

            # error function to calculate negative log likelhood for this distribution
            # err_ = -sum(log(normpdf(rrts, fits(1)/fits(2), 1/fits(2))));
            err_ = -np.sum(np.log(norm.pdf(self.rrts_to_fit)))

            if np.isinf(err_):
                err_ = 10**6
            return err_
        def laterModel_mean_bound(params):
            #params... tuple repesenting model parameters. Expects two parameters
            #1.... mean of rate of rise
            #2.... distance (~bound height or starting point)

            # it fits the cached rrts in self as "self.rrts_to_fit"
            if params[1] ==0:
                err_ = 10**6
            else:
                # initialize normal distribution for these parameters
                norm = stats.norm(loc = params[0]/params[1],scale = 1/params[1])

                # error function to calculate negative log likelhood for this distribution
                # err_ = -sum(log(normpdf(rrts, fits(1)/fits(2), 1/fits(2))));
                err_ = -np.sum(np.log(norm.pdf(self.rrts_to_fit)))

                if np.isinf(err_):
                    err_ = 10**6

                return err_

        def laterModel_mean_std(params):
            #params... tuple repesenting model parameters. Expects two parameters
            #1.... mean of rate of rise
            #2.... standard deviation of rate of rise

            # it fits the cached rrts in self as "self.rrts_to_fit"

            # initialize normal distribution for these parameters
            norm = stats.norm(loc = params[0]/self.LATER_B0,scale = params[1]/self.LATER_B0)

            # error function to calculate negative log likelhood for this distribution
            # err_ = -sum(log(normpdf(rrts, fits(1)/fits(2), 1/fits(2))));
            err_ = -np.sum(np.log(norm.pdf(self.rrts_to_fit)))


            if np.isinf(err_):
                err_ = 10**6

            return err_

        def laterModel_mean_bound_std(params):
            #params... tuple repesenting model parameters. Expects two parameters
            #1.... mean of rate of rise
            #2.... bound (~distance)
            #3.... standard deviation of rate of rise

            # it fits the cached rrts in self as "self.rrts_to_fit"
            if params[1]==0:
                err_ = 10**6
            else:
                # initialize normal distribution for these parameters
                norm = stats.norm(loc = params[0]/params[1],scale = params[2]/params[1])

                # error function to calculate negative log likelhood for this distribution
                # err_ = -sum(log(normpdf(rrts, fits(1)/fits(2), 1/fits(2))));
                err_ = -np.sum(np.log(norm.pdf(self.rrts_to_fit)))


                if np.isinf(err_):
                    err_ = 10**6

            return err_


        # parse model type and run model function to minimize error
        if model_type == 'mean':


            # Minimize error function
            res = optimize.minimize_scalar(laterModel_mean,method='Golden',bracket=(0.001,1000))

            # update fits for this model type
            self.LATER_rise_rate = res.x
            self.LATER_rise_std = self.LATER_S0
            self.LATER_rise_bound = self.LATER_B0
            self.LATER_num_params = 1
        elif model_type == 'null':

            res = laterModel_null()

            # null model that has no additional parameters
            self.LATER_rise_rate = self.LATER_M0
            self.LATER_rise_std = self.LATER_S0
            self.LATER_rise_bound = self.LATER_B0
            self.LATER_num_params = 0

        elif model_type == 'std':


            # Minimize error function
            res = optimize.minimize_scalar(laterModel_std,method='Golden',bracket=((0.001,1000)))
            # update fits for this model type
            self.LATER_rise_rate = self.LATER_M0
            self.LATER_rise_std = res.x
            self.LATER_rise_bound = self.LATER_B0
            self.LATER_num_params = 1

        elif model_type =='bound':


            # Minimize error function
            res = optimize.minimize_scalar(laterModel_bound,method='Golden',bracket=((0.001,1000)))
            # update fits for this model type
            self.LATER_rise_rate = self.LATER_M0
            self.LATER_rise_std = self.LATER_S0
            self.LATER_rise_bound = res.x
            self.LATER_num_params = 1


        elif model_type == 'mean_bound':

            # Minimize error function
            res = optimize.minimize(laterModel_mean_bound,x0=(M0,B0),method='SLSQP',bounds=((0.001,1000),(0.001,1000)))
            # update fits for this model type
            self.LATER_rise_rate = res.x[0]
            self.LATER_rise_std = self.LATER_S0
            self.LATER_rise_bound = res.x[1]
            self.LATER_num_params = 2


        elif model_type == 'mean_std':

            # Minimize error function
            res = optimize.minimize(laterModel_mean_std,x0=(M0,S0),method='SLSQP',bounds=((0.001,1000),(0.001,1000)))
            # update fits for this model type
            self.LATER_rise_rate = res.x[0]
            self.LATER_rise_std = res.x[1]
            self.LATER_rise_bound = self.LATER_B0
            self.LATER_num_params = 2

        elif model_type == 'mean_bound_std':

            # Minimize error function
            res = optimize.minimize(laterModel_mean_bound_std,x0=(M0,B0,S0),method='SLSQP',bounds=((0.001,1000),(0.001,1000),(0.001,1000)))
            # update fits for this model type
            self.LATER_rise_rate = res.x[0]
            self.LATER_rise_bound = res.x[1]
            self.LATER_rise_std = res.x[2]
            self.LATER_num_params = 3

        # save LLE
        if model_type == 'null':
            self.LATER_lle = res
        else:
            self.LATER_lle = res.fun
        self.LATER_model_type=model_type

        # note we are using log-likelhoods for BIC (not negative log likelhoods). Better fit = higher log liklehood
        self.LATER_bic = sm.tools.eval_measures.bic(llf=-1*self.LATER_lle,\
            nobs = len(self.rrts_to_fit),df_modelwc = self.LATER_num_params)


        # calculate r-squared 
        # estimate probabilty distribution function of rrts
        ho = np.histogram(self.rrts_to_fit,density=True,bins=10)  
        y_true=ho[0]
        x=ho[1]

        # create normal dist object based on params
        norm = stats.norm(loc = self.LATER_rise_rate/self.LATER_rise_bound,\
                  scale = self.LATER_rise_std/self.LATER_rise_bound)
        y_pred = norm.pdf(x[:-1])
        self.LATER_r2score = r2_score(y_true,y_pred)


        # return later fits
        return (self.LATER_rise_rate,self.LATER_rise_bound,self.LATER_rise_std),self.LATER_lle, self.LATER_bic

    def fitLATER_byCondition(self,rts_A=[], rts_B=[],evQuery = 'error==0&fastResponse==0', model_type = 'best',model_type_list=[]):
        # This function fits the LATER model to two RT distributions to study the algorithmic correlates of systemic RT variability. It first first the LATER model to the first distribution, and uses those parameters

        # rts_A ...rts for condition A
        # rts_B ... rts for condition B
        # evQuery ... filter to query the data if no RTs are given
        # model_type .. free parameters


        # if no RTs are given, get RTs from choice Events as short and long delay condition
        if (len(rts_A)==0)&(len(rts_B)==0):
            # filter by choice ev
            choiceEv = self.ev_df.query('type=="RESPONSE"')

            # additional filter (e.g., error trials==0)
            if evQuery!=None:
                choiceEv_A = choiceEv.query(evQuery+'&delay==500')
                choiceEv_B = choiceEv.query(evQuery+'&delay==1500')
            else:
                choiceEv_A = choiceEv.query('delay==500')
                choiceEv_B = choiceEv.query('delay==1500')

            rts_A = choiceEv_A['RT'].to_numpy().astype('float')
            rts_B = choiceEv_B['RT'].to_numpy().astype('float')

        # initialize delay condition dict 
        self.laterByCond_dict = {'rts_A':rts_A,'rts_B':rts_B}


        # Fit first RT distribution using two parameter model
        self.fitLATER(rts=rts_A, model_type = 'mean_std', B0 = [], M0 = [], S0 = [])

        # get fits from first dist
        self.laterByCond_dict['rrtsA']=self.rrts_to_fit
        self.laterByCond_dict['paramsA_rise_rate'] = self.LATER_rise_rate  
        self.laterByCond_dict['paramsA_rise_bound'] = self.LATER_rise_bound
        self.laterByCond_dict['paramsA_rise_std'] = self.LATER_rise_std 
        self.laterByCond_dict['lle_A'] = self.LATER_lle
        self.laterByCond_dict['bic_A'] = self.LATER_bic
        self.laterByCond_dict['model_type_A'] = self.LATER_model_type
        self.laterByCond_dict['r2score_A'] = self.LATER_r2score


        # for second distribution, start with the parameters from the first distribution
        if model_type=='best':

            # parse model_type list

            if model_type_list == []:
                model_type_list = ['null','mean','bound','std','mean_bound','mean_std']

            #'mean_bound','mean_std']
            #,'mean_bound_std'

            # model_dict (to collect data from each fit)
            model_dict_list = []

            for m in model_type_list:

                # initialize model_dict for this model fit
                model_dict={}

                # fit later to second RT distribution using model type and starting parameters from previous fit
                self.fitLATER(rts=rts_B, model_type = m, B0 = self.laterByCond_dict['paramsA_rise_bound'], M0 = self.laterByCond_dict['paramsA_rise_rate'], S0 = self.laterByCond_dict['paramsA_rise_std'])

                # collect data in model_dict
                model_dict['rrtsB'] = self.rrts_to_fit
                model_dict['paramsB_rise_rate'] = self.LATER_rise_rate
                model_dict['paramsB_rise_bound'] = self.LATER_rise_bound
                model_dict['paramsB_rise_std'] = self.LATER_rise_std
                model_dict['lle_B'] = self.LATER_lle
                model_dict['bic_B'] = self.LATER_bic
                model_dict['model_type_B'] = self.LATER_model_type
                model_dict['r2score_B'] = self.LATER_r2score

                # add to model_dict_list
                model_dict_list.append(model_dict)

            # convert do df
            model_df = pd.DataFrame(model_dict_list)

            #choose model with lowest bic
            best_model_idx = np.argmin(model_df['bic_B'].to_numpy()) 

            # append data from best fit model to model A
            self.laterByCond_dict.update(model_dict_list[best_model_idx])

        # calculate difference params (B - A)
        self.laterByCond_dict['paramsDiff_rise_rate'] = self.laterByCond_dict['paramsB_rise_rate'] - self.laterByCond_dict['paramsA_rise_rate']
        self.laterByCond_dict['paramsDiff_rise_bound'] = self.laterByCond_dict['paramsB_rise_bound'] - self.laterByCond_dict['paramsA_rise_bound']
        self.laterByCond_dict['paramsDiff_rise_std'] = self.laterByCond_dict['paramsB_rise_std'] - self.laterByCond_dict['paramsA_rise_std']

    def getRTs_for_LATER2(self,cond_a_str='delay==500',cond_b_str = 'delay==1500',premature_RT_threshold_ms = 250, fastest_rt_ms = -500, slowest_rt_ms = 1000,cond_a_offset_ms = 1000):
        # Returns RTS for later 2 model. expects that events are not filtered using evQuery.
        choiceEv = self.ev_df.query('type=="RESPONSE"')

        # returns rts as expected by fitLATER2_by condition
        rts_A = (choiceEv.query(cond_a_str+'&RT>@fastest_rt_ms&RT<@slowest_rt_ms')['RT_targ']).to_numpy()+cond_a_offset_ms
        rts_B = (choiceEv.query(cond_b_str+'&RT>@fastest_rt_ms&RT<@slowest_rt_ms')['RT_targ']).to_numpy()

        premature_RT_threshold_ms = 250
        pred_idx_A = choiceEv.query(cond_a_str+'&RT>@fastest_rt_ms&RT<@slowest_rt_ms').eval('RT<@premature_RT_threshold_ms').to_numpy()
        pred_idx_B = choiceEv.query(cond_b_str+'&RT>@fastest_rt_ms&RT<@slowest_rt_ms').eval('RT<@premature_RT_threshold_ms').to_numpy()

        return rts_A,rts_B,pred_idx_A,pred_idx_B



    def fitLATER2_byCondition(self,rts_A,rts_B,pred_idx_A,pred_idx_B,fastest_correct_response_ms = 1750, slowest_correct_response_ms = 2500,fastest_premature_response_ms = 1000,model_type = 'std_bias'):
        """
        Fit the hybrid LATER model to two RT distributions from different conditions (e.g., short and long foreperiod delay).

        Note: this function expects a very specific representation of RTs. RTs must be provided in ms and cannot be negative. To fit premature responses (express errors), add an offset such that these errors are associated with postitive RTs. Additionally, the RTs from both conditions must reflec the same offset from stimulus onset. For example, if condition A is a 500 ms foreperiod delay condition, and condition B is a 1500 ms foreperiod delay condition, rts_A should indicate response times from trial start + 1000 ms, and rts_B should indicate response times from trial start. use getRTs_for_LATER2 to return these RTs from unfiltered ev_df.

        rts_A ...array, rts in ms from condition A (see note above)
        rts_B ...array, rts in ms from condition B (see note above)
        pred_idx_A ...bool, indicates trials where premature responses on condition A (this can include pre-stimulus responses and responses made very quickly after stimulus onset, e.g. 250 ms)
        pred_idx_B ...bool, indicates trials where premature responses occrred on condition B
        fastest_correct_response_ms ... indicates fastest correct response in ms (to aid in constraining parameter search)
        slowest_correct_response_ms ... indicates slowest correct response in ms (to aid in constraining parameter search)

        Model description:


        The proposed model is based on the idea that the brain has parallel systems for prediction and reaction that each contribute to responses on stimulus detection tasks. We make the following assumptions

        (1) We assume that these systems work independently and in parallel because we cannot assess conflict between these processes in the current task.

        (2) We assume that the prediction system begins generating responses from trial start, whereas the reaction system begins generating responses after stimulus onset. 

        (3) We model response times (and prediction times) using the LATER framework that makes  several simplifying assumptions about the process leading up to response. We can conceptually think of the prediction unit as integrating expectation (belief) representations, whereas the reaction unit integrates sensory representations. Noise in each of these representations can be modeled independently

        Strategy: Plot reciprocal response times relative to target onset. First fit a gaussian to the premature responses (prediction times). Then fit a gaussian to residual response times
        """

        # convert rts to reciprocal rts in seconds. converting to seconds helps interpret gaussian parameters


        # process RTs 
        # convert to reciprocals
        rrts_A = 1000/rts_A
        rrts_B = 1000/rts_B
    
        # identify prediction times (for initial fits)
        rpts_A = rrts_A[pred_idx_A]
        rpts_B = rrts_B[pred_idx_B]


        # define subfunctions
        def later_mean_std(params,rrts):
            # fits a basic rate of rise distribution to a Gaussian
            #params... tuple repesenting model parameters. 
            #0.... mean of rate of rise (for response unit)
            #1.... standard deviation of rate of rise (for response unit)

            #Inputs
            #rrts...reciprocal Rts to fit
            
            # initialize gaussian object
            norm = stats.norm(loc = params[0],scale = params[1])

            # error function to calculate negative log likelhood for this distribution
            err_ = -np.sum(np.log(norm.pdf(rrts)))

            if np.isinf(err_):
                err_ = 10**6

            return err_
        def later_mean_std_bias(params,rrts,Mp,Sp):
            # fits a rate of rise distribution to rrts and a bias towards premature responses defined by a separate gaussian of mean Mp and std Sp
            #params... tuple repesenting model parameters. 
            #0.... mean of rate of rise (for response unit)
            #1.... standard deviation of rate of rise (for response unit)
            #2.... bias towards premature responses

            #Inputs
            #rrts...reciprocal Rts to fit
            # Mp ... mean of error dist 
            # Sp ... std of error dist

            # initialize gaussian object
            norm_react = stats.norm(loc = params[0],scale = params[1])
            norm_pred = stats.norm(loc = Mp,scale = Sp)


            # weigthed probability distribution 
            prob = ((1-params[2])*norm_react.pdf(rrts)) + ((params[2])*norm_pred.pdf(rrts))
        
            # error function to calculate negative log likelhood for this distribution.
            err_ = -np.sum(np.log(prob))
        
            if np.isinf(err_):
                err_ = 10**6

            return err_

        def later_mean_bias(params,rrts,S0,Mp,Sp):
            # fits a rate of rise distribution to rrts and a bias towards premature responses defined by a separate gaussian of mean Mp and std Sp. std of the reaction unit is fixed (S0) 
            #params... tuple repesenting model parameters. 
            #0.... mean of rate of rise (for response unit)
            #1.... bias towards premature responses

            #Inputs
            #rrts...reciprocal Rts to fit
            # S0 ... std of reaction unit
            # Mp ... mean of error dist 
            # Sp ... std of error dist

            # initialize gaussian object
            norm_react = stats.norm(loc = params[0],scale = S0)
            norm_pred = stats.norm(loc = Mp,scale = Sp)


            # weigthed probability distribution 
            prob = ((1-params[1])*norm_react.pdf(rrts)) + ((params[1])*norm_pred.pdf(rrts))
        
            # error function to calculate negative log likelhood for this distribution.
            err_ = -np.sum(np.log(prob))
        
            if np.isinf(err_):
                err_ = 10**6

            return err_
        def later_distance_bias(params,rrts,M0,S0,Mp,Sp):
            # fits a rate of rise distribution to rrts and a bias towards premature responses defined by a separate gaussian of mean Mp and std Sp.  Mean (M0) and std of the reaction unit are fixed (S0) 
            #params... tuple repesenting model parameters. 
            #0.... distance for reaction unit
            #1.... bias towards premature responses

            #Inputs
            #rrts...reciprocal Rts to fit
            # M0 ... mean of reaction unit
            # S0 ... std of reaction unit
            # Mp ... mean of error dist 
            # Sp ... std of error dist

            # initialize gaussian object
            norm_react = stats.norm(loc = M0/params[0],scale = S0/params[0])
            norm_pred = stats.norm(loc = Mp,scale = Sp)


            # weigthed probability distribution 
            prob = ((1-params[1])*norm_react.pdf(rrts)) + ((params[1])*norm_pred.pdf(rrts))
        
            # error function to calculate negative log likelhood for this distribution.
            err_ = -np.sum(np.log(prob))
        
            if np.isinf(err_):
                err_ = 10**6

            return err_
        def later_std_bias(params,rrts,M0,Mp,Sp):
            # fits a rate of rise distribution to rrts and a bias towards premature responses defined by a separate gaussian of mean Mp and std Sp.  Mean (M0) and std of the reaction unit are fixed (S0) 
            #params... tuple repesenting model parameters. 
            #0.... std for reaction unit
            #1.... bias towards premature responses

            #Inputs
            #rrts...reciprocal Rts to fit
            # M0 ... mean of reaction unit
            # Mp ... mean of error dist 
            # Sp ... std of error dist

            # initialize gaussian object
            norm_react = stats.norm(loc = M0,scale = params[0])
            norm_pred = stats.norm(loc = Mp,scale = Sp)


            # weigthed probability distribution 
            prob = ((1-params[1])*norm_react.pdf(rrts)) + ((params[1])*norm_pred.pdf(rrts))
        
            # error function to calculate negative log likelhood for this distribution.
            err_ = -np.sum(np.log(prob))
        
            if np.isinf(err_):
                err_ = 10**6

            return err_


        def later_mean_distance_bias(params,rrts,S0,Mp,Sp):
            # fits a rate of rise distribution to rrts and a bias towards premature responses defined by a separate gaussian of mean Mp and std Sp
            #params... tuple repesenting model parameters. 
            #0.... mean of rate of rise (for response unit)
            #1.... distance travelled (scaling factor for simultaneous change in mean and std) (for response unit)
            #2.... bias towards premature responses

            #Inputs
            #rrts...reciprocal Rts to fit
            # S0 ... std of reaction unit
            # Mp ... mean of error dist 
            # Sp ... std of error dist
            # initialize gaussian object
            norm_react = stats.norm(loc = params[0]/params[1],scale = S0/params[1])
            norm_pred = stats.norm(loc = Mp,scale = Sp)


            # weigthed probability distribution 
            prob = ((1-params[2])*norm_react.pdf(rrts)) + ((params[2])*norm_pred.pdf(rrts))
        
            # error function to calculate negative log likelhood for this distribution.
            err_ = -np.sum(np.log(prob))
        
            if np.isinf(err_):
                err_ = 10**6

            return err_

        def later_std_distance_bias(params,rrts,M0,Mp,Sp):
            # fits a rate of rise distribution to rrts and a bias towards premature responses defined by a separate gaussian of mean Mp and std Sp
            #params... tuple repesenting model parameters. 
            #0.... std of rate of rise (for response unit)
            #1.... distance travelled (scaling factor for simultaneous change in mean and std) (for response unit)
            #2.... bias towards premature responses

            #Inputs
            #rrts...reciprocal Rts to fit
            # M0 ... mean of reaction unit
            # Mp ... mean of error dist 
            # Sp ... std of error dist
            # initialize gaussian object
            norm_react = stats.norm(loc = M0/params[1],scale = params[0]/params[1])
            norm_pred = stats.norm(loc = Mp,scale = Sp)


            # weigthed probability distribution 
            prob = ((1-params[2])*norm_react.pdf(rrts)) + ((params[2])*norm_pred.pdf(rrts))
        
            # error function to calculate negative log likelhood for this distribution.
            err_ = -np.sum(np.log(prob))
        
            if np.isinf(err_):
                err_ = 10**6

            return err_
        ### STEP 1: Initial fit of rate of rise distribution associated with premature responses (prediction unit responses).2 free parameters. Mp = mean of rate of rise associated with premature responses, Sp = std of rate of ris distribution for premature responses. Assumes that this is the same distribution across both conditions. 

        # estimate this by fitting all premature responses
        rpts = np.concatenate((rpts_A,rpts_B))

        # initial guess of Mp0 and Sp0 if there are at least 5 premature responses total
        if len(rpts)>5:
            Mp0 = np.mean(rpts)
            Sp0 = np.std(rpts)
        else:
            Mp0 = 1000/fastest_correct_response_ms
            Sp0 = 0.1
        # bounds for Mp and Sp
        # contrain Mp within range of reciprocal premature responses
        #Mp_bounds = (0.4,1)
        Mp_bounds =(1000/fastest_correct_response_ms,1000/fastest_premature_response_ms)
        # constrain std so we dont have run-away variance as a description of bi-modal RTs        
        Sp_bounds = (0.001,0.1)

        # get M0 and Sp0 based on correct responses from condition A
        res = optimize.minimize(later_mean_std,x0=(Mp0,Sp0),\
                            args=(rpts),method='SLSQP',\
                            bounds=(Mp_bounds,Sp_bounds))
        Mp = res.x[0]
        Sp = res.x[1]


        ### STEP 2: Fit of responses on condition A. 3 free parameters.  M_A = mean of rate of rise distribution, S_A = std of rate of rise distribution, bias = bias towards rate of rise distribution (using fixed Mp and Sp parameters from previous step)

        # initial guess of M0 and S0 (based on correct responses on condition A
        M0 = np.mean(rrts_A[pred_idx_A==False]) 
        S0 = np.std(rrts_A[pred_idx_A==False])
        bias0 = 0.001 # start assuming minimal influence of premature responses

        # bounds for M and S
        # contrain M within rate of reciprocal correct responses
        M_bounds = (1000/slowest_correct_response_ms,1000/fastest_correct_response_ms) 
        # constrain std so we dont have run-away variance as a description of bi-modal RTs
        S_bounds = (0.001,0.1)

        # constrain bias from 0 to 1
        bias_bounds = (0.001,1)

        # get M_A, S_A and bias_A based on all responses from condition A. Mp and Sp are fixed parameters from step 1
        res = optimize.minimize(later_mean_std_bias,x0=(M0,S0,bias0),\
                            args=(rrts_A,Mp,Sp),method='SLSQP',\
                            bounds=(M_bounds,S_bounds,bias_bounds))
        M_A = res.x[0]
        S_A = res.x[1]
        bias_A = res.x[2]
        # Distance A is 1 (as a reference value)
        D_A = 1



        ### STEP 3: Fit responses on condition B. 3 free parameters.  M_B = mean of rate of rise distribution, bias_B bias towards rate of rise distribution (using fixed Mp and Sp parameters from previous step), and one of the following:
        # S_B = std of rate of rise 
        # or 
        # D_B = distance travelled (scaling factor on mean and std)

        # initial guess of parameters underlying long delay responses essentially assume a null model
        M0 =  M_A # start with gaussian fit on short delay trials
        S0 = S_A # start with gaussian fit on short delay trials
        D0 = D_A # assume no change in distance
        bias0 = bias_A# start assuming minimal influence of premature responses

        # bounds for M and S
        # contrain M within rate of reciprocal correct responses
        M_bounds =(0.001,1) #(1000/slowest_correct_response_ms,1000/fastest_correct_response_ms) 

        # constrain std so we dont have run-away variance as a description of bi-modal RTs
        S_bounds = (0.001,0.1)

        # constrain distance so we dont divide by 0
        D_bounds =(.1,10) #(0.001,1000)

        # constrain bias from 0 to 1
        bias_bounds = (0.001,1)


        if model_type == 'mean_bias':

            # get M_A, S_A and bias_A based on all responses from condition A. S0, Mp and Sp are fixed parameters from step 1
            res = optimize.minimize(later_mean_bias,x0=(M0,bias0),\
                               args=(rrts_B,S0,Mp,Sp),method='SLSQP',\
                               bounds=(M_bounds,bias_bounds))
            M_B = res.x[0] # mean 
            bias_B = res.x[1] # bias on condition B

            S_B = S_A # assume std does not change
            D_B = D_A # assume distance does not change
        elif model_type == 'std_bias':

            # get M_A, S_A and bias_A based on all responses from condition A. S0, Mp and Sp are fixed parameters from step 1
            res = optimize.minimize(later_std_bias,x0=(S0,bias0),\
                               args=(rrts_B,M0,Mp,Sp),method='SLSQP',\
                               bounds=(S_bounds,bias_bounds))
            S_B = res.x[0] # std on condition B 
            bias_B = res.x[1] # bias on condition B

            M_B = M_A # assume mean does not change
            D_B = D_A # assume distance does not change        
        elif model_type == 'dist_bias':
            # get M_A, S_A and bias_A based on all responses from condition A. M0, S0, Mp and Sp are fixed parameters from step 1
            res = optimize.minimize(later_distance_bias,x0=(D0,bias0),\
                               args=(rrts_B,M0,S0,Mp,Sp),method='SLSQP',\
                               bounds=(D_bounds,bias_bounds))

            D_B = res.x[0]
            bias_B = res.x[1]

            M_B = M_A # assume mean does not change
            S_B = S_A # assume std does not change
        elif model_type == 'mean_std_bias':

            # get M_A, S_A and bias_A based on all responses from condition A. Mp and Sp are fixed parameters from step 1
            res = optimize.minimize(later_mean_std_bias,x0=(M0,S0,bias0),\
                               args=(rrts_B,Mp,Sp),method='SLSQP',\
                               bounds=(M_bounds,S_bounds,bias_bounds))
            M_B = res.x[0] # mean 
            S_B = res.x[1] # std on condition B
            bias_B = res.x[2] # bias on condition B

            # assume S_A is the same as S_B. This model assumes that stochastic variance is constant between conditions, but that changes in rt variance between conditions is driven by changes in premature bias and distance
            D_B = 1 # assume distance does not change

        elif model_type == 'mean_dist_bias':

            # get M_A, S_A and bias_A based on all responses from condition A. Mp and Sp are fixed parameters from step 1
            res = optimize.minimize(later_mean_distance_bias,x0=(M0,D0,bias0),\
                               args=(rrts_B,S0,Mp,Sp),method='SLSQP',\
                               bounds=(M_bounds,D_bounds,bias_bounds))
            M_B = res.x[0] # mean 
            D_B = res.x[1] # distance on condition B
            bias_B = res.x[2] # bias on condition B

            # assume S_A is the same as S_B. This model assumes that stochastic variance is constant between conditions, but that changes in rt variance between conditions is driven by changes in premature bias and distance
            S_B = S_A
        elif model_type == 'std_dist_bias':

            # get M_A, S_A and bias_A based on all responses from condition A. Mp and Sp are fixed parameters from step 1
            res = optimize.minimize(later_std_distance_bias,x0=(S0,D0,bias0),\
                               args=(rrts_B,M0,Mp,Sp),method='SLSQP',\
                               bounds=(S_bounds,D_bounds,bias_bounds))
            S_B = res.x[0] # std 
            D_B = res.x[1] # distance on condition B
            bias_B = res.x[2] # bias on condition B

            # assume S_A is the same as S_B. This model assumes that stochastic variance is constant between conditions, but that changes in rt variance between conditions is driven by changes in premature bias and distance
            M_B = M_A # assume mean does not change

        # store data in later2_dict dict
        later2_dict = {}
        later2_dict['model_type'] = model_type
        later2_dict['rrts_A'] = rrts_A
        later2_dict['rrts_B'] = rrts_B
        later2_dict['rrts_premature'] = rpts
        later2_dict['pred_idx_A'] = pred_idx_A
        later2_dict['pred_idx_B'] = pred_idx_B
        later2_dict['Mp'] = Mp
        later2_dict['Sp'] = Sp
        later2_dict['M_A'] = M_A
        later2_dict['S_A'] = S_A
        later2_dict['D_A'] = D_A
        later2_dict['bias_A'] = bias_A
        later2_dict['M_B'] = M_B
        later2_dict['S_B'] = S_B
        later2_dict['D_B'] = D_B
        later2_dict['bias_B'] = bias_B

        # calculate params_diff (only mean, distance and bias are free to change between conditions)
        later2_dict['paramsDiff_M'] = M_B - M_A
        later2_dict['paramsDiff_S'] = S_B - S_A
        later2_dict['paramsDiff_D'] = D_B - D_A
        later2_dict['paramsDiff_B'] = bias_B - bias_A


        # calculate rsquare for full distribution
        later2_dict = self.calcLATER2_rsquared(later2_dict,rrt_range = (0.4,1), smoothing_factor = 10)

        # update later2_dict in self
        self.later2_dict = later2_dict

        return later2_dict

    def calcLATER2_rsquared(self, later2_dict,rrt_range = (0.4,1), smoothing_factor = 10,plot_it = False,ax = None):
        # This function applies the full later 2 model that has already been fit and tests predictions vs. the full reciprocal rt distribution

        # inputs
        # range ... # reciprocal rt range of 0.4 to 1 is the range over which we are testing predictions of the probability distribution. Default 0.4 to 1, which maps on to 1000 ms to 2500 ms from target onset, or premature responses 500 ms prior to stimulus (which we can sample for both delay conditions)
        #smoothing_factor ... sets the std of the gaussian kernel used to smooth RTs and estimate a probability distribution function associated with all reciprocal Rts


        # returns
        #later2_dict with updated r-sq values


        # get reciprocal rts for both groups
        rrts = np.concatenate((later2_dict['rrts_A'],later2_dict['rrts_B']))

        # estimate empirical probabilty distribution function of all rrts
        y_true,xvals = np.histogram(rrts,bins=1000,range = rrt_range, density=True)
        y_true = ndimage.gaussian_filter1d(y_true,smoothing_factor)
        xvals=xvals[:-1]


        # create predicted probability distribution based on full model
        norm_pred = stats.norm(loc = later2_dict['Mp'],scale = later2_dict['Sp'])
        norm_react_A = stats.norm(loc = later2_dict['M_A']/later2_dict['D_A'],scale = later2_dict['S_A']/later2_dict['D_A'])
        norm_react_B = stats.norm(loc = later2_dict['M_B']/later2_dict['D_B'],scale = later2_dict['S_B']/later2_dict['D_B'])


        # predicted probabilities based on full model. it weights long and short probability distributions based on the relative number of trials in each condition
        condB_bias = len(later2_dict['rrts_B'])/(len(later2_dict['rrts_A'])+len(later2_dict['rrts_B']))


        y_pred_A = ((1-later2_dict['bias_A'])*norm_react_A.pdf(xvals)) + ((later2_dict['bias_A'])*norm_pred.pdf(xvals))
        y_pred_B = ((1-later2_dict['bias_B'])*norm_react_A.pdf(xvals)) + ((later2_dict['bias_B'])*norm_pred.pdf(xvals))
        y_pred_full = ((1-condB_bias)*y_pred_A) + ((condB_bias)*y_pred_B)

        # update later2 dict
        later2_dict['rsquared'] = r2_score(stats.zscore(y_true),stats.zscore(y_pred_full))
        later2_dict['rsquared_rrt_range'] = rrt_range        
        later2_dict['rsquared_smoothing_factor'] = smoothing_factor

        # plot full fit (z-scored probability distributions)
        if plot_it == True:
            if ax is None:
                f = plt.figure()
                ax = plt.subplot(111)
            ax.plot(xvals,stats.zscore(y_true),color='k')
            ax.plot(xvals,stats.zscore(y_pred_full),color='k',alpha=0.5,linewidth=3,linestyle='--')
            ax.set_title('$R^2 =${x:.2f}, $\Delta$ M = {M:.2f},\n  $\Delta$ D = {D:.2f}, $\Delta$ B = {B:.2f}'.format(x=later2_dict['rsquared'],M=later2_dict['paramsDiff_M'],D=later2_dict['paramsDiff_D'],B=later2_dict['paramsDiff_B']))
            ax.axis('tight')
        return later2_dict

    def plotLATER2_fits(self,ax_list = None,later2_dict = None,figsize=(12,3),fsize_lbl = 12,stim_on_ms = 1500,fastResp_ms = 1750,invert_xaxis=True):
        """plots later 2 fits. 1x4 plot. Premature responses, short delay resonses, lond delay responses, full fit  """
        # stim_on_ms = 1500 indicates the response time aligned to stim onset (to plot vertical line)
        # fastResp_ms = 1500 indicates the response time aligned to fast response threshold (below which we consider premature responses (again, to plot vertical line)

        # ax 
        if later2_dict is None:
            later2_dict = self.later2_dict

        if ax_list is None:
           f = plt.figure(figsize=figsize)
           ax_P = plt.subplot(141)
           ax_A = plt.subplot(142)
           ax_B = plt.subplot(143)
           ax_R = plt.subplot(144)
        else:
            ax_P = ax_list[0]
            ax_A = ax_list[1]
            ax_B = ax_list[2]
            ax_R = ax_list[3]


        # subfunctions
        def plotPredicted(ax,loc,scale):
            xvals = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],1000)
            norm = stats.norm(loc = loc,scale = scale)

            #ax2 = ax.twinx()
            ax.plot(xvals,norm.pdf(xvals),color='0.5',linewidth=3,linestyle='--')
            ax.set_yticks([])


        def plotPredictedWeighted(ax,M,S,D,bias,Mp,Sp):
            xvals = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],1000)
            norm_react = stats.norm(loc = M/D,scale = S/D)
            norm_pred = stats.norm(loc = Mp,scale = Sp)

            # pdf
            prob = ((1-bias)*norm_react.pdf(xvals)) + ((bias)*norm_pred.pdf(xvals))

            #ax2 = ax.twinx()
            ax.plot(xvals,prob,color='0.5',linewidth=3,linestyle='--')

            ax.set_yticks([])


        ## plot distributions

        # RRTS on premature responses with predicted fit
        #rrts_premature = np.concatenate((later2_dict['rrts_A'][later2_dict['pred_idx_A']],later2_dict['rrts_B'][later2_dict['pred_idx_B']]))

        ax2_P = ax_P.twinx()
        ho = np.histogram(later2_dict['rrts_premature'],density=False,bins=1000)
        x_P=ho[1][:-1]
        ax_P.plot(x_P,ho[0],'r',alpha=0.5)
        ax_P.set_xlim(later2_dict['rsquared_rrt_range'])  

        plotPredicted(ax2_P,later2_dict['Mp'],later2_dict['Sp'])
        plt.gca().set_title('Mp = {Mp:.2f},Sp = {Sp:.2f}'.format(Mp =later2_dict['Mp'], Sp = later2_dict['Sp']),fontsize = fsize_lbl)
        plt.gca().set_xlabel('1/RT (from target)',fontsize = fsize_lbl)

        # plot v-line indicating stimulus onset
        plt.gca().vlines((1000/stim_on_ms,1000/fastResp_ms),plt.gca().get_ylim()[0],plt.gca().get_ylim()[1],linestyle='--',color = ('red','green'))

        # invert x axis (so fast RTs are on right)
        if invert_xaxis==True:
            ax_P.invert_xaxis()
        ax_P.set_xlabel('1/RT',fontsize=fsize_lbl)
        ax_P.set_ylabel('Count',fontsize=fsize_lbl)


        # RRTS on condition A
        ax2_A = ax_A.twinx()
        ho = np.histogram(later2_dict['rrts_A'],density=False,bins=1000)
        x_A=ho[1][:-1]
        ax_A.plot(x_A,ho[0],'C0',alpha=0.5)
        ax_A.set_xlim(later2_dict['rsquared_rrt_range']) 

        plotPredictedWeighted(ax2_A,later2_dict['M_A'],later2_dict['S_A'],later2_dict['D_A'],later2_dict['bias_A'],later2_dict['Mp'],later2_dict['Sp'])
        plt.gca().set_title('M_A = {M:.2f}, S_A = {S:.2f}, \n D_A = {D:.2f}, bias_A = {bias:.2f}'.format(M =later2_dict['M_A'], S= later2_dict['S_A'], D= later2_dict['D_A'], bias= later2_dict['bias_A']),fontsize = fsize_lbl)
        ax2_A.set_xlabel('1/RT (from target)',fontsize = fsize_lbl)

        # plot v-line indicating stimulus onset
        ax2_A.vlines((1000/stim_on_ms,1000/fastResp_ms),plt.gca().get_ylim()[0],plt.gca().get_ylim()[1],linestyle='--',color = ('red','green'))

        # invert x axis (so fast RTs are on right)
        if invert_xaxis==True:
            ax2_A.invert_xaxis()

        ax_A.set_xlabel('1/RT',fontsize=fsize_lbl)
        ax_A.set_ylabel('Count',fontsize=fsize_lbl)
        # RRTS on condition B

        ax2_B = ax_B.twinx()
        ho = np.histogram(later2_dict['rrts_B'],density=False,bins=1000)
        x_B=ho[1][:-1]
        ax_B.plot(x_B,ho[0],'C1',alpha=0.5)
        ax_B.set_xlim(later2_dict['rsquared_rrt_range'])  

        plotPredictedWeighted(ax2_B,later2_dict['M_B'],later2_dict['S_B'],later2_dict['D_B'],later2_dict['bias_B'],later2_dict['Mp'],later2_dict['Sp'])
        plt.gca().set_title('M_B = {M:.2f}, S_B = {S:.2f}, \n D_B = {D:.2f}, bias_B = {bias:.2f}'.format(M =later2_dict['M_B'], S= later2_dict['S_B'], D= later2_dict['D_B'], bias= later2_dict['bias_B']),fontsize = fsize_lbl)
        ax2_B.set_xlabel('1/RT (from target)',fontsize = fsize_lbl)

        # plot v-line indicating stimulus onset
        ax2_B.vlines((1000/stim_on_ms,1000/fastResp_ms),plt.gca().get_ylim()[0],plt.gca().get_ylim()[1],linestyle='--',color = ('red','green'))

        # invert x axis (so fast RTs are on right)
        if invert_xaxis==True:
            ax2_B.invert_xaxis()
        ax_B.set_xlabel('1/RT',fontsize=fsize_lbl)
        ax_B.set_ylabel('Count',fontsize=fsize_lbl)
        ### overall fit to the rrts probabilites with rsquared and change in parameters
        self.calcLATER2_rsquared(later2_dict, ax = ax_R, rrt_range = (0.4,1), smoothing_factor = 10,plot_it = True)
        ax_R.vlines((1000/stim_on_ms,1000/fastResp_ms),ax_R.get_ylim()[0],ax_R.get_ylim()[1],linestyle='--',color = ('red','green'))

        if invert_xaxis==True:
            ax_R.invert_xaxis()
        ax_R.set_xlabel('1/RT',fontsize=fsize_lbl)
        ax_R.set_ylabel('$z$ score',fontsize=fsize_lbl)        

        plt.tight_layout()








    #basic function to plot RT
    def plotRT(self, evQuery = 'error==0&fastResponse==0', ax = None,plot_type = 'standard', bins = 40, alpha = 1,label = None,plot_median = False,color='C0',apply_reciprobit_smooth = False,plot_reciprobit_line=True, reciprobit_sd = 10,yL=None,model_type = 'best',later_params = [],rrts_to_fit=[],rts_to_fit_raw=[],xval_pdf=None,model_color=None):
        # Note: this funciton doesnt set the axes for the RT plot. Run set_axes_rt afterwards
        # Inputs:
        #evQuery .. how to filter events
        #ax .. axes to plot on
        #plot_type ..'standard'... plots standard rts
        #            'reciprocal'...-1/rt
        #            'zrrt'...z-score (-1/RT) (as used in later analyses)
        #            for reciprocal and zrrt, no need to invert axes as high RTs are on the right
        #            'reciprobit'...cum probability vs. 1/rt
        #bins......  number of bins for hist plots
        #alpha.....transperency of histrogram
        #label..... label of distribution
        #plot_median = False,  ... option to plot median verticla line
        #color='C0',... option to set color
        #apply_reciprobit_smooth = True, ... option to smooth the x-axis of the reciprobit plot 
        #reciprobit_sd = 10,... option to adjust the std. of gaussian used to smooth the reciprobit plot 
        #yL=None,.... option to set the ylim
        #model_type = 'mean_bound', .... option to set the type of LATER model to fit
        #later_params = [] .... option to provide the parameters of the later model to plot. Expects a tuple as (LATER_rise_rate,LATER_rise_bound,LATER_rise_std), if empty, it fits LATER independently to this rt distribution. 



        # parse fig,ax
        if ax is None:
            fig = figure(figsize=(5,5))
            ax = subplot(111)

        # filter by choice ev
        choiceEv = self.ev_df.query('type=="RESPONSE"')


        # additional filter (e.g., error trials)
        if evQuery!=None:
            choiceEv = choiceEv.query(evQuery)

        # plot RT dist for various formats
        if plot_type == 'standard': # plot standard RT
            h = ax.hist(choiceEv['RT'].to_numpy(), bins = bins, alpha = alpha,label = label,color = color,edgecolor='k')
            col = h[2][0].get_facecolor()
            med_rt = np.median(choiceEv['RT'].to_numpy())
        elif plot_type == 'targOn':
            h = ax.hist(choiceEv['RT_targ'].to_numpy(), bins = bins, alpha = alpha,label = label,color = color,edgecolor='k')
            col = h[2][0].get_facecolor()
            med_rt = np.median(choiceEv['RT_targ'].to_numpy())            

        elif plot_type == 'reciprocal': # plot distribution of reciprocal RT
            h = ax.hist(-1./choiceEv['RT'].to_numpy().astype('float'), bins = bins, alpha = alpha,label=label,color = color)
            col = h[2][0].get_facecolor()
            med_rt = np.median(-1./choiceEv['RT'].to_numpy().astype('float'))
        elif plot_type == 'targOn_reciprocal': # plot distribution of reciprocal RT
            h = ax.hist(-1./choiceEv['RT_targ'].to_numpy().astype('float'), bins = bins, alpha = alpha,label=label,color = color)
            col = h[2][0].get_facecolor()
            med_rt = np.median(-1./choiceEv['RT_targ'].to_numpy().astype('float'))


        elif plot_type =='zrrt':# z-score -1/RT
            h = ax.hist(stats.zscore(-1./choiceEv['RT'].to_numpy().astype('float')),bins = bins, alpha = alpha,label = label,color = color)
            col = h[2][0].get_facecolor()
            med_rt = np.median(stats.zscore(-1./choiceEv['RT'].to_numpy().astype('float')))
        elif plot_type == 'LATER':
            # plot reciprocal rts (that were fit) with overlying normal pdf
            # NOTE: we add negative signs to both the hist plot and the norm pdf plot because the model was fit to positive reciprocal RTs, not negative reciprocal RTs. This is easier than inverting the axes
            # fit LATER if params are not provided
            if later_params == []:
                self.fitLATER(evQuery = evQuery,model_type=model_type)
                later_params = (self.LATER_rise_rate,self.LATER_rise_bound,self.LATER_rise_std)
            if rrts_to_fit ==[]:
                rrts_to_fit = self.rrts_to_fit

            yl = ax.get_ylim()
            h = ax.hist((rrts_to_fit),bins = bins, density = False, alpha = alpha, label = label,color = color,edgecolor='k')
            # hack to make sure correct ylims are plotting when plotting by delay
            ax.set_ylim(yl[0],np.max((np.max(h[0]),yl[1])))

            # patches
            col = h[2][0].get_facecolor()

            # plot norm pdf
            norm = stats.norm(loc = (later_params[0]/later_params[1]), scale = later_params[2]/later_params[1])

            # new axes
            ax2 = ax.twinx()
            # h[1] is bins from hist
            #ax2.plot(h[1],norm.pdf(h[1]),color = col,linewidth=3)


            if xval_pdf is None:
                xvals = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],100)
            else:
                xvals = np.linspace(xval_pdf[0],xval_pdf[1],100)

            if model_color is None:
                model_color = col
            ax2.plot(xvals,norm.pdf(xvals),color = model_color,linewidth=3,alpha=0.5)

            # no ax2 ytick llabels
            ax2.set_yticks([])

            med_rt = np.median((rrts_to_fit))

        elif plot_type == 'LATER_reciprobit':
            # plot reciprocal rts (that were fit) with overlying normal pdf
            # NOTE: we add negative signs to both the hist plot and the norm pdf plot because the model was fit to positive reciprocal RTs, not negative reciprocal RTs. This is easier than inverting the axes
            # fit LATER
            if later_params == []:
                self.fitLATER(evQuery = evQuery,model_type=model_type)
                later_params = (self.LATER_rise_rate,self.LATER_rise_bound,self.LATER_rise_std)
            if rrts_to_fit ==[]:
                rrts_to_fit = self.rrts_to_fit
            if rts_to_fit_raw == []:
                rts_to_fit_raw = self.rts_to_fit_raw


            # plot reciprobit of rts that were fit by LATER
            # calculate x-values
            rts_sort = np.sort(rts_to_fit_raw)
            x_vals = -1/rts_sort

            #calculate y-values (see below)
            cum_prob = np.cumsum(rts_sort)/np.sum(rts_sort)
            y_vals = stats.norm.ppf(cum_prob);


            # plot model distribution on reciprobit axis

            # generate norm pdf (no negative sign; we want an exact match of the distribution that was fit)
            norm = stats.norm(loc = (later_params[0]/later_params[1]), scale = later_params[2]/later_params[1])

            # calculate y_values for modelled distribution
            # get modelled RT d 
            #rts_sort_mod = np.sort(norm.pdf(rrts_to_fit))
            #rts_sort_mod = norm.cdf(rrts_to_fit)
            x_vals_mod = x_vals

            # calculate y-values for model distribution
            cum_prob_mod = np.cumsum(rts_sort_mod)/np.sum(rts_sort_mod)
            y_vals_mod = stats.norm.ppf(cum_prob_mod)

            # option to smooth the x-values
            if apply_reciprobit_smooth==True:
                x_vals = ndimage.filters.gaussian_filter1d(x_vals,sigma=reciprobit_sd) 
                x_vals_mod = ndimage.filters.gaussian_filter1d(x_vals_mod,sigma=reciprobit_sd) 

            # plot cumulative probabilities
            l = ax.plot(x_vals,y_vals,linestyle='-',alpha=alpha,label = label,color = color,linewidth=3)
            ax.plot(x_vals_mod,y_vals_mod,linestyle='--',alpha=alpha,label = label,color = color,linewidth=3)

            col = l[0].get_color()
            med_rt = np.median(x_vals)


        elif plot_type == 'reciprobit': # plot reciprobit plot
            # plot empirical cumulative distribution function of RT dist
            # x values are sorted RT data
            rt_sort = np.sort(choiceEv['RT'].to_numpy().astype('float'))
            x_vals = -1/rt_sort

            # y values are cumulative probabilities (cumulative sum of x values that are normalized)
            cum_prob = np.cumsum(rt_sort)/np.sum(rt_sort)

            # convert cum_prob to probit scale (inverse of CDF)
            cum_prob_probit = stats.norm.ppf(cum_prob);

            # option to smooth the x-values
            if apply_reciprobit_smooth==True:
                x_vals = ndimage.filters.gaussian_filter1d(x_vals,sigma=reciprobit_sd) 

            if plot_reciprobit_line == True:
                # pick 3 points (10, 50, 90 percentile). x_vals is already sorted so no need to run np.percentile function
                idx = np.array([int(.10*len(x_vals)),int(.25*len(x_vals)),int(.5*len(x_vals)),int(.75*len(x_vals)),int(.90*len(x_vals))])
                x_pts = x_vals[idx]
                y_pts = cum_prob_probit[idx]  

                #fit line
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_pts,y_pts)
                # plot reciprobit line (to y intercept)
                x_vals_plt = np.array(list(x_vals)+[0])
                l = ax.plot(x_vals_plt,intercept+(slope*x_vals_plt),linestyle='-',alpha=0.5,label = label,color = color,linewidth=3)
                ax.plot(x_pts,y_pts,'x',color = color)
        elif plot_type == 'targOn_reciprobit': # plot reciprobit plot
            # plot empirical cumulative distribution function of RT dist
            # x values are sorted RT data
            rt_sort = np.sort(choiceEv['RT_targ'].to_numpy().astype('float'))
            x_vals = -1/rt_sort

            # y values are cumulative probabilities (cumulative sum of x values that are normalized)
            cum_prob = np.cumsum(rt_sort)/np.sum(rt_sort)

            # convert cum_prob to probit scale (inverse of CDF)
            cum_prob_probit = stats.norm.ppf(cum_prob);

            # option to smooth the x-values
            if apply_reciprobit_smooth==True:
                x_vals = ndimage.filters.gaussian_filter1d(x_vals,sigma=reciprobit_sd) 

            if plot_reciprobit_line == True:
                # pick 3 points (10, 50, 90 percentile). x_vals is already sorted so no need to run np.percentile function
                idx = np.array([int(.10*len(x_vals)),int(.25*len(x_vals)),int(.5*len(x_vals)),int(.75*len(x_vals)),int(.90*len(x_vals))])
                x_pts = x_vals[idx]
                y_pts = cum_prob_probit[idx]  

                #fit line
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_pts,y_pts)
                # plot reciprobit line (to y intercept)
                x_vals_plt = np.array(list(x_vals)+[0])
                l = ax.plot(x_vals_plt,intercept+(slope*x_vals_plt),linestyle='-',alpha=0.5,label = label,color = color,linewidth=3)
                ax.plot(x_pts,y_pts,'x',color = color)
            else:
                # plot cumulative probabilities
                l = ax.plot(x_vals,cum_prob_probit,linestyle='-',alpha=0.5,label = label,color = color,linewidth=3)
               
            col = l[0].get_color()
            med_rt = np.median(x_vals)

        if yL!=None:
            ax.set_ylim(yL[0],yL[1])

        if plot_median== True:
            if plot_type == 'LATER':
                yl = ax.get_ylim()
                yl2 = ax2.get_ylim()
            else:
                yl = ylim()

            ax.vlines(x = med_rt,ymin = yl[0]*10, ymax = yl[1]*10,linestyles = 'dashed',alpha = 0.7,color = col)
            
            if plot_type == 'LATER':
                ax.set_ylim(yl)
                ax2.set_ylim(yl2)
            else:
                ylim(yl)

            

    # set axes for RT plot
    def set_axes_rt(self, ax,plot_type = 'standard',fig_params_dict = None,add_legend = True,ax2= None):
        # Input:
        #ax... axis on which RT distributions are plotted
        #plot_type ..'standard'... plots standard rts
        #            'reciprocal'...1/rt
        #            'reciprobit'...cum probability vs. 1/rt

        #default fig params
        fig_params = {'legend_fontsize':12,'fsize_tick':14}

        # update if needed
        if fig_params_dict != None:
            fig_params.update(fig_params_dict)

        # invert x-axis and label xticks with RT values
        #,'LATER'
        if plot_type in ['reciprocal','reciprobit','LATER_reciprobit','targOn_reciprocal','targOn_reciprobit']:
            # if reciprocal, or reciprobit RT, reverse x-axis (so that increasing RT values show up on the right side)
            #ax.invert_xaxis()
            #06/20 - no need to do this because added - sign to computation of both values

            # set xtick labels as 1/xtick values so that RT values are shown
            ax.set_xticklabels(labels = -1*np.round(1/ax.get_xticks(),1),fontsize=fig_params['fsize_tick'])

        elif plot_type in ['LATER']:


            ax.set_xticklabels(labels = ax.get_xticks().astype('int'),fontsize=fig_params['fsize_tick'])
        
            # invert axes (because we have not added the negative sign to the computation of rrts as in the section above)
            ax.invert_xaxis()
            # get second axis
            #if ax2 is None:
            #    plt.gcf().get_axes()[-1].invert_xaxis()


        if plot_type in ['standard','reciprocal','targOn']:
            # set labels
            ax.set_xlabel('RT (ms)')
            ax.set_ylabel('Count')
        elif plot_type == 'LATER':
            ax.set_xlabel('1/RT (1/s)')
            ax.set_ylabel('Count')


        elif plot_type == 'zrrt':
            ax.set_xlabel('z(-1/RT)')
            ax.set_ylabel('Count')
        elif plot_type in ['reciprobit','LATER_reciprobit']:
            # set labels
            ax.set_xlabel('RT (ms)')
            ax.set_ylabel('z-score Cumulative probability')

        # set label
        if add_legend == True:
            ax.legend(fontsize = fig_params['legend_fontsize'])

        # set tick size
        #ax.set_xticklabels(ax.get_xticklabels(),fontsize=fig_params['fsize_tick'])
        #ax.set_yticklabels(ax.get_yticklabels(),fontsize=fig_params['fsize_tick'])


    # plot RT distribution by delay
    # run function
    def plot_RT_by_delay(self,bins = 20, evQuery=None,label = None, plot_type = 'standard',fig_params_dict=None,ax = None,plot_median = True,yL = None,model_type = 'mean_bound',model_type_list=[],add_legend=True):
        # Inputs
        # ax ... must be of length two (left is full RT dist, right is by delay)
        # default fig_params
        fig_params={'figsize':(5,5),'label':evQuery,'title':'RT '+plot_type,'title_fontsize':15}

        # update fig_params
        if fig_params_dict!=None:
            fig_params.update(fig_params_dict)

        if ax == None:
            fig = figure(figsize=(fig_params['figsize'][0],fig_params['figsize'][1]))

        # set title
        #fig.suptitle(fig_params['title'],fontsize=fig_params['title_fontsize'])

        # # plot full RT
        # # create axes
        # if ax == None:
        #     thisAx = subplot(1,2,1)
        # else:
        #     thisAx = ax[0]

        # # plot RT
        # self.plotRT(evQuery = evQuery,bins=bins, ax = thisAx,plot_type = plot_type,label=fig_params['label'],alpha =0.5, plot_median = plot_median,yL = yL,model_type='mean_std')
        # self.set_axes_rt(ax=thisAx,plot_type = plot_type, fig_params_dict = fig_params_dict)

        # plot RT by delay
        # create axes
        if ax == None:
            # thisAx = subplot(1,2,2)
            thisAx = subplot(1,1,1)
        else:
            thisAx = ax

        # plot RT dist for delay trials
        delay_list = np.unique(self.ev_df['delay'].to_numpy())



        # containers for model parameters
        rise_rate = []
        rise_std = []
        rise_bound = [] 

        #fit model upfront 
        if model_type == 'best':
            self.fitLATER_byCondition(rts_A=[], rts_B=[],evQuery = evQuery, model_type = model_type,model_type_list=model_type_list)
            later_params_dict = {}
            later_params_dict['500_params']=(self.laterByCond_dict['paramsA_rise_rate'],self.laterByCond_dict['paramsA_rise_bound'],self.laterByCond_dict['paramsA_rise_std'])
            later_params_dict['1500_params']=(self.laterByCond_dict['paramsB_rise_rate'],self.laterByCond_dict['paramsB_rise_bound'],self.laterByCond_dict['paramsB_rise_std'])
            later_params_dict['500_rrts_to_fit']=self.laterByCond_dict['rrtsA']
            later_params_dict['1500_rrts_to_fit']=self.laterByCond_dict['rrtsB']
            later_params_dict['500_rts_to_fit_raw']=self.laterByCond_dict['rts_A']
            later_params_dict['1500_rts_to_fit_raw']=self.laterByCond_dict['rts_B']
            # calculate xvalues for pdf (that accounts for both reciprocal rt distributions)
            later_params_dict['xval_pdf'] = ((-.5+np.min(np.append(self.laterByCond_dict['rrtsA'],self.laterByCond_dict['rrtsB']))),(.5+np.max(np.append(self.laterByCond_dict['rrtsA'],self.laterByCond_dict['rrtsB']))))
        else: 
            later_params_dict ={}
            later_params_dict['500_params']=[]
            later_params_dict['1500_params']=[]
            later_params_dict['500_rrts_to_fit']=[]
            later_params_dict['1500_rrts_to_fit']=[]
            later_params_dict['500_rts_to_fit_raw']=[]
            later_params_dict['1500_rts_to_fit_raw']=[]
            later_params_dict['xval_pdf']=[] 
        # loop through delay conditions
        for i in np.arange(0,len(delay_list)):

            if delay_list[i]==500:
                color = 'C0'
            elif delay_list[i]==1000:
                if model_type == 'best':
                    continue
                    color = 'C2'
            elif delay_list[i]==1500:
                color = 'C1'

            # store thisDelay in self
            self.thisDelay = delay_list[i]



            # plot RTs based on filtered events with additional delay filter
            self.plotRT(evQuery = evQuery+'&(delay == @self.thisDelay)',bins=bins, ax = thisAx,plot_type = plot_type,alpha = 0.5,label = ('delay '+str(self.thisDelay)+' ms'),plot_median=plot_median,yL = yL,color = color,model_type=model_type, later_params=later_params_dict[str(delay_list[i])+'_params'],rrts_to_fit = later_params_dict[str(delay_list[i])+'_rrts_to_fit'],rts_to_fit_raw=later_params_dict[str(delay_list[i])+'_rts_to_fit_raw'],xval_pdf = later_params_dict['xval_pdf'])

            # collect model fits
            if plot_type in ['LATER','LATER_reciprobit']:
                rise_rate.append(np.round(self.LATER_rise_rate,2))
                rise_std.append(np.round(self.LATER_rise_std,2))
                rise_bound.append(np.round(self.LATER_rise_bound,2))


        # set axes
        self.set_axes_rt(ax=thisAx,plot_type = plot_type,fig_params_dict = fig_params_dict,add_legend=add_legend)

        # set title to list how model parameters
       # ax = plt.gca()
        if plot_type in ['LATER','LATER_reciprobit']:
            if model_type=='best':
                thisAx.set_title('rate: '+str(np.round(self.laterByCond_dict['paramsDiff_rise_rate'],2))+' bound: '+str(np.round(self.laterByCond_dict['paramsDiff_rise_bound'],2))+' std: '+str(np.round(self.laterByCond_dict['paramsDiff_rise_std'],2))
                +'\n '+self.laterByCond_dict['model_type_B'])
                #print(self.laterByCond_dict['paramsB_rise_rate'])

            else:
                thisAx.set_title('rate: '+str(rise_rate)+'\n bound: '+str(rise_bound)\
                +'\n std:'+str(rise_std))


class Session(Subject):
# Subclass. Will inherit attributes and funcitons from Subject. Will filter trials within that session. Also has session speicfic functions.
    # Inputs:
    #Subject is an instance of Subject class

    # constructor
    def __init__(self,subj,sess_idx=0,paramsDict=None, do_init=True):
        # paramsDict ... additional parameter definition as in Subject
        #sess_idx ... index of session in session list. Default, analyzes first session

        # invoke constructor of parent class
        Subject.__init__(self,subj,paramsDict)


        if sess_idx > len(self.sess_list)-1 | sess_idx < 0:
            error('Session not found')

        #log sess data
        self.sess_idx = sess_idx
        self.sess_label = self.sess_list[sess_idx]

        # filter events by trials in this session
        self.ev_df = self.ev_df.query('session==@self.sess_label')
        self.ev_sessFilt = self.sess_label

    # over-write revertEvents function to also include a session filter
    def revertEvents(self):

        self.ev_df = self.ev_df_master.copy()
        self.ev_evQuery = None
        # also filter by session
        self.ev_df = self.ev_df.query('session==@self.sess_label')
        self.ev_sessFilt = self.sess_label

    # Session version -plots trial by trial data within this session
    #If called from Subject class, it will loop through session list
    def plot_TrialByTrial(self,evQuery = None, ax=None, fig_params_dict = None):


        #default fig_params
        fig_params = {'figsize':(5,20),
                      'markersize':10,
                         'legend_fontsize':10}

        # update this dict if dictionary values are provided
        if fig_params_dict != None:
            fig_params.update(fig_params_dict)

        if ax == None:
            fig = figure(figsize=fig_params['figsize'])
            ax = subplot(111)

        # filter events
        choiceEv = self.ev_df.query('type=="RESPONSE"')

        # additional filter (e.g., error trials)
        if evQuery!=None:
            choiceEv = choiceEv.query(evQuery)


        # initialize line obj
        mark_error = None
        mark_cc = None
        mark_correct = None
        mark_fast = None

        #loop through trials and plot RT data
        for t in choiceEv['trial'].to_numpy():
            #filter data for this trial
            trEv = choiceEv.query('trial==@t')

            # plot delay time
            mark_cc, = plot(trEv['delay'].to_numpy(),t,'ob',markersize = fig_params['markersize'], fillstyle='none',alpha=0.5,label='color change')

            # plot RT (relative to delay)
            if trEv['error'].to_numpy()==True:
                mark_error, = plot(trEv['RT'].to_numpy()+trEv['delay'].to_numpy(),t,'xr',markersize=fig_params['markersize'],label='error')
            else:
                if trEv['fastResponse'].to_numpy()==False: # correct trial
                    mark_correct, = plot(trEv['RT'].to_numpy()+trEv['delay'].to_numpy(),t,'xg',markersize=fig_params['markersize'],label='correct')
                elif trEv['fastResponse'].to_numpy()==True: # fast response trial
                    mark_fast, = plot(trEv['RT'].to_numpy()+trEv['delay'].to_numpy(),t,'xk',markersize=fig_params['markersize'],label='fast')


        # set axes
        ax.set_xlim(0,3000)
        ax.set_ylabel('Trial')
        ax.set_xlabel('Time from Fixation On (ms)')
        ax.set_xticks((500,1500))

        # legend
        mark_error = None
        ax.legend((mark_cc,mark_correct,mark_fast,mark_error),('color change','correct','fast','error'),fontsize=fig_params['legend_fontsize'])


class GroupBehavior(Subject):
# Subclass. Will inherit attributes from Subject.
# aggegates behavioral analyses across subjects and does group level analyses


    # CONSTRUCTOR
    def __init__(self,subj_list = None):

        #basic inputs
        self.subj_list = subj_list

        # PARSE INPUTS
        # if a list of subjects is not provided, we construct it based on the label
        if np.all(subj_list == None):

            # load ccdt_info JSON file to get directories and subjInfor
            with open("ccdt_info.json", "r") as read_file:
                ccdt_info = json.load(read_file)

            # construct uElbl list by looping through subjects, and anatomical data
            self.subj_list = []


            for s in np.arange(0,len(ccdt_info['subjects'])):
  
                # collect subj list
                if ccdt_info["subjects"][s]['exclude']==False:
                    self.subj_list.extend([ccdt_info["subjects"][s]['subj']])

            # save master list (use this for collecting group dat, this will not get filtered)
            self.subj_list_master = self.subj_list


        # Initialize subj electrode with the first electrode in the list. so you have access to key methods
        Subject.__init__(self,subj=self.subj_list[0])
    # get num trials
    def getNumTrials(self):
        n_sess= []
        n_trials = []

        for s in np.arange(0,len(self.subj_list)):
            # subejct
            S = Subject(subj = self.subj_list[s])

            # num trials
            n_trials.append(len(np.unique(S.ev_df['trial'])))

            # num sess
            n_sess.append(len(np.unique(S.ev_df['session'])))

        return n_trials,n_sess



    # COLLECT_RT_DATA
    def getRTData(self,evQuery = 'error==0&fastResponse==0'):

        # create a list to hold each subject's behavioral dict dictionary
        beh_dict_list= []

        # rt_dist_types
        rt_dist_types = ['standard','reciprocal','zrrt','reciprobit','LATER']

        # loop through subjects
        for s in np.arange(0,len(self.subj_list)):

            # initialize Subject
            S = Subject(subj = self.subj_list[s]) 

            #print(S.subj)
            beh_dict = {}

  
            # loop through dist types
            for r in rt_dist_types:

                if r == 'reciprobit': #special case
                    slope, intercept,rt_median,std_err,r_sq = S.getReciprobitFits(evQuery=evQuery)
                    beh_dict['reciprobit_slope'] = slope
                    beh_dict['reciprobit_intercept'] = intercept
                    beh_dict['reciprobit_median'] = rt_median
                    beh_dict['reciprobit_std_err'] = std_err
                    beh_dict['reciprobit_rsq'] = r_sq
                elif r == 'LATER': # special case 2
                    rise_rate, rise_distance, lle = S.fitLATER(evQuery = evQuery)
                    beh_dict['LATER_rise_rate'] = rise_rate
                    beh_dict['LATER_rise_distance'] = rise_distance
                    beh_dict['LATER_lle'] = lle

                else:
                    # std
                    rts = S.getRTs(evQuery = evQuery, rt_dist_type = r)
                    beh_dict['rts_'+r] = [rts] 
                    beh_dict['rts_'+r+'_mean'] = np.mean(rts)
                    beh_dict['rts_'+r+'_std'] = np.std(rts)


            # update list of dictionaries
            beh_dict_list.append(beh_dict)

        # convert to a data frame
        beh_df = pd.DataFrame(beh_dict_list,index = self.subj_list)

        # update self
        self.beh_df = beh_df

    def getRTData_compareDelay(self,evQuery = 'error==0&fastResponse==0'):
        # collects a data frame of RT-related differences between delay conditions (cd = 'compare delay'). Computes differences as Long - Short 

        # create a list to hold each subject's behavioral dict dictionary
        beh_dict_list= []

        # rt_dist_types
        rt_dist_types = ['standard','reciprocal','zrrt','reciprobit','LATER']

        # loop through subjects
        for s in np.arange(0,len(self.subj_list)):

            # initialize Subject
            S = Subject(subj = self.subj_list[s]) 

            #print(S.subj)
            beh_dict = {}

            # loop through dist types
            for r in rt_dist_types:

                if r == 'reciprobit': #special case
                    # get short delay
                    slope_s, intercept_s,rt_median_s,std_err_s,r_sq_s = S.getReciprobitFits(evQuery=evQuery+'&delay==500')
                    # get long delay
                    slope_l, intercept_l,rt_median_l,std_err_l,r_sq_l = S.getReciprobitFits(evQuery=evQuery+'&delay==1500')


                    beh_dict['reciprobit_slope'] = slope_l - slope_s
                    beh_dict['reciprobit_intercept'] = intercept_l - intercept_s
                    beh_dict['reciprobit_median'] = rt_median_l - rt_median_s
                    beh_dict['reciprobit_std_err'] = std_err_l - std_err_s
                    beh_dict['reciprobit_rsq'] = r_sq_l - r_sq_s

                elif r == 'LATER': # special case 2
                    # fit LATER by delay conditions
                    S.fitLATER_byCondition(rts_A=[], rts_B=[],evQuery = evQuery, model_type = 'best')

                    beh_dict['LATER_rise_rate'] = S.laterByCond_dict['paramsDiff_rise_rate']
                    beh_dict['LATER_rise_bound'] = S.laterByCond_dict['paramsDiff_rise_bound']
                    beh_dict['LATER_rise_std'] = S.laterByCond_dict['paramsDiff_rise_std']
                    beh_dict['LATER_model_type'] = S.laterByCond_dict['model_type_B']

                else:
                    # rts short delay
                    rts_s = S.getRTs(evQuery = evQuery+'&delay==500', rt_dist_type = r)

                    # rts long delay 
                    rts_l = S.getRTs(evQuery = evQuery+'&delay==1500', rt_dist_type = r)

                    beh_dict['rts_'+r+'_mean'] = np.mean(rts_l)-np.mean(rts_s)
                    beh_dict['rts_'+r+'_std'] = np.std(rts_l)-np.std(rts_s)


            # update list of dictionaries
            beh_dict_list.append(beh_dict)

        # convert to a data frame
        beh_df_cd = pd.DataFrame(beh_dict_list,index = self.subj_list)

        # update self
        self.beh_df_cd = beh_df_cd

    # PLOT RTs for all subjects
    # SEE FUNCTION IN fig_grpBeh NOTEBOOK
    # def plotRTs(self,evQuery = 'error==0&fastResponse==0',plot_type='standard',plot_median=True,yL = (-2,2),model_type = 'mean_std',alpha = 0.5,bins = None, color = '0.5'):

    #     # loop through subjects
    #     for s in np.arange(0,len(self.subj_list)):

    #         # initialize Subject
    #         S = Subject(subj = self.subj_list[s]) 

    #         f = plt.figure()

    #         S.plotRT(evQuery=evQuery,plot_type=plot_type,plot_median=plot_median,yL=yL,model_type=model_type,alpha = alpha,bins = bins,color = color)
    #         S.set_axes_rt(ax=plt.gca(),plot_type = plot_type)

    # PLOT RTs for all subjects
    def plotRTs_byDelay_bySubj(self,evQuery = 'error==0&fastResponse==0',plot_type='standard',plot_median=True,yL = (-2,2),model_type = 'best',model_type_list = []):

        # loop through subjects
        for s in np.arange(0,len(self.subj_list)):

            # initialize Subject
            S = Subject(subj = self.subj_list[s]) 

            S.plot_RT_by_delay(evQuery=evQuery,plot_type=plot_type,plot_median=plot_median,yL=yL,model_type=model_type,model_type_list = model_type_list)

    # perform RT 
    def fitMemoryRegression(self,evQuery = 'error==0&fastResponse==0',decay_model='best'):
        # GROUPBEHAVIOR CLASS. overwrites Subject.fitMemoryRegression()

        # initialize 
        mem_dict_list= []


        # loop through subjects
        for s in np.arange(0,len(self.subj_list)):

            # initialize Subject
            S = Subject(subj = self.subj_list[s]) 

            # append mem regression dict
            mem_dict_list.append(S.fitMemoryRegression(evQuery = evQuery,decay_model = decay_model))

    
        # convert to dataframe
        self.memReg_df = pd.DataFrame(mem_dict_list,index = self.subj_list)


class Electrode(Session):
# Subclass. Will inherit attributes from Session.
# Will process ephys data for a single electrode from a single session
# Will need separate functions under Sessions to aggregate data across electrodes
# Will need separate functions under Subject to aggregate data across sessions

    # Inputs:
    #Initializers for subj, sess,  is an instance of Subject class

    # constructor
    def __init__(self,subj,sess_idx=0,elec1_lbl=None,elec2_lbl=None,paramsDict=None,do_init=True):
        # paramsDict ... additional parameter definition as in Subject
        #sess_idx ... index of session in session list. Default, analyzes first session

        # invoke constructor of parent class
        Session.__init__(self,subj,sess_idx,paramsDict)

        # initialize badElectrode as false (once this is set to true, should skip when performing analyses)
        self.isBadElectrode = False


        # if bipolar montage
        if self.params['montage'] == 'bipolar':
            # error check: if bipolar montage, we are expecting two electrode labels
            if elec2_lbl == None:
                raise NameError('Please provide second electrode label for bipolar montage')
            else:
                # store electrode related data
                self.eLbl = elec1_lbl+'-'+elec2_lbl
                self.eNum1 = self.eNum_js[self.eLbl_js.index(elec1_lbl)]
                self.eNum2 = self.eNum_js[self.eLbl_js.index(elec2_lbl)]
                self.eegfname_mother = np.unique(self.ev_df['eegfile'])[0]
                self.eegfname1 = self.eegDir+self.eegfname_mother+'.'+format(self.eNum1 , '03')
                self.eegfname2 = self.eegDir+self.eegfname_mother+'.'+format(self.eNum2 , '03')


                if len(np.unique(self.ev_df['eegfile']))>1:
                    raise NameError('multiple eegfilenames found')

                # anat info
                uElbl = subj+'-'+elec1_lbl+'-'+elec2_lbl
                self.anat_dict=self.tal_df.query('uElbl==@uElbl').iloc[0]  
                self.anat = self.anat_dict['anat_native']

        if self.params['montage'] == 'monopolar':
            self.eLbl = elec1_lbl
            self.eNum1 = self.eNum_js[self.eLbl_js.index(elec1_lbl)]
            self.eNum2 = None
            # [ ] need to come up with a method to do a local re-reference
            # add anat info here

        # make filename
        self.fname = self.subj+'-'+self.sess_label+'-'+self.eLbl
        self.do_init = do_init

        #functions to run at startup
        if do_init==True:
            self.run_init_functions()

    # maintains list of init_funcitons
    def run_init_functions(self):
        self.loadRawEEG()
        self.classifyBadElectrode()
        self.toEpochsArray()
        self.getRandomEpochsArray()
        self.findBadTrials()
        self.do_init = True


    # ms_to_samples
    def ms_to_samples(self,ms):
        # Input
        #ms ... ms value to convert to samples (int or array)
        samples = (ms/1000)*self.samplerate
        return samples


    # samples_to_ms
    def samples_to_ms(self,samples):
        # Input
        #samples ... sample value to convert to samples (int or array)
        ms = (samples/self.samplerate)*1000
        return ms

    # getWaveFreqs
    def getWaveFreqs(self,wave_frange,wave_number_of_freq):
        myfreqs = np.logspace(np.log10(wave_frange[0]), np.log10(wave_frange[1]), num=wave_number_of_freq)
        return myfreqs

    # loadRawEEG
    def loadRawEEG(self):
        #this function loads raw data from this electrode for the session
        if self.params['montage'] == 'bipolar':
            # read electrode 1
            f = open(self.eegfname1, "r")
            if 'int16' in self.dataformat:
                eeg1 = np.fromfile(f, dtype=np.int16)
            f.close()

            # read electrode 2
            f = open(self.eegfname2, "r")
            if 'int16' in self.dataformat:
                eeg2 = np.fromfile(f, dtype=np.int16)
            f.close()

            # bipolar signal
            self.eeg = eeg1-eeg2

            # make a MNE RAW array (will be used in multiple functions below to create epochs arrays)
            # create metadata and create info
            n_channels = 1
            info = mne.create_info(ch_names=[self.eLbl], sfreq=self.samplerate,ch_types='eeg')

            # get the data and reshape it in n_chan x time
            data = np.reshape(self.eeg,[n_channels,np.shape(self.eeg)[0]])

            # Create MNE RAW object
            self.Raw = mne.io.RawArray(data, info,verbose =False)
    def classifyBadElectrode(self):
        # This function uses self.Raw to identify whether an electrode is noise 'self.isBadElectrode = True'. 
        # get eeg 
        eeg = self.Raw.get_data().squeeze()

        # check STD of signal
        eeg_std = np.nanstd(eeg)

        # calc PSD (20 Hz and 60 Hz
        psd_beta,freqs_beta = mne.time_frequency.psd_array_welch(x=eeg,sfreq=self.samplerate,fmin = 18, fmax = 22,verbose=False)
        psd_noise,freq_noise = mne.time_frequency.psd_array_welch(x=eeg,sfreq=self.samplerate,fmin = 58, fmax = 62,verbose=False)

        # if STD is 0, the channel is broken or disconnected
        if eeg_std == 0:
            self.isBadElectrode = True

        # if 60 Hz power is > 20 Hz power, the channel is noisy
        if np.nanmean(psd_noise) > np.nanmean(psd_beta):
            self.isBadElectrode = True

    def toEpochsArray(self,incl_buffer = True):
        # This function creates an MNE Epochs array for eeg data. Must already have loaded data (loadData
        # Inputs:
            # incl_buffer = True. By default will include the buffer when constructing EpochsArray.
            # Must be removed later after performing the analysis function (e.g., hilbert, wavelet, phase etc)


            # get a list of event types
            self.evType_list =  np.unique(self.ev_df['type'].to_numpy())

            for evType in self.evType_list:

                # filt events
                filtEv = self.ev_df.query('type==@evType')

                # create mne events array
                # column 1 - eegoffset
                # column 2 - 0s
                # column 3 - ev ID (ones for now)
                mne_ev = np.concatenate((filtEv.eegoffset.to_numpy().astype('int').reshape(len(filtEv),1),
                                         np.zeros((len(filtEv),1)).astype('int'),
                                         np.ones((len(filtEv),1)).astype('int')),
                                         axis=1)

                # create Epochs Aray and save
                if incl_buffer==True:
                    start_s = (self.params['tmin_ms']-self.params['buffer_ms'])/1000
                    stop_s = (self.params['tmax_ms']+self.params['buffer_ms'])/1000
                else:
                    start_s = self.params['tmin_ms']/1000
                    stop_s = self.params['tmax_ms']/1000
                setattr(self,'Epochs_'+evType,
                        mne.Epochs(self.Raw,events=mne_ev,
                                   tmin=start_s,
                                   tmax=stop_s,preload=False,verbose =False))

    def findBadTrials(self):
        # This function identifies all bad trials as those with noise in the time series data and updates the events dataframe with a column marking those trials
        # It also labels the electrode as isBad if > 50 % of trials are bad


        # create mne events array with all trials in session
        # column 1 - eegoffset
        # column 2 - 0s
        # column 3 - ev ID (ones for now)
        mne_ev = np.concatenate((self.ev_df['eegoffset'].to_numpy().astype('int').reshape(len(self.ev_df),1),
                                         np.zeros((len(self.ev_df),1)).astype('int'),
                                         np.ones((len(self.ev_df),1)).astype('int')),
                                         axis=1)


        # deal with repeating events with the same eegoffset
        #(this can happen when button is pressed a the same time as color change e.g., (HUP133, session 3, set 3, trial 1)
        # we subtract 1 from the first event sample time
        repeat_idx = np.nonzero(np.diff(mne_ev[:,0])==0)[0]
        mne_ev[repeat_idx,0] = mne_ev[repeat_idx,0]-1

        # find start and stop times (including buffer)
        start_s = (self.params['tmin_ms']-self.params['buffer_ms'])/1000
        stop_s = (self.params['tmax_ms']+self.params['buffer_ms'])/1000

        # make epochs array of all events
        mne_all = mne.Epochs(self.Raw,events=mne_ev,tmin = start_s,tmax=stop_s,preload=False,verbose=False)

        # get eeg as trials x time
        eeg_all = mne_all.get_data().squeeze()

        # get overlal mean and std
        eeg_all_mean = eeg_all.mean()
        eeg_all_std = eeg_all.std()

        # get std for each trial
        eeg_all_sd_z = stats.zscore(np.nanstd(eeg_all,axis=1))

        # get absolue and mean (NOT nanmean) for each trial (this will identify trials with infs and nans)
        eeg_all_mean_z = stats.zscore((np.mean(np.absolute(eeg_all),axis=1)))

        # identify bad trials as those that either have nans or inf
        # or are above 20 std above the average std of a trial
        bad_trials_idx = ((np.isinf(eeg_all_mean_z)) |
                          (np.isnan(eeg_all_mean_z)) |
                          (np.absolute(eeg_all_mean_z)>=self.params['reject_z_thresh'])|
                          (np.absolute(eeg_all_sd_z)>=self.params['reject_z_thresh'])
                         )

        # make sure that every trial has a homogenous designation
        trial_nums = self.ev_df['trial'].to_numpy()
        addl_bad_trials = []
        for i in np.nonzero(bad_trials_idx)[0]:
            # find all events associated with this trial number
            addl_bad_trials.extend(np.nonzero(trial_nums==trial_nums[i])[0])
        #update bad trials with additional idx so that all events in a trial are excluded together
        bad_trials_idx[addl_bad_trials] = True

        # update ev_df with bad_trials_idx
        self.ev_df.insert(loc = len(self.ev_df.columns),column='badTrial',value=bad_trials_idx)

        # classify as bad electrode if > 50% of trials are bad
        if (len(bad_trials_idx.nonzero()[0])>np.round(0.5*len(bad_trials_idx))):
            self.isBadElectrode = True

    def getRandomEpochsArray(self, incl_buffer=True):
        #This function generates a null EpochsArray using list of random timestamps ranging from the
        # beginning to the end of the session

        # Inputs:
        # incl_buffer = True. By default will include the buffer when constructing EpochsArray.
            # Must be removed later after performing the analysis function (e.g., hilbert, wavelet, phase etc)

        # raise error if events have been filtered
        if self.ev_evQuery != None:
            raise NameError('WARNING::: session events have been filtered. Please revertEvents() before getting random events')

        #generate random timestamps (using unfiltered events)
        start_offset = self.ev_df_master['eegoffset'].to_numpy()[0]
        end_offset = self.ev_df_master['eegoffset'].to_numpy()[-1]


        # generate array of all timestamps during the session
        # (need a list for random.sample())
        offsets = np.arange(start_offset,end_offset+1)

        # run random.sample to sample unique random timestamps
        # returns a list, then convert to an array and reshape
        rd.seed(a=0) # initialize random seed so we can reproduce results
        rand_offsets = np.array(rd.sample(offsets.tolist(),len(self.ev_df))).reshape(len(self.ev_df),1)

        #make mne random events
                    # column 1 - eegoffset
                    # column 2 - 0s
                    # column 3 - ev ID (delay for now)
        randEv_mne = np.concatenate((rand_offsets,
                                 np.zeros((len(self.ev_df),1)).astype('int'),
                                 np.ones((len(self.ev_df),1)).astype('int')),
                                 axis=1)

        # random epochs
        # create Epochs Aray and save
        if incl_buffer==True:
            start_s = (self.params['tmin_ms']-self.params['buffer_ms'])/1000
            stop_s = (self.params['tmax_ms']+self.params['buffer_ms'])/1000
        else:
            start_s = self.params['tmin_ms']/1000
            stop_s = self.params['tmax_ms']/1000
        setattr(self,'Epochs_RANDOM',
                mne.Epochs(self.Raw,events=randEv_mne,
                           tmin=start_s,
                           tmax=stop_s,
                           preload=False,verbose=False))


    def calcHilb(self,evType='CC',hilb_frange_lbl = 'HFA'):
        # This function calculates hilbert transform for a particular freq range and event type
        # get Epochs
        Epochs = getattr(self,'Epochs_'+evType)

        # get EEG (2D, n trials x ntime)
        eeg = np.copy(Epochs.get_data().squeeze())

        # filter data
        eeg_filt = mne.filter.filter_data(data = eeg,
                                          sfreq = self.samplerate,
                                          l_freq = self.params['frange_'+hilb_frange_lbl][0],
                                          h_freq = self.params['frange_'+hilb_frange_lbl][1],
                                          method='fir')

        # apply hilbert, returns a complex valued vector
        eeg_hilbert = signal.hilbert(eeg_filt)

        # get power:  absolute(hilbert transform)^2
        #eeg_hilbert_power = np.abs(eeg_hilbert)**2

        # get power:  absolute(hilbert transform)^2
        #eeg_hilbert_phase = np.angle(eeg_hilbert,deg=False)

        # remove buffer
        buff_idx = int((self.params['buffer_ms']/1000)*self.samplerate)
        eeg_hilbert = eeg_hilbert[:,buff_idx:-buff_idx]
        #eeg_hilbert_power = eeg_hilbert_power[:,buff_idx:-buff_idx]
        #eeg_hilbert_phase = eeg_hilbert_phase[:,buff_idx:-buff_idx]

        # store data
        setattr(self,'hilbComplex_'+hilb_frange_lbl+'_'+evType, eeg_hilbert)
        #setattr(self,'hilbPower_'+hilb_frange_lbl+'_'+evType, eeg_hilbert_power)
        #setattr(self,'hilbPhase_'+hilb_frange_lbl+'_'+evType,eeg_hilbert_phase)

    def hilb2pow(self,pow_frange_lbl,pow_evType):
        # returns power from complex hilbert analytic signal. Self must already contain calculated hilbert analyitic signal

        # get hilbert
        hilbComplex = getattr(self,'hilbComplex_'+pow_frange_lbl+'_'+pow_evType)

        # compute power
        hilb_power = np.abs(hilbComplex)**2

        # log transform power
        hilb_power = log10(hilb_power)

        # return power
        return hilb_power

    def hilb2amp(self,amp_frange_lbl,amp_evType):
        # returns amplitude from complex hilbert analytic signal. Self must already contain calculated hilbert analyitic signal

        # get hilbert
        hilbComplex = getattr(self,'hilbComplex_'+amp_frange_lbl+'_'+amp_evType)

        # compute power
        hilb_amp = np.abs(hilbComplex)

        # return hilb amplitude
        return hilb_amp

    def hilb2phase(self,phase_frange_lbl,phase_evType):
        # returns power from complex hilbert analytic signal. Self must already contain calculated hilbert analyitic signal

        # get hilbert
        hilbComplex = getattr(self,'hilbComplex_'+phase_frange_lbl+'_'+phase_evType)

        # compute power
        hilb_phase = np.angle(hilbComplex,deg=False)

        # return power
        return hilb_phase


    def calcWavelet(self,evType='CC',wave_frange=(2,200),wave_number_of_freq = 15, wave_number_of_cycles=5):
        # This function performs wavelet analysis for all event types.
        # Wavelet parameters are set in params dictionary
        # Inputs
        #wave_frange=(2,200), ... freq range for calculating wavelets
        #wave_number_of_freq=15 ... number of frequencies
        #wave_number_of_cycles=5 .... higher wave number results in greater
                            #freq resolution and reduced temporal resolution
                            #lower wave number results in greater temporal resolution
                            #but reduced frequency resolution

        # get Epochs
        Epochs = getattr(self,'Epochs_'+evType)

        # get EEG (2D, n trials x nchan x time)
        eeg = np.copy(Epochs.get_data())

        #calculate log-spaced frequencies
        myfreqs = self.getWaveFreqs(wave_frange=wave_frange,wave_number_of_freq=wave_number_of_freq)

        # get complex value morlet wavelets
        tfr = mne.time_frequency.tfr_array_morlet(eeg,sfreq=self.samplerate,
                                                  freqs=myfreqs,
                                                  n_cycles=wave_number_of_cycles,
                                                  output='complex')

        # Transform complex values to get power and phase (taken from tfr_array_morlet code)
        wave_complex = tfr
        #power = (tfr * tfr.conj()).real  # power
        #phase = np.angle(tfr) # phase

        # reshape to freq x time x trials
        wave_complex = np.moveaxis(wave_complex,(2,3,0),(0,1,2)).squeeze()
        #power = np.moveaxis(power,(2,3,0),(0,1,2)).squeeze()
        #phase = np.moveaxis(phase,(2,3,0),(0,1,2)).squeeze()

        # drop the buffer
        wave_complex = wave_complex[:,int(self.ms_to_samples(self.params['buffer_ms'])):-1*int(self.ms_to_samples(self.params['buffer_ms'])),:]
        #power = power[:,int(self.ms_to_samples(self.params['buffer_ms'])):-1*int(self.ms_to_samples(self.params['buffer_ms'])),:]
        #phase = phase[:,int(self.ms_to_samples(self.params['buffer_ms'])):-1*int(self.ms_to_samples(self.params['buffer_ms'])),:]

        # log transform the power
        #power = log10(power)

        # store vars
        setattr(self,'waveComplex_'+evType, wave_complex)
        #setattr(self,'wavePower_'+evType, power)
        #setattr(self,'wavePhase_'+evType, phase)
        setattr(self,'wave_freqs', myfreqs)
        setattr(self,'wave_number_of_cycles', wave_number_of_cycles)
    def wave2pow(self,pow_evType):
        # This function returns power from the complex wavelet signal (must calc power first)

        # get cache'd complex signal
        waveComplex = getattr(self,'waveComplex_'+pow_evType)

        # calculate power
        wavePower = (waveComplex * waveComplex.conj()).real

        # log transform power
        wavePower = log10(wavePower)

        # return power
        return wavePower

    def wave2amp(self,pow_evType):
        # This function returns amplitude from the complex wavelet signal

        # get cache'd complex signal
        waveComplex = getattr(self,'waveComplex_'+pow_evType)

        # calculate amplitude
        waveAmp = sqrt((waveComplex * waveComplex.conj()).real)

        # return power
        return waveAmp
    def wave2phase(self,phase_evType):
        # This function returns power from the complex wavelet signal (must calc power first)

        # get cache'd complex signal
        waveComplex = getattr(self,'waveComplex_'+phase_evType)

        # calculate phase
        wavePhase = np.angle(waveComplex) # phase

        # return phase
        return wavePhase


    def calcTFWrapper(self,do_hilb = True,hilb_frange_lbl = 'HFA',do_wave=True):
        # This function wraps around all event types and runs calcHilb for all event types including RANDOM
        # Run this function before running getHilb() which will load these data
        # Inputs
        #do_hilb == True ... will calculate hilbert transfrom for specified freq range

        #
        # create list of event types (including RANDOM)
        evType_list = ['RANDOM']+list(np.unique(self.ev_df['type'].to_numpy()))

        # Hilbert
        if do_hilb == True:

            # look for saved file (for hilbert, need to include frange in label)
            fname_hilb =  self.params_dir+'HILBERT_'+hilb_frange_lbl+'_'+self.fname

            # if exists and flags are on, load it
            if (os.path.exists(fname_hilb)==True) & (self.params['saveTF']==False)  & (self.params['overwriteFlag']==False):

                tfDict = self.load_pickle(fname_hilb)

                #unpack data from waveDict and store in self
                self.unpackTFDict(tfDict)
            else:
                # compute data
                for evType in evType_list:
                    self.calcHilb(evType=evType,hilb_frange_lbl=hilb_frange_lbl)

                # create TF Dictionary
                tfDict = self.mkTFDict(pow_method='hilb',
                                       evType_list=evType_list,
                                       hilb_frange_lbl=hilb_frange_lbl)

                # save TFDict
                if self.params['saveTF']==True:
                    self.save_pickle(obj = tfDict, fpath = fname_hilb)

        # Wavelet
        if do_wave == True:

            # look for saved file,
            fname_wave =  self.params_dir+'WAVELET_'+self.fname

            # if exists and if overwrite flag is off, load it
            if (os.path.exists(fname_wave)==True) & (self.params['saveTF']==False)  & (self.params['overwriteFlag']==False):
                tfDict = self.load_pickle(fname_wave)

                #unpack data from tfDict and store in self
                self.unpackTFDict(tfDict)

            else:
                # compute data
                for evType in evType_list:
                    self.calcWavelet(evType=evType,
                                     wave_frange = self.params['wave_frange'],
                                     wave_number_of_freq = self.params['wave_number_of_freq'],
                                     wave_number_of_cycles = self.params['wave_number_of_cycles'])

                # create TF Dictionary
                tfDict = self.mkTFDict(pow_method='wave',evType_list=evType_list)

                # save waveDict
                if self.params['saveTF']==True:
                    self.save_pickle(obj = tfDict, fpath = fname_wave)


    def mkTFDict(self,pow_method = 'wave',evType_list=None,hilb_frange_lbl='HFA'):
        # This function constructs a dictionary to store TF data.
        # Assumes that self is already poulated with TF data
        # Inputs
        # pow_method = 'wave' (or 'hilb')... flag for whether we are working with wavelet or hilbert data
        # evType_list.... list of event types

        # create empty Dict to store wavelet data
        tfDict = {}

        # create list of attributes to populate (for each event type)
        if pow_method == 'hilb':
            attr_list = ['hilbComplex_'+hilb_frange_lbl+'_']

        elif pow_method == 'wave':
            attr_list = ['waveComplex_']

        # populate TF Dict with ev type attributes
        for attr in attr_list:
            for ev in evType_list:
                tfDict[attr+ev] = getattr(self,attr+ev)

        # add additional attributes (not related to event type)
        if pow_method == 'wave':
            tfDict['wave_freqs'] = getattr(self,'wave_freqs')
            tfDict['wave_number_of_cycles'] = getattr(self,'wave_number_of_cycles')

        # return tfDict
        return tfDict

    def unpackTFDict(self,tfDict):
        # This function unpacks TF dictionary and loads data into self.
        # It loops through each key in the dictionary, so no need to specify wave vs hilb

        # loop through keys in dictionary
        for i in tfDict.keys():
            # store in self
            setattr(self,i,tfDict[i])


    def makeTimeBins(self,time_bin_size_ms = 100):
        # This function generates a time window dictionary and updates self object. Can be used for other
        # functions. Can then make copies of this dictionary as it was used for a particular function and
        # then recompute them with a different bin size

        #
        time_bin_starts = np.arange(self.params['tmin_ms'],self.params['tmax_ms'],time_bin_size_ms)

        #create time bin dictionary
        self.time_bin_dict = {'size_ms':time_bin_size_ms,
                              'starts_ms':time_bin_starts,
                              'lbl':(time_bin_starts+time_bin_size_ms/2).astype('int')}

        return self.time_bin_dict


    def powMat2FreqBin_3d(self,powMat,freqs,foi):
        # This function takes a 3D power matrix (freq x samples x trials) and
        # collapses within a frequency range and returns a 2D power matrix (trials x samples)
        # Inputs
        # powMat.... 3D power matrix (freq x samples x trials)
        # freqs ... vector of frequencies used to construct the power matrix
                    #freqs = self.wave_freqs

        # foi .... tuple frequency range of interest (e.g., (70,200))

        # freq index
        f_start = np.nonzero(freqs.astype('int')>=foi[0])[0][0]
        f_end = np.nonzero(freqs.astype('int')<=foi[1])[0][-1]

        # collapse within freq
        #(Transpose at the end is to return the correct shape)
        powMat_frange = np.nanmean(powMat[f_start:f_end,:,:],axis=0).T

        # return
        return powMat_frange

    def phsMat2phsVec(self,phsMat,freqs,foi):
        # This function takes a 3D phase matrix (freq x samples x trials) and
        # collapses within a frequency range by finding the wavelet that most closely
        # matches the frequency range of interest (foi)
        # returns a 2D phase matrix (trials x samples)
        # Inputs
        # phsMat.... 3D phase matrix (freq x samples x trials)
        # freqs ... vector of frequencies used to construct the phase matrix
                    #freqs = self.wave_freqs

        # foi .... tuple frequency range of interest (e.g., (70,200))

        # freq index
        f_idx = np.argmin(np.absolute(freqs.astype('int')-np.mean(foi)))[0]


        # collapse within freq
        #(Transpose at the end is to return the correct shape)
        phsMat_frange = phsMat[filt_idx,:,:].T

        # return
        return phsMat_frange

    def powMat2timeBins_2d(self,powMat, time_bin_size_ms = 100):
        # This function takes a 2D power matrix (trials x samples) and bins it into time windows
        # returns the binned matrix as a variable (does not store in self)
        # Inputs
        # powMat..2D matrix (trials x samples). Indented for power
        # but can be any variable that can be averaged across rows
        # time_bin_size_ms = 100....length of time window to bin

        # make time bins
        self.makeTimeBins(time_bin_size_ms=time_bin_size_ms)

        # get sample vector
        samps_vec = np.arange(int(self.ms_to_samples(self.params['tmin_ms'])),
                              (self.ms_to_samples(self.params['tmax_ms'])))

        # initialize bin vector
        powMat_bin = np.empty((powMat.shape[0],self.time_bin_dict['starts_ms'].shape[0]))

        # loop through time bins
        for t in arange(0,len(self.time_bin_dict['starts_ms'])):
            samp_start = int((self.time_bin_dict['starts_ms'][t]/1000)*self.samplerate)
            samp_start_idx = np.argmin(np.abs(samps_vec-samp_start))

            samp_end = int(samp_start+(self.time_bin_dict['size_ms']/1000)*self.samplerate)
            samp_end_idx = np.argmin(np.abs(samps_vec-samp_end))

            powMat_bin[:,t] = np.nanmean(powMat[:,samp_start_idx:samp_end_idx],axis=1)

        return powMat_bin
    def phsMat2timeBins_2d(self,phsMat, time_bin_size_ms = 100):
        # This function takes a 2D phase matrix (trials x samples) and bins it into time windows
        # returns the binned matrix as a variable (does not store in self)
        # Inputs
        # phsMat..2D matrix (trials x samples).
        # time_bin_size_ms = 100....length of time window to bin

        # make time bins
        self.makeTimeBins(time_bin_size_ms=time_bin_size_ms)

        # get sample vector
        samps_vec = np.arange(int(self.ms_to_samples(self.params['tmin_ms'])),
                              (self.ms_to_samples(self.params['tmax_ms'])))

        # initialize bin vector
        phsMat_bin = np.empty((phsMat.shape[0],self.time_bin_dict['starts_ms'].shape[0]))

        # loop through time bins
        for t in arange(0,len(self.time_bin_dict['starts_ms'])):
            samp_start = int((self.time_bin_dict['starts_ms'][t]/1000)*self.samplerate)
            samp_start_idx = np.argmin(np.abs(samps_vec-samp_start))

            samp_end = int(samp_start+(self.time_bin_dict['size_ms']/1000)*self.samplerate)
            samp_end_idx = np.argmin(np.abs(samps_vec-samp_end))

            phsMat_bin[:,t] = circ.mean(phsMat[:,samp_start_idx:samp_end_idx],axis=1)

        return phsMat_bin


    def powMat2timeBins_3d(self,powMat, time_bin_size_ms = 100):
        # This function takes a 3D power matrix (freq x samples x trials) and bins it into time windows
        # returns the binned matrix as a variable (does not store in self)
        # Inputs
        # powMat..3D power matrix (freq x samples x trials). Intended for power
        # but can be any variable that can be averaged across rows
        # time_bin_size_ms = 100....length of time window to bin

        # make time bins
        self.makeTimeBins(time_bin_size_ms=time_bin_size_ms)

        # get sample vector
        samps_vec = np.arange(int(self.ms_to_samples(self.params['tmin_ms'])),
                              (self.ms_to_samples(self.params['tmax_ms'])))

        # initialize bin vector
        powMat_bin = np.empty((powMat.shape[0],self.time_bin_dict['starts_ms'].shape[0],powMat.shape[2]))

        # loop through time bins
        for t in arange(0,len(self.time_bin_dict['starts_ms'])):
            samp_start = int((self.time_bin_dict['starts_ms'][t]/1000)*self.samplerate)
            samp_start_idx = np.argmin(np.abs(samps_vec-samp_start))

            samp_end = int(samp_start+(self.time_bin_dict['size_ms']/1000)*self.samplerate)
            samp_end_idx = np.argmin(np.abs(samps_vec-samp_end))

            powMat_bin[:,t,:] = np.nanmean(powMat[:,samp_start_idx:samp_end_idx,:],axis=1)

        return powMat_bin

    # function to concatonate 2d matrix to vectors
    def mat2vec(self,mat, xval_ms, pre_lims = -1000, post_lims = 1000):
        """ This function takes a 2 dim matrix (e.g, eeg, phase, pow) with
        the associated x values (e.g, -1000 to 1000 ms) and concatenates it
        to a long array of continuous data

        Inputs:
        mat ... 2dim matrix of event-related values (can be eeg, power, phase, amp etc)
        xval_ms... corresponding xvalues relative to 0 ms (labels for the columns).
                   Must be in ms. Use self.samples_to_ms to convert if needed
        pre_lims ...int or array. Marks the pre-zero interval in ms. Finds xval that is
                    closest match If array, it will loop through the matrix and
                    clip data trial by trial
        post_lims ...int or array. Marks the pre-zero interval in ms. Finds xval that is
                    closest match If array, it will loop through the matrix and
                    clip data trial by trial


         Returns:
         vec ... 1 dim vector, concatenated

         """
        # parse pre and post lims
        if (type(pre_lims)==int) & (type(post_lims)==int):
            pre_idx = np.argmin(np.absolute(xval_ms - pre_lims))
            post_idx = np.argmin(np.absolute(xval_ms - post_lims))

            # clip data pre-lim
            mat = mat[:,pre_idx:post_idx]

            # reshape to a 1 d array by inferring length from the matrix
            vec = np.reshape(mat, newshape = -1)

        else:
            # we need a loop to clip data trial by trial

            # make sure both pre and post lims are arrays (repeat values if needed)
            if (type(pre_lims)==int):
                pre_lims = np.repeat(pre_lims,np.shape(mat)[0])
            if (type(post_lims)==int):
                post_lims = np.repeat(post_lims,np.shape(mat)[0])

            # loop through trials
            for t in arange(0,np.shape(mat)[0]):

                # find idx
                pre_idx = np.argmin(np.absolute(xval_ms - pre_lims[t]))
                post_idx = np.argmin(np.absolute(xval_ms - post_lims[t]))

                # mark nans
                mat[t,:pre_idx] = nan
                mat[t,post_idx:] = nan

            # reshape to 1d array
            vec = np.reshape(mat, newshape = -1)

            # remove nans
            vec = vec[isnan(vec)==False]
        #
        return vec
    def getPow_2d(self,pow_evType='CC',pow_frange_lbl = 'HFA',pow_method = 'wave',pow_evQuery = None, do_zscore = True,apply_gauss_smoothing = True,gauss_sd_scaling = 0.075,apply_time_bins=False,time_bin_size_ms=100):
        # loads wavelet data and
        # gets the mean and sem of wavelet power for the event type, freq range, ev_query
        # it also has options to zscores to apply gaussian smoothing of raw power data and bin within non-overlapping time windows

        # First,  it tries to load saved pickle file of downsampled power, if not available it looks to looks to see if calcTFWrapper has been run, if not it runs calcTFWrapper. Then, it performs its function and saves downsampled power pickle file per params
        #

        # !!!! NOTE:: Buffer has already been removed by calcTFWrapper() (calcWavelet and calcHilbert, respectively)

        # Inputs
        # pow_evType='CC' .....  event type to load
        # pow_frange_lbl = 'HFA'.....  fRange to load
        # pow_method = 'hilb'(or 'wave')
        # pow_evQuery = None .....  event query to apply to filter trials
        # z_score = True .....  z-score by mean and std power for this freq range obtained from random timestamps
        # apply_gauss_smoothing = False .....  apply gaussian smoothing (intended for HFA)
        # gauss_sd_scaling = 0.1.. (samplerate * 0.1) sets the width of gaussian to smooth with.
        # apply_time_bins=False ... whether or not to averaage power data within time bins
        # time_bin_size_ms=100  .... length of the time window to use


        # generate filename
        self.fname_pow2d = self.fname+'-POWER-'+pow_evType+pow_frange_lbl+pow_method+str(do_zscore)+str(apply_gauss_smoothing)+str(gauss_sd_scaling)+str(apply_time_bins)+str(time_bin_size_ms)

        # look for saved pickle file of z-scored and downsampled power
        if (os.path.exists(self.params_dir+self.fname_pow2d)==True) & (self.params['overwriteFlag']==False):

            # load downsampled power mat
            powMat = self.load_pickle(self.params_dir+self.fname_pow2d)

            # create time bins dict (for saving purposes)
            self.makeTimeBins(time_bin_size_ms = time_bin_size_ms)

        #else, calculate zscored and downsampled power
        else:

            # run init functions if we did not at startup
            if self.do_init==False:
                self.run_init_functions()


            # load TF data
            if pow_method == 'hilb':

                # if TF data has not been calculated
                if hasattr(self,pow_method+'Complex_'+pow_frange_lbl+pow_evType)==False:


                    # run calc TF wrapper (wavelet only)
                    self.calcTFWrapper(do_hilb=True,hilb_frange_lbl=pow_frange_lbl)

                powMat = self.hilb2pow(pow_frange_lbl,pow_evType)

            elif pow_method == 'wave':
                 # if TF data has not been calculated
                if hasattr(self,pow_method+'Complex_'+pow_evType)==False:
                    # run calc TF wrapper (wavelet only)
                    self.calcTFWrapper(do_hilb=False)

                powMat = self.wave2pow(pow_evType)

                # collapse into frequency range (only for wavelet)
                powMat = self.powMat2FreqBin_3d(powMat,freqs=self.wave_freqs,foi = self.params['frange_'+pow_frange_lbl])

            # Optional: z-score by mean,std power from random events
            if do_zscore==True:

                # get random power
                if pow_method == 'hilb':
                    randPowMat = self.hilb2pow(pow_frange_lbl=pow_frange_lbl,pow_evType='RANDOM')

                elif pow_method == 'wave':
                    randPowMat = self.wave2pow('RANDOM')

                    # collapse into frequency range (for wavelet only)
                    randPowMat = self.powMat2FreqBin_3d(randPowMat,freqs=self.wave_freqs,foi = self.params['frange_'+pow_frange_lbl])


                # remove trials in rand pow mat with with "inf"
                inf_idx = np.isinf(np.absolute(np.mean(randPowMat,axis=1)))
                randPowMat = randPowMat[inf_idx==False,:]

                #mean
                randMeanPow  = np.nanmean(np.nanmean(randPowMat,axis=1),axis=0)
                randMeanPow = np.reshape(randMeanPow,(1,1))
                randMeanPow = np.repeat(np.repeat(randMeanPow,powMat.shape[0],axis=0),
                                            powMat.shape[1],axis=1)

                #std
                randStdPow  = np.nanstd(np.nanmean(randPowMat,axis=1),axis=0)
                randStdPow = np.reshape(randStdPow,(1,1))
                randStdPow = np.repeat(np.repeat(randStdPow,powMat.shape[0],axis=0),
                                            powMat.shape[1],axis=1)
                # z-score
                powMat = (powMat-randMeanPow)/randStdPow


            # Optional: apply Gaussian smoothing
            if apply_gauss_smoothing==True:
                powMat = ndimage.filters.gaussian_filter1d(powMat,sigma =(gauss_sd_scaling*self.samplerate))

            # Optional: average within time bins
            if apply_time_bins==True:
                # bin pow mat into time bins
                powMat=self.powMat2timeBins_2d(powMat, time_bin_size_ms = time_bin_size_ms)

            # save z-scored and downsampled 2d power
            if self.params['savePow'] == True:
                self.save_pickle(obj=powMat,
                                 fpath=self.params_dir+self.fname_pow2d)


        # outside the If statement
        # filter events if this is a real event type
        if pow_evType != 'RANDOM':

            # run bad trials functions if we did not at startup
            if (pow_evQuery!=None):
                if ('badTrial' in pow_evQuery)&(self.do_init==False):
                    self.run_init_functions()

            # load matching events
            # .resetindex() is to get matching indices with erp matrix
            pow_ev_filt = self.ev_df.query('type==@pow_evType').reset_index()

            # error check- make sure we have the same number of events and trials
            if pow_ev_filt.shape[0]!=powMat.shape[0]:
                raise NameError('mismatch between events and power data')

            # apply evQuery filter to events and hilbPow
            if pow_evQuery!=None:
                pow_ev_filt = pow_ev_filt.query(pow_evQuery)
                filt_idx = pow_ev_filt.index.to_numpy()
                powMat = powMat[filt_idx,:]

            self.pow_ev_filt = pow_ev_filt

        # store  vars
        self.powMat = np.copy(powMat)
        self.powMat_mean = np.nanmean(powMat,axis=0)
        self.powMat_sem = stats.sem(powMat,axis=0,nan_policy='omit')
        self.pow_evType=pow_evType
        self.pow_frange_lbl = pow_frange_lbl
        self.pow_evQuery = pow_evQuery
        self.pow_apply_gauss_smoothing = apply_gauss_smoothing
        self.pow_do_zscore = do_zscore
        self.pow_gauss_sd_scaling = 0.1
        self.pow_apply_time_bins=apply_time_bins
        self.pow_time_bin_size_ms=time_bin_size_ms

        if apply_time_bins==True:
            self.pow_xval = self.time_bin_dict['lbl']
        else:
            self.pow_xval = np.linspace(self.ms_to_samples(self.params['tmin_ms']),
                                         self.ms_to_samples(self.params['tmax_ms']),
                                         np.shape(powMat)[1])

    def binPowByRT_2d(self,powMat,rts,num_bins = 10):
      # This function bins power matrix into RT bins. Returns binned powMat data.
      # inputs:
      # powMat ... power matrix
      # rts ... associated RT data for each trial
      # num_bins ... number of bins


      # convert rts to percentiles 
      rts_percentile = (stats.rankdata(rts)/len(rts))*100 

      # create bins
      bins = np.arange(0,100,num_bins)

      # assign each trial to a percentile bin
      # -1 is so that the idx are zero indexed
      rts_bin_idx = np.digitize(rts_percentile,bins)-1

      # initialize containers
      binPow_mean = np.empty((num_bins,powMat.shape[1]))
      binPow_mean[:] = nan

      binPow_sem = np.copy(binPow_mean)

      # loop through bins and populate containers
      for b in np.arange(0,num_bins):
        binPow_mean[b,:] = np.nanmean(powMat[rts_bin_idx==b,:],axis = 0,keepdims=False)
        binPow_sem[b,:] = stats.sem(powMat[rts_bin_idx==b,:],axis = 0,nan_policy = 'omit')

      # return binPow_mean and binPow_sem
      return binPow_mean, binPow_sem, rts_bin_idx, bins 


    def getPhase_2d(self,phs_evType='CC',phs_frange_lbl = 'HFA',phs_method = 'wave',phs_evQuery = None,apply_time_bins=False,time_bin_size_ms=100):
        # loads hilbert/wavelet phase data and
        # computes rvl (resultant vector length; consistency) and preferred phase for the event type, freq range, ev_query
        # It also has the option to take the mean of phase angles within time bins

        # First,  it tries to load saved pickle file of saved phsae , if not available it looks to looks to see if calcTFWrapper has been run, if not it runs calcTFWrapper.
        # Then, it performs its function and saves phase pickle file if specified by params
        #

        # !!!! NOTE:: Buffer has already been removed by calcTFWrapper() (calcWavelet and calcHilbert, respectively)

        # Inputs
        # phs_evType='CC' .....  event type to load
        # phs_frange_lbl = 'HFA'.....  fRange to load
        # phs_method = 'hilb'(or 'wave')
        # phs_evQuery = None .....  event query to apply to filter trials
        # apply_time_bins=False ... whether or not to averaage power data within time bins
        # time_bin_size_ms=100  .... length of the time window to use


        # generate filename
        self.fname_phs2d = self.fname+'-PHASE-'+phs_evType+phs_frange_lbl+phs_method+str(apply_time_bins)+str(time_bin_size_ms)

        # look for saved pickle file of z-scored and downsampled phase
        if (os.path.exists(self.params_dir+self.fname_phs2d)==True) & (self.params['overwriteFlag']==False):

            # load downsampled phase mat
            phsMat = self.load_pickle(self.params_dir+self.fname_phs2d)

            # create time bins dict (for saving purposes)
            self.makeTimeBins(time_bin_size_ms = time_bin_size_ms)

        #else, calculate phase
        else:

            # run init functions if we did not at startup
            if self.do_init==False:
                self.run_init_functions()


            # load TF data
            if phs_method == 'hilb':

                # if TF data has not been calculated
                if hasattr(self,phs_method+'Complex_'+phs_frange_lbl+phs_evType)==False:


                    # run calc TF wrapper (wavelet only)
                    self.calcTFWrapper(do_hilb=True,hilb_frange_lbl=phs_frange_lbl)

                phsMat = self.hilb2phase(phs_frange_lbl,phs_evType)

            elif phs_method == 'wave':
                 # if TF data has not been calculated
                if hasattr(self,phs_method+'Complex_'+phs_evType)==False:
                    # run calc TF wrapper (wavelet only)
                    self.calcTFWrapper(do_hilb=False)

                phsMat = self.wave2phase(phs_evType)

                # identify wavelet that most closely matches the freq range (only for wavelet)
                phsMat = self.phsMat2phsVec(phsMat,freqs=self.wave_freqs,foi = self.params['frange_'+phs_frange_lbl])

            # Optional: average within time bins
            if apply_time_bins==True:
                # bin phs mat into time bins
                phsMat=self.phsMat2timeBins_2d(phsMat, time_bin_size_ms = time_bin_size_ms)

            # save 2d phase
            if self.params['savePhs'] == True:
                self.save_pickle(obj=phsMat,
                                 fpath=self.params_dir+self.fname_phs2d)


        # outside the If statement
        # filter events if this is a real event type
        if phs_evType != 'RANDOM':

            # run bad trials functions if we did not at startup
            if (phs_evQuery!=None):
                if ('badTrial' in phs_evQuery)&(self.do_init==False):
                    self.run_init_functions()

            # load matching events
            # .resetindex() is to get matching indices with erp matrix
            phs_ev_filt = self.ev_df.query('type==@phs_evType').reset_index()

            # error check- make sure we have the same number of events and trials
            if phs_ev_filt.shape[0]!=phsMat.shape[0]:
                raise NameError('mismatch between events and power data')

            # apply evQuery filter to events and hilbphs
            if phs_evQuery!=None:
                phs_ev_filt = phs_ev_filt.query(phs_evQuery)
                filt_idx = phs_ev_filt.index.to_numpy()
                phsMat = phsMat[filt_idx,:]

            self.phs_ev_filt = phs_ev_filt

        # store  vars
        self.phsMat = np.copy(phsMat)
        self.phsMat_rvl = circ.resultant_vector_length(phsMat,axis=0)
        self.phsMat_prefPhase = circ.mean(phsMat,axis=0)
        self.phs_evType=phs_evType
        self.phs_frange_lbl = phs_frange_lbl
        self.phs_evQuery = phs_evQuery
        self.phs_apply_time_bins=apply_time_bins
        self.phs_time_bin_size_ms=time_bin_size_ms

        if apply_time_bins==True:
            self.phs_xval = self.time_bin_dict['lbl']
        else:
            self.phs_xval = np.linspace(self.ms_to_samples(self.params['tmin_ms']),
                                         self.ms_to_samples(self.params['tmax_ms']),
                                         np.shape(phsMat)[1])

    def getAmp_2d(self,amp_evType='CC',amp_frange_lbl = 'HFA',amp_method = 'wave',amp_evQuery = None, do_zscore = True,apply_gauss_smoothing = True,gauss_sd_scaling = 0.1,apply_time_bins=False,time_bin_size_ms=100):
        # loads hilbert/wavelet phase data and
        # computes r-square (consistency) and preferred phase for the event type, freq range, ev_query
        # It also has the option to take the mean of phase angles within time bins

        # First,  it tries to load saved pickle file of saved amplitude , if not available it looks to looks to see if calcTFWrapper has been run, if not it runs calcTFWrapper.
        # Then, it performs its function and saves phase pickle file if specified by params
        #

        # !!!! NOTE:: Buffer has already been removed by calcTFWrapper() (calcWavelet and calcHilbert, respectively)

        # Inputs
        # amp_evType='CC' .....  event type to load
        # amp_frange_lbl = 'HFA'.....  fRange to load
        # amp_method = 'hilb'(or 'wave')
        # amp_evQuery = None .....  event query to apply to filter trials
        # z_score = True .....  z-score by mean and std power for this freq range obtained from random timestamps
        # apply_gauss_smoothing = False .....  apply gaussian smoothing (intended for HFA)
        # gauss_sd_scaling = 0.1.. (samplerate * 0.1) sets the width of gaussian to smooth with.
        # apply_time_bins=False ... whether or not to averaage power data within time bins
        # time_bin_size_ms=100  .... length of the time window to use


        # generate filename
        self.fname_amp2d = self.fname+'-AMPLITUDE-'+amp_evType+amp_frange_lbl+amp_method+str(do_zscore)+str(apply_gauss_smoothing)+str(gauss_sd_scaling)+str(apply_time_bins)+str(time_bin_size_ms)

        # look for saved pickle file of z-scored and downsampled phase
        if (os.path.exists(self.params_dir+self.fname_amp2d)==True) & (self.params['overwriteFlag']==False):

            # load downsampled phase mat
            ampMat = self.load_pickle(self.params_dir+self.fname_amp2d)

            # create time bins dict (for saving purposes)
            self.makeTimeBins(time_bin_size_ms = time_bin_size_ms)

        #else, calculate phase
        else:

            # run init functions if we did not at startup
            if self.do_init==False:
                self.run_init_functions()


            # load TF data
            if amp_method == 'hilb':

                # if TF data has not been calculated
                if hasattr(self,amp_method+'Complex_'+amp_frange_lbl+amp_evType)==False:

                    # run calc TF wrapper (wavelet only)
                    self.calcTFWrapper(do_hilb=True,hilb_frange_lbl=amp_frange_lbl)

                ampMat = self.hilb2amp(amp_frange_lbl,amp_evType)

            elif amp_method == 'wave':
                 # if TF data has not been calculated
                if hasattr(self,amp_method+'Complex_'+amp_evType)==False:
                    # run calc TF wrapper (wavelet only)
                    self.calcTFWrapper(do_hilb=False)

                ampMat = self.wave2amp(amp_evType)

                # Collapse into freq range (only for wavelet)
                # NOTE: Use same function as for power
                ampMat = self.powMat2FreqBin_3d(ampMat,freqs=self.wave_freqs,foi = self.params['frange_'+amp_frange_lbl])

            # Optional: z-score by mean,std power from random events
            if do_zscore==True:

                # get random amp
                if amp_method == 'hilb':
                    randAmpMat = self.hilb2amp(amp_frange_lbl=amp_frange_lbl,amp_evType='RANDOM')

                elif amp_method == 'wave':
                    randAmpMat = self.wave2amp('RANDOM')

                    # collapse into frequency range (for wavelet only)
                    randAmpMat = self.powMat2FreqBin_3d(randAmpMat,freqs=self.wave_freqs,foi = self.params['frange_'+amp_frange_lbl])


                # remove trials in rand amp mat with with "inf"
                inf_idx = np.isinf(np.absolute(np.mean(randAmpMat,axis=1)))
                randAmpMat = randAmpMat[inf_idx==False,:]

                #mean
                randMeanAmp  = np.nanmean(np.nanmean(randAmpMat,axis=1),axis=0)
                randMeanAmp = np.reshape(randMeanAmp,(1,1))
                randMeanAmp = np.repeat(np.repeat(randMeanAmp,ampMat.shape[0],axis=0),
                                            ampMat.shape[1],axis=1)

                #std
                randStdAmp  = np.nanstd(np.nanmean(randAmpMat,axis=1),axis=0)
                randStdAmp = np.reshape(randStdAmp,(1,1))
                randStdAmp = np.repeat(np.repeat(randStdAmp,ampMat.shape[0],axis=0),
                                            ampMat.shape[1],axis=1)
                # z-score
                ampMat = (ampMat-randMeanAmp)/randStdAmp


            # Optional: apply Gaussian smoothing
            if apply_gauss_smoothing==True:
                ampMat = ndimage.filters.gaussian_filter1d(ampMat,sigma =(gauss_sd_scaling*self.samplerate))

            # Optional: average within time bins
            # NOTE: Use same function as for power (no circular mean)
            if apply_time_bins==True:
                # bin amp mat into time bins
                ampMat=self.powMat2timeBins_3d(ampMat, time_bin_size_ms = time_bin_size_ms)

            # save 2d phase
            if self.params['saveAmp'] == True:
                self.save_pickle(obj=ampMat,
                                 fpath=self.params_dir+self.fname_amp2d)


        # outside the If statement
        # filter events if this is a real event type
        if amp_evType != 'RANDOM':

            # run bad trials functions if we did not at startup
            if (amp_evQuery!=None):
                if ('badTrial' in amp_evQuery)&(self.do_init==False):
                    self.run_init_functions()

            # load matching events
            # .resetindex() is to get matching indices with erp matrix
            amp_ev_filt = self.ev_df.query('type==@amp_evType').reset_index()

            # error check- make sure we have the same number of events and trials
            if amp_ev_filt.shape[0]!=ampMat.shape[0]:
                raise NameError('mismatch between events and power data')

            # apply evQuery filter to events and hilbamp
            if amp_evQuery!=None:
                amp_ev_filt = amp_ev_filt.query(amp_evQuery)
                filt_idx = amp_ev_filt.index.to_numpy()
                ampMat = ampMat[filt_idx,:]

            self.amp_ev_filt = amp_ev_filt

        # store  vars
        self.ampMat = np.copy(ampMat)
        self.ampMat_mean = np.nanmean(ampMat,axis=0)
        self.ampMat_sem = stats.sem(ampMat,axis=0,nan_policy='omit')
        self.amp_evType=amp_evType
        self.amp_frange_lbl = amp_frange_lbl
        self.amp_evQuery = amp_evQuery
        self.amp_apply_gauss_smoothing = apply_gauss_smoothing
        self.amp_do_zscore = do_zscore
        self.amp_gauss_sd_scaling = 0.1
        self.amp_apply_time_bins=apply_time_bins
        self.amp_time_bin_size_ms=time_bin_size_ms

        if apply_time_bins==True:
            self.amp_xval = self.time_bin_dict['lbl']
        else:
            self.amp_xval = np.linspace(self.ms_to_samples(self.params['tmin_ms']),
                                         self.ms_to_samples(self.params['tmax_ms']),
                                         np.shape(ampMat)[1])
    def getEEG(self,eeg_evType='CC',eeg_evQuery = None,eeg_apply_filt=True,eeg_filt_fRange=(3,8)):
        # loads eeg data for an event type, applies events filtering, also has the
        # option to filter by a particular frequency range

        # !!!! NOTE: It will remove buffer after applying filter

        # Inputs
        # eeg_evType='CC' .....  event type to load
        # eeg_evQuery = None .....  event query to apply to filter trials
        # eeg_apply_filt.... option to apply filter
        # eeg_filt_frange = (3,8).....  tuple, frange to apply filter

        # run init functions if we did not at startup
        if self.do_init==False:
            self.run_init_functions()

        # load eeg
        epochs = getattr(self,'Epochs_'+eeg_evType)

        # full eeg
        eeg = epochs.get_data().squeeze()

        # apply filter
        if eeg_apply_filt == True:
            eeg_filt = mne.filter.filter_data(data = eeg,
                                              sfreq = self.samplerate,
                                              l_freq = eeg_filt_fRange[0],
                                              h_freq = eeg_filt_fRange[1],
                                              method='fir')
        else:
            eeg_filt = eeg

        # remove buffer
        eeg = eeg[:,int(self.ms_to_samples(self.params['buffer_ms'])):-1*int(self.ms_to_samples(self.params['buffer_ms']))]
        eeg_filt = eeg_filt[:,int(self.ms_to_samples(self.params['buffer_ms'])):-1*int(self.ms_to_samples(self.params['buffer_ms']))]

        # filter events if this is a real event type
        if eeg_evType != 'RANDOM':

            # run bad trials functions if we did not at startup
            if (eeg_evQuery!=None):
                if ('badTrial' in eeg_evQuery)&(self.do_init==False):
                    self.run_init_functions()

            # load matching events
            # .resetindex() is to get matching indices with erp matrix
            eeg_ev_filt = self.ev_df.query('type==@eeg_evType').reset_index()

            # error check- make sure we have the same number of events and trials
            if eeg_ev_filt.shape[0]!=eeg.shape[0]:
                raise NameError('mismatch between events and power data')

            # apply evQuery filter to events
            if eeg_evQuery!=None:
                eeg_ev_filt = eeg_ev_filt.query(eeg_evQuery)
                filt_idx = eeg_ev_filt.index.to_numpy()
                eeg = eeg[filt_idx,:]
                eeg_filt = eeg_filt[filt_idx,:]

            self.eeg_ev_filt = eeg_ev_filt

        # store  vars
        self.eeg = eeg
        self.eeg_filt = eeg_filt
        self.eeg_evType=eeg_evType
        self.eeg_evQuery = eeg_evQuery
        self.eeg_apply_filt=eeg_apply_filt
        self.eeg_filt_fRange = eeg_filt_fRange

        self.eeg_xval = np.linspace(self.ms_to_samples(self.params['tmin_ms']),
                                         self.ms_to_samples(self.params['tmax_ms']),
                                         np.shape(eeg)[1])



    # get continous EEG
    def catEEG(self, cat_method = None):
        #cat_method .... string indicating how to concatenate
        #                trial by trial data to continuous data
        #                'choice' ... will clip from fix start to RT (will error if evType is not CC)
        #                'postCC' ... will clip - 500ms:RT surrounding CC (will error if evType is not CC)
        #                 None ... will defaault to tmin_ms to tmax_ms (any evType)


        # NOTE: If you run this using subjElectrode object, it will concatonate data across sessions
        # error check if eeg has been Done
        if hasattr(self,'eeg') == False:
            raise NameError('run getEEG first')


        # parse cat_method to get pre_lims and post_lims
        if cat_method == None:
            pre_lims = self.params['tmin_ms']
            post_lims = self.params['tmax_ms']
        elif cat_method == 'choice':
            #error check to make sure eeg_evType = 'CC'
            if self.eeg_evType != 'CC':
                raise NameError('run getEEG on choice Events if using "choice" cat_method')
            pre_lims = self.eeg_ev_filt['delay'].to_numpy()*-1
            post_lims = self.eeg_ev_filt['RT'].to_numpy()

            #warning
            if self.params['tmin_ms'] > np.min(pre_lims):
                print('WARNING: tmin_ms is shorter than the longest delay condition')
        elif cat_method == 'postCC':
            #error check to make sure eeg_evType = 'CC'
            if self.eeg_evType != 'CC':
                raise NameError('run getEEG on choice Events if using "choice" cat_method')

            pre_lims = -500
            post_lims = self.eeg_ev_filt['RT'].to_numpy()

        # run mat2vec to get eeg_vec
        # (NOTE: convert xval samples to ms assuming that we are not binning eeg into time bins)
        self.eeg_vec = self.mat2vec(mat= self.eeg,
            xval_ms= self.samples_to_ms(self.eeg_xval),
            pre_lims = pre_lims, post_lims = post_lims)
        self.eeg_cat_method = cat_method

    # find LFO
    def calcLFO(self):
        # this function wraps around fitFOOF. Applies FOOF on continous EEG data
        # it updates params dict with frange_LFO which is the frequency range of
        # the dominant low freq oscillation (LFO)

        self.fitFOOOF(cat_trials = True,bg_param = 'knee',fRange = None)

    # plot LFO
    def plotLFO(self):
        # This wraps around plotFOOOF functions to plot the power spectrum and
        # print results of the FOOOF fit
        self.plotFOOOF_fits()
        self.plotFOOOF_params()
        self.ff_fm.print_results()

    # Apply FOOOF
    def fitFOOOF(self, cat_trials = True,bg_param = 'knee',fRange = None):
        # this function fits FOOOF to eeg data that is contained in self.

        # options:
        #cat_trials....bool; option to concatonate eeg data across trials into
        #                   a single timeseries. If this is True,
        #                   assumes that self contains eeg_vec which is
        #                   continous EEG throughout the session
        #                   (must run  getEEG and catEEG first)



        # Initialize fooof containers (clear previous results)

        # fooof results (initialize parameter containers as nans in case there are no peaks)
        self.ff_fm = [] # for foooof object
        self.ff_aperiodic =  np.array([nan,nan,nan])
        self.ff_LFO_peak =  np.array([nan,nan,nan]) # dominant low freq oscillation
        self.ff_LFO_frange = (nan,nan)
        self.ff_theta =  self.ff_aperiodic.copy()
        self.ff_alpha =  self.ff_aperiodic.copy()
        self.ff_beta =  self.ff_aperiodic.copy()
        self.ff_gamma = self.ff_aperiodic.copy()
        self.ff_r_sq =  []
        self.ff_fit_error = []
        self.ff_cat_trials = cat_trials

        # parse inputs
        if fRange == None:
            fRange = self.params['frange_LFA']

        if cat_trials == True:
            # option to concatonate trial data into a single timeseries
            # error check if cat eeg is Done
            if hasattr(self,'eeg_vec') == False:
                raise NameError('run catEEG first (and getEEG before that)')

            # calculate PSD (without logging)
            self.ff_psd,self.ff_psd_freqs = mne.time_frequency.psd_array_welch(self.eeg_vec,sfreq = self.samplerate, fmin=fRange[0],fmax=fRange[1])

            # Initialize FOOOF model

            self.ff_fm = ff.FOOOF(aperiodic_mode=bg_param) # using knee parameter leads to a better fit over long freq ranges

            # Fit FOOOF model
            self.ff_fm.fit(self.ff_psd_freqs,self.ff_psd,fRange)
        else:
            pass
            raise NameError('Not yet implemented')

        # access fit attributes
        #print('Background params [intercept, (knee), slope]:',fm.background_params_,'\n')
        #print('Peak params [CF: Center Frequencies, Amp: Amplitude Values, BW: Bandwidths]: \n \n',fm.peak_params_,
        #     '\n')
        #print('R Squared of full model fit to log10(psd):', fm.r_squared_,'\n')
        #print('root mean sq error of model fit', fm.error_,'\n')

        #collect model fit results (bg params, peak params, r sq, fit error, gaussian params)
        self.ff_aperiodic,peak_params,self.ff_r_sq,self.ff_fit_error,gauss_params = self.ff_fm.get_results()

        # get dominant low frequency peak (LFA freq range)
        self.ff_LFO_peak = ff.analysis.get_band_peak_fm(self.ff_fm,
        self.params['frange_LFA'], select_highest=True)

        # get dominat low frequency freq range
        self.ff_LFO_frange = ((self.ff_LFO_peak[0]-(self.ff_LFO_peak[2]/2)),(self.ff_LFO_peak[0]+(self.ff_LFO_peak[2]/2)))

        # make sure the floor is set to 2 Hz
        if self.ff_LFO_frange[0]<2:
            self.ff_LFO_frange= (2,self.ff_LFO_frange[1])



        # update params dict with LFO frange (data-driven freq range)
        #self.params['frange_LFO'] = np.copy(self.ff_LFO_frange)
        self.updateParamsDict('frange_LFO',np.copy(self.ff_LFO_frange))

        # parse peak_params into narrowbands
        # loop through peaks
        for i in (arange(0,peak_params.shape[0])):
            cf = peak_params[i][0]
            if (cf>=self.params['frange_theta'][0])&(cf<=self.params['frange_theta'][1]):
                self.ff_theta = peak_params[i]
            elif (cf>=self.params['frange_alpha'][0])&(cf<=self.params['frange_alpha'][1]):
                self.ff_alpha = peak_params[i]
            elif (cf>=self.params['frange_beta'][0])&(cf<=self.params['frange_beta'][1]):
                self.ff_beta = peak_params[i]
            elif (cf>=self.params['frange_gamma'][0])&(cf<=self.params['frange_gamma'][1]):
                self.ff_gamma = peak_params[i]

    def plotFOOOF_fits(self,ax=None):
        if ax==None:
            fig = figure()
            ax = subplot(111)

        # plot fits
        self.ff_fm.plot(ax=ax)

        # title is r-sq
        ax.set_title('r-sq = '+str(np.around(self.ff_r_sq,decimals=2)),loc='right')

        # reset legend
        ax.legend()

    def plotFOOOF_params(self,ax=None,color='blue',alpha = 0.8):
        if ax==None:
            fig = figure()
            ax = subplot(111)


        # aperiodic slope
        plot(0,self.ff_aperiodic[2],'o',color=color,alpha = alpha)

        # for each narrowband (CF vs. amplitude)
        plot(self.ff_theta[0],self.ff_theta[1],'o',color=color,alpha = alpha)
        plot(self.ff_alpha[0],self.ff_alpha[1],'o',color=color,alpha = alpha)
        plot(self.ff_beta[0],self.ff_beta[1],'o',color=color,alpha = alpha)
        plot(self.ff_gamma[0],self.ff_gamma[1],'o',color=color,alpha = alpha)

        xlabel('Peak frequency')
        ylabel('Peak Amplitude')
        ax.set_xticks([0,self.params['frange_theta'][0],
                    self.params['frange_alpha'][0],
                    self.params['frange_beta'][0],
                    self.params['frange_gamma'][0],
                    self.params['frange_gamma'][1]])
        ax.set_xticklabels(['SLOPE',str(self.params['frange_theta'][0]),
                    str(self.params['frange_alpha'][0]),
                    str(self.params['frange_beta'][0]),
                    str(self.params['frange_gamma'][0]),
                    str(self.params['frange_gamma'][1])])


    # PAC FUNCTIONS
    # pac_find optimal amplitude range
    def pac_findAmpRange(self,idpac = (2,0,0),frange = None,frange_lbl = 'LFO',
        n_bins = 50,f_width = 10, n_jobs = 1):
        # this function finds the optimal HFA range for PAC with the provided
        # low frequency range
        # Inputs:
        # idpac = (2,0,0); tuple that sets parameters for tensorpac pac object
        #                  (2,0,0) - modulation index, without surrogates or normalization
        # frange = manual input of frequency range (ovewrites frange lbl)
        # frange_lbl = 'LFO' (uses the corresponding freq range from params_dict)
        # n_bins = 10; number of phase bins to use to bin amplitudes
        # f_width = 10; width of frequency bands to consider in high frequency space
        # n_jobs = 1; number of jobs to run in parallel w tensorpac fcn


        # parse fRange
        if frange == None:
            # parse frange_lbl
            frange = self.params['frange_'+frange_lbl]
        print('Finding optimal amplitude coupling for freq range',frange)


        # if there is no LFO frange found using fooof, update self with nans 
        # this will no occur if a frange is specified
        if np.all(np.isnan(frange)):

            # update self
            self.pac_findAmp_p = nan # pac obj
            self.pac_findAmp_tripac = nan# pac values across amplitude grid search
            self.pac_findAmp_trif = nan # array of amp franges
            self.pac_findAmp_tridx = nan # idx associated with trif
            self.pac_findAmp_frange_pha = nan # frange used for phase
            self.pac_findAmp_frange_amp = nan # optimal amplitude frange

            # update params Dict
            self.updateParamsDict('frange_HFO',(nan,nan))
        else:

            # find optimal amplitude range
            trif, tridx = tp.utils.pac_trivec(f_start=self.params['frange_HFA'][0], f_end=self.params['frange_HFA'][1], f_width=f_width)
            p = tp.Pac(idpac=idpac,f_pha = frange,f_amp = trif,n_bins = n_bins)
            tripac = p.filterfit(int(self.samplerate),self.eeg_vec,
                n_jobs = n_jobs)
            frange_HFO = trif[np.argmax(tripac.mean(-1))]; # optimal amplitude

            # update self
            self.pac_findAmp_p = p # pac obj
            self.pac_findAmp_tripac = np.copy(tripac) # pac values across amplitude grid search
            self.pac_findAmp_trif = np.copy(trif) # array of amp franges
            self.pac_findAmp_tridx = np.copy(tridx) # idx associated with trif
            self.pac_findAmp_frange_pha = frange # frange used for phase
            self.pac_findAmp_frange_amp = frange_HFO # optimal amplitude frange

            # update params Dict
            self.updateParamsDict('frange_HFO',np.copy(frange_HFO))

            print('Optimal amplitude range',frange_HFO)
    def pac_plot_triPlot(self, ax = None):
        # plots pac values across the grid search for the optimal amplitude range

        if np.all(np.isnan(self.pac_findAmp_tripac)):
            print('No tripac data to plot...')
            return

        # run pac_findAmpRange first
        if ax == None:
            fig = figure()
            ax = subplot(111)

        self.pac_findAmp_p.triplot(self.pac_findAmp_tripac.mean(-1),self.pac_findAmp_trif,
            self.pac_findAmp_tridx, cmap='Spectral_r', rmaxis=True,
            title=r'Optimal $[Fmin; Fmax]hz$ band for amplitude')

    def pac_xPAC(self,idpac = (2,3,4), n_bins = 50, n_perm = 1000,frange_pha = None, frange_amp = None,
        frange_pha_lbl = 'LFO', frange_amp_lbl = 'HFO',n_jobs = 1):
        # computes PAC on continous EEG data (eeg_vec). Uses single freq range for
        #phase and amp, respectively. Default, will use LFO (identified by findLFO)
        # and HFO (identified by pac_findAmpRange)

        # Inputs:
        # frange_pha = manual input of frequency range for phase (ovewrites frange lbl)
        # frange_pha_lbl = 'LFO' (uses the corresponding freq range from params_dict)
        # frange_amp = manual input of frequency range for amplitude (ovewrites frange lbl)
        # frange_amp_lbl = 'HFO' (uses the corresponding freq range from params_dict)
        # n_bins = 10; number of phase bins to use to bin amplitudes
        # n_perm = 1000; number of permutations to run when generating null distribution
        # idpac = (2,3,4); tuple that sets parameters for tensorpac pac object
        #                  (2,3,0) - modulation index, time lag surrogates, z-score normalization vs. null
        # n_jobs = 1; number of jobs to run in parallel w tensorpac fcn


        # parse frange_pha
        if frange_pha == None:
            # parse frange_lbl
            frange_pha = self.params['frange_'+frange_pha_lbl]

        # parse frange_amp
        if frange_amp == None:
            # parse frange_lbl
            frange_amp = self.params['frange_'+frange_amp_lbl]


        # if no dominant LFO is found using FOOOF, fill self with nans
        if np.all(np.isnan(frange_pha)):

            # update self w nan
            self.pac_xpac_p = nan
            self.pac_xpac = nan
            self.pac_xpac_shuf = nan
            self.pac_xpac_pval = nan
            self.pac_xpac_idpac = nan
            self.pac_xpac_n_bins = nan
            self.pac_xpac_n_perm = nan
            self.pac_xpac_frange_pha= nan
            self.pac_xpac_frange_pha_lbl = nan
            self.pac_xpac_frange_amp = nan
            self.pac_xpac_frange_amp_lbl = nan

        else:
            print('Finding cont PAC between', frange_pha,' and ', frange_amp)

            # run Pac
            p = tp.Pac(idpac=idpac,f_pha = frange_pha,
                f_amp = frange_amp,n_bins = n_bins)
            xpac = p.filterfit(sf = int(self.samplerate),
                x_pha = self.eeg_vec,
                x_amp = self.eeg_vec,
                n_perm = n_perm,
                n_jobs = n_jobs)

            # calc P Values
            xpac_shuf = p.surrogates.squeeze()
            xpac_pval = np.count_nonzero(xpac_shuf>xpac)/n_perm

            # update self
            self.pac_xpac_p = p
            self.pac_xpac = np.copy(xpac)
            self.pac_xpac_shuf = np.copy(xpac_shuf)
            self.pac_xpac_pval = np.copy(xpac_pval)
            self.pac_xpac_idpac = idpac
            self.pac_xpac_n_bins = n_bins
            self.pac_xpac_n_perm = n_perm
            self.pac_xpac_frange_pha= frange_pha
            self.pac_xpac_frange_pha_lbl = frange_pha_lbl
            self.pac_xpac_frange_amp = frange_amp
            self.pac_xpac_frange_amp_lbl = frange_amp_lbl

    def pac_plot_xpac(self, ax_list=None):
        #This function plots continuous PAC data by showing amplitude binned by phase
        # and null distribution of xpac

        if np.all(np.isnan(self.pac_xpac)):
            print('No xpac data to plot...')
            return

        if ax_list==None:
            fig = figure()
            ax_list = []
            ax_list.append(subplot(121))
            ax_list.append(subplot(122))


        # calc amp binned by phase
        # bin amplitudes by phase
        b_obj = tp.utils.BinAmplitude(x = self.eeg_vec,sf = int(self.samplerate),
                                      f_pha=[int(self.pac_xpac_frange_pha[0]),int(self.pac_xpac_frange_pha[1])],
                                      f_amp=[int(self.pac_xpac_frange_amp[0]),int(self.pac_xpac_frange_amp[1])],
                                      n_bins=self.pac_xpac_n_bins)
        b_obj._phase = np.linspace(-180, 180, self.pac_xpac_n_bins)
        width = 360 / self.pac_xpac_n_bins

        # plot phase/amp distribution in ax[0]
        axes(ax_list[0])
        # divide amp by max to that amp range is 0 - 1
        amp_binned = (b_obj.amplitude/b_obj.amplitude.max())
        bar(b_obj._phase,amp_binned,
        width = width,color = 'red',alpha = 0.5)

        # set ylim dynamically
        xlim(-180,180)
        ylim(np.min(amp_binned)-0.05, 1.05)
        ylabel('Amplitude/max(amplitude)')
        xlabel('Phase Bin')

        # plot null distribution in ax [1]
        axes(ax_list[1])
        hist(self.pac_xpac_shuf,bins = self.pac_xpac_n_bins);
        vlines(self.pac_xpac,ymin = ylim()[0],ymax = ylim()[1],color = 'red')
        title('bootstrap procedure; p = '+str(self.pac_xpac_pval))


    def getPow_3d(self,pow_evType='CC',pow_evQuery = None, do_zscore = True,apply_time_bins=False,time_bin_size_ms=100):
        # This function returns 3d power matrix from wavelet data.
        # This function does not work for hilbert power.
        # It z-scores within frequency by using random Epochs as a baseline.
        # It bins within time windows.
        # First,  it tries to load saved pickle file of downsampled power,
        # if not available it looks to looks to see if calcTFWrapper has been run,
        # if not it runs calcTFWrapper.
        # Then, it performs its function and saves downsampled power pickle file per params

        #!!!! NOTE:: Buffer has already been removed by calcTFWrapper()

        # Inputs:

        # pow_evType='CC' .....  event type to load
        # pow_evQuery = None .....  event query to apply to filter trials
        # do_zscore = True .....  z-score each freq by baseline from random timestamps
        # apply_time_bins=False ... whether or not to averaage power data within time bins
        # time_bin_size_ms=100  .... length of the time window to use


        # generate filename
        self.fname_pow3d = self.fname+'-POWER-'+pow_evType+str(do_zscore)+str(apply_time_bins)+str(time_bin_size_ms)

        # look for saved pickle file of zscored and downsampled power
        if (os.path.exists(self.params_dir+self.fname_pow3d)==True) & (self.params['overwriteFlag']==False):

            # load zscored and downsampled power mat
            powMat = self.load_pickle(self.params_dir+self.fname_pow3d)

            # create time bins dict (for saving purposes)
            self.makeTimeBins(time_bin_size_ms = time_bin_size_ms)

        # need to calculate downsampled power
        else:

            # run init functions if we did not at startup
            if self.do_init==False:
                self.run_init_functions()

            # if TF data has not been calculated
            if hasattr(self,'waveComplex_'+pow_evType)==False:

                # run calc TF wrapper (wavelet only)
                self.calcTFWrapper(do_hilb=False)


            #now, load pow from TF data (3d, freq x time x trial)
            powMat = self.wave2pow(pow_evType)

            # Optional: z-score within freq by mean,std power from random events
            if do_zscore==True:

                # get random power
                randPowMat = self.wave2pow('RANDOM')

                # remove trials in rand pow mat with with "inf"
                inf_idx = np.isinf(np.absolute(np.mean(np.mean(randPowMat,axis=1),axis=0)))
                randPowMat = randPowMat[:,:,inf_idx==False]

                #mean
                randMeanPow  = np.nanmean(np.nanmean(randPowMat,axis=1),axis=1)
                randMeanPow = randMeanPow.reshape((randMeanPow.shape[0],1,1))
                randMeanPow = np.repeat(np.repeat(randMeanPow,
                                                          powMat.shape[1],axis=1),powMat.shape[2],axis=2)

                #std
                randStdPow  = np.nanstd(np.nanmean(randPowMat,axis=1),axis=1)
                randStdPow = randStdPow.reshape((randStdPow.shape[0],1,1))
                randStdPow = np.repeat(np.repeat(randStdPow,powMat.shape[1],axis=1),
                                                powMat.shape[2],axis=2)
                # z-score
                powMat = (powMat-randMeanPow)/randStdPow

            # Downsample by averaging within time bins
            if apply_time_bins==True:
                # bin pow mat into time bins
                powMat =self.powMat2timeBins_3d(powMat, time_bin_size_ms = time_bin_size_ms)

            # Save downsampled power
            if self.params['savePow'] == True:
                self.save_pickle(obj = powMat,
                                 fpath = self.params_dir+self.fname_pow3d)

        # Outside the if/then
        # filter trials if this is a real event type
        if pow_evType != 'RANDOM':

            # run bad trials functions if we did not at startup
            if (pow_evQuery!=None):
                if ('badTrial' in pow_evQuery)&(self.do_init==False):
                    self.run_init_functions()

            # load matching events
            # .resetindex() is to get matching indices with erp matrix
            pow_ev_filt = self.ev_df.query('type==@pow_evType').reset_index()

            # error check- make sure we have the same number of events and trials
            if pow_ev_filt.shape[0]!=powMat.shape[2]:
                raise NameError('mismatch between events and power data')

            # apply evQuery filter to events and hilbPow
            if pow_evQuery!=None:
                pow_ev_filt = pow_ev_filt.query(pow_evQuery)
                filt_idx = pow_ev_filt.index.to_numpy()
                powMat = powMat[:,:,filt_idx]

                self.pow3d_ev_filt = pow_ev_filt

        # store  vars (outside if statement)
        self.pow3d = np.copy(powMat)
        self.pow3d_mean = np.nanmean(powMat,axis=2)
        self.pow3d_sem = stats.sem(powMat,axis=2)
        self.pow3d_evType=pow_evType
        self.pow3d_evQuery = pow_evQuery
        self.pow3d_do_zscore = do_zscore
        self.pow3d_apply_time_bins=apply_time_bins
        self.pow3d_time_bin_size_ms=time_bin_size_ms
        self.pow3d_freqs = self.getWaveFreqs(wave_frange=self.params['wave_frange'],
                                             wave_number_of_freq=self.params['wave_number_of_freq'])
        if apply_time_bins==True:
            self.pow3d_xval = self.time_bin_dict['lbl']
        else:
            self.pow3d_xval = np.linspace(self.ms_to_samples(self.params['tmin_ms']),
                                         self.ms_to_samples(self.params['tmax_ms']),
                                         np.shape(powMat)[1])


    def plotPow_2d(self,ax = None,lbl=None,add_vline=False,fsize_lbl=16,fsize_tick=16,yL=None,alpha = 0.6,color = None,xL_ms = None):
        # This function plots mean and sem of calculated power
        # (uses cached data from the getPow_2d function, so need to run that first)
     
        #plot it
        if ax==None:
            fig = figure()
            ax = subplot(111)

        if lbl == None:
            if self.pow_evQuery==None:
                lbl = 'all trials'
            else:
                lbl = self.pow_evQuery
        # parse xlim
        if xL_ms is None:
            if self.pow_apply_time_bins == False:
                xL_ms = (self.samples_to_ms(self.pow_xval[0]),self.samples_to_ms(self.pow_xval[-1]))
            else:
                xL_ms = (self.pow_xval[0],self.pow_xval[-1])



        ax.plot(self.pow_xval,self.powMat_mean,label=lbl,alpha=alpha,color = color)
        ax.fill_between(self.pow_xval,self.powMat_mean+self.powMat_sem,self.powMat_mean-self.powMat_sem,alpha=alpha-.2,color = color)


        # if x val are in samples, then covert tick labels
        if self.pow_apply_time_bins == False:
            xt = np.array([self.pow_xval[0],0,0.5*self.samplerate,self.pow_xval[-1]])
            ax.set_xticks(xt)
            ax.set_xticklabels(np.round(1000*(xt/self.samplerate)).astype('int'),fontsize=fsize_tick)
            ax.set_xlim(self.ms_to_samples(xL_ms[0]),self.ms_to_samples(xL_ms[1]))
        else:
            ax.set_xlim(xL_ms[0],xL_ms[1])
            xt = np.array([self.pow_xval[0],0,self.pow_xval[np.argmin(np.abs(self.pow_xval-500))],self.pow_xval[-1]])
            ax.set_xticks(xt)
            ax.set_xticklabels(xt,fontsize=fsize_tick)
            ax.set_xlim(xL_ms[0],xL_ms[1])

        ax.set_xlabel('Time from '+self.pow_evType+' (ms)',fontsize=fsize_lbl)
        if self.pow_do_zscore == True:
            ax.set_ylabel('z-score '+self.pow_frange_lbl,fontsize=fsize_lbl)
        else:
            ax.set_ylabel('Power (a.u.)',fontsize=fsize_lbl)

        # set ylim
        if yL != None:
            ax.set_ylim(yL)
        else:
            yL = ax.get_ylim()


        # set yticklabels
        yticks(np.linspace(yL[0], yL[1],5), np.round(np.linspace(yL[0], yL[1],5),2), fontsize=fsize_tick)

        # v line
        if add_vline==True:
            labels = ax.get_xticklabels()
            xtl = np.zeros((len(labels)))
            for i in arange(0,len(labels)):
                xtl[i] = float(labels[i].get_text())

            ax.vlines(x=np.argmin(np.abs(xtl-0)),
                      ymin=ax.get_ylim()[0],
                      ymax=ax.get_ylim()[1],
                      linestyles='--',alpha=0.5)


    # plot RT by pow
    def plotPowByRT_2d(self, ax = None,num_bins = 10,lbl=None,add_vline=True,fsize_lbl=16,fsize_tick=16,yL=None,xL_ms = None,figsize = (8,4),delays_list = [500,1500]):
        # This function plots mean power in various RT bins within each delay condition. (uses cached data from the getPow_2d function, so need to run that first)
    
        # create axes  
        if ax == None:
            fig = figure(figsize=figsize)
            ax = subplot(111)

        if lbl == None:
            if self.pow_evQuery==None:
                lbl = 'all trials'
            else:
                lbl = self.pow_evQuery

        # parse xlim
        if xL_ms == None:
            if self.pow_apply_time_bins == False:
                xL_ms = (self.samples_to_ms(self.pow_xval[0]),self.samples_to_ms(self.pow_xval[-1]))
            else:
                xL_ms = (self.pow_xval[0],self.pow_xval[-1])

        # get variables
        # get pow mat
        powMat = self.powMat

        # get rts
        rts = self.pow_ev_filt['RT'].to_numpy()

        # delay conditions
        delays = self.pow_ev_filt['delay'].to_numpy() 
        if delays_list == None:
            delays_list = np.unique(delays)
        


        # loop through delays
        for d in delays_list:

            # set the color for the delay condition
            if d == 500:
                color = 'C0'
            elif d == 1000:
                color = 'C2'
            elif d == 1500:
                color = 'C1'

            # create RT bins
            binPow_mean, binPow_sem,rts_bin_idx, bins = self.binPowByRT_2d(powMat=powMat[delays==d,:],rts = rts[delays==d],num_bins=num_bins)

            # loop through bins
            for b in np.arange(0,num_bins):
              # plot it (so we dont have to loop again)
              ax.plot(self.pow_xval,binPow_mean[b,:], color = color,alpha = 0.1+(bins[b]/100),label='RT bin'+str(b))
              #ax.fill_between(self.pow_xval,binPow_mean[b,:]+binPow_sem[b,:],alpha=0.1+(bins[b]/100),color = color)

        # if x val are in samples, then covert tick labels
        if self.pow_apply_time_bins == False:
            
            xt = np.array([self.pow_xval[0],0,0.5*self.samplerate,self.pow_xval[-1]])
            ax.set_xticks(xt)
            ax.set_xticklabels(np.round(1000*(xt/self.samplerate)).astype('int'),fontsize=fsize_tick)
            ax.set_xlim(self.ms_to_samples(xL_ms[0]),self.ms_to_samples(xL_ms[1]))
        else:
            ax.set_xlim(xL_ms[0],xL_ms[1])
            xt = np.array([self.pow_xval[0],0,self.pow_xval[np.argmin(np.abs(self.pow_xval-500))],self.pow_xval[-1]])
            ax.set_xticks(xt)
            ax.set_xticklabels(xt,fontsize=fsize_tick)
            ax.set_xlim(xL_ms[0],xL_ms[1])

        
        #set x label
        ax.set_xlabel('Time from '+self.pow_evType+' (ms)',fontsize=fsize_lbl)
       

        #set y label    
        if self.pow_do_zscore == True:
            ax.set_ylabel('z-score '+self.pow_frange_lbl,fontsize=fsize_lbl)
        else:
            ax.set_ylabel('Power (a.u.)',fontsize=fsize_lbl)

        # set ylim
        if yL != None:
            ax.set_ylim(yL)
        else:
            yL = ax.get_ylim()


        # set yticklabels
        yticks(np.linspace(yL[0], yL[1],5), np.round(np.linspace(yL[0], yL[1],5),2), fontsize=fsize_tick)

        # v line
        if add_vline==True:
            # get vL_ticks
            if self.pow_apply_time_bins == False:
                if self.pow_evType=='FIX_START':
                    vL_ticks = [0,int(0.5*self.samplerate),int(1.5*self.samplerate)]
                else:
                    vL_ticks = [0]

            else:
                if self.pow_evType=='FIX_START':
                    vL_ticks= [self.pow_xval[np.argmin(np.abs(self.pow_xval-0))],self.pow_xval[np.argmin(np.abs(self.pow_xval-500))],self.pow_xval[np.argmin(np.abs(self.pow_xval-1500))]]
                else:
                    vL_ticks= [self.pow_xval[np.argmin(np.abs(self.pow_xval-0))]]

            for v in vL_ticks:
                if (v > xL_ms[0]) & (v < xL_ms[1]):
                    ax.vlines(x=v,
                      ymin=ax.get_ylim()[0],
                      ymax=ax.get_ylim()[1],
                      linestyles='--',alpha=0.5)
        # fig layout
        plt.tight_layout()

    def plotPhase_2d(self,ax = None,ax_polar = None,lbl=None,
        add_vline=False,fsize_lbl=16,
        fsize_tick=16,yL=None,alpha = 0.6,color = None,
        bins = 40, plot_rsq_polar = False):
        # This function plots mean and sem of calculated power
        # (uses cached data from the getPow_2d function, so need to run that first)
        #plot it
        if ax==None:
            fig = figure()
            ax = subplot(121)
            ax_polar = subplot(122,projection='polar')

        if lbl == None:
            if self.phs_evQuery==None:
                lbl = 'all trials'
            else:
                lbl = self.phs_evQuery


        ax.plot(self.phs_xval,self.phsMat_rvl)
        max_idx = np.argmax(self.phsMat_rvl)
        ax.plot(self.phs_xval[max_idx],self.phsMat_rvl[max_idx],'or')

        # if x val are in samples, then covert tick labels
        if self.phs_apply_time_bins == False:
            xt = np.array([self.phs_xval[0],0,0.5*self.samplerate,self.phs_xval[-1]])
            ax.set_xticks(xt)
            ax.set_xticklabels(np.round(1000*(xt/self.samplerate)).astype('int'),fontsize=fsize_tick)
        else:
            xt = np.array([self.phs_xval[0],0,self.phs_xval[np.argmin(np.abs(self.phs_xval-500))],self.phs_xval[-1]])
            ax.set_xticks(xt)
            ax.set_xticklabels(xt,fontsize=fsize_tick)



        ax.set_xlabel('Time from '+self.phs_evType+' (ms)',fontsize=fsize_lbl)
        ax.set_ylabel('Resultant Vector Length (a.u.)',fontsize=fsize_lbl)

        # set ylim
        if yL != None:
            ax.set_ylim(yL)
        else:
            yL = ax.get_ylim()


        # set yticklabels
        yticks(np.linspace(yL[0], yL[1],5), np.round(np.linspace(yL[0], yL[1],5),2), fontsize=fsize_tick)

        # v line
        if add_vline==True:
            labels = ax.get_xticklabels()
            xtl = np.zeros((len(labels)))
            for i in arange(0,len(labels)):
                xtl[i] = float(labels[i].get_text())

            ax.vlines(x=np.argmin(np.abs(xtl-0)),
                      ymin=ax.get_ylim()[0],
                      ymax=ax.get_ylim()[1],
                      linestyles='--',alpha=0.5)


        # plot polar hist of max phase consistency
        if plot_rsq_polar == False:
            self.rose_plot(ax_polar, self.phsMat[:,max_idx],bins=bins,density=False)
            #title('time '+str(self.phs_xval[max_idx]))

            ax_polar.set_yticks([])
            ax_polar.set_xticks([deg2rad(0),deg2rad(90),deg2rad(180),deg2rad(270)])
            ax_polar.set_xticklabels(labels=[0,90,180,270],fontdict={'fontsize':16})
            ax_polar.grid(False)
            ax_polar.set_theta_direction(1)
        else:
            ax_polar.plot([0,self.phsMat_prefPhase[max_idx]],[0,self.phsMat_rvl[max_idx]],linewidth = 3,color='C1')

            ax_polar.set_rlim([0,0.5])
            ax_polar.set_rticks([])
            ax_polar.set_xticks([deg2rad(0),deg2rad(90),deg2rad(180),deg2rad(270)])
            ax_polar.set_xticklabels(labels=[0,90,180,270],fontdict={'fontsize':16})
            ax_polar.grid(False)



    #https://stackoverflow.com/questions/22562364/circular-histogram-for-python
    def rose_plot(self,ax, angles, bins=16, density=None, offset=0, lab_unit="degrees",
                  start_zero=False, **param_dict):
        """
        Plot polar histogram of angles on ax. ax must have been created using
        subplot_kw=dict(projection='polar'). Angles are expected in radians.
        """
        # Wrap angles to [-pi, pi)
        angles = (angles + np.pi) % (2*np.pi) - np.pi

        # Set bins symetrically around zero
        if start_zero:
            # To have a bin edge at zero use an even number of bins
            if bins % 2:
                bins += 1
            bins = np.linspace(-np.pi, np.pi, num=bins+1)

        # Bin data and record counts
        count, bin = np.histogram(angles, bins=bins)

        # Compute width of each bin
        widths = np.diff(bin)

        # By default plot density (frequency potentially misleading)
        if density is None or density is True:
            # Area to assign each bin
            area = count / angles.size
            # Calculate corresponding bin radius
            radius = (area / np.pi)**.5
        else:
            radius = count

        # Plot data on ax
        ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
               edgecolor='C0', fill=False, linewidth=1)

        # Set the direction of the zero angle
        ax.set_theta_offset(offset)

        # Remove ylabels, they are mostly obstructive and not informative
        ax.set_yticks([])

        if lab_unit == "radians":
            label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                      r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
            ax.set_xticklabels(label)



    def plotPow_3d(self,ax = None,add_cbar = True,
                   cbar_loc=[.95, 0.1, 0.03, 0.8],clim=(-2,2),
                   fsize_lbl=16,fsize_tick=16):
        # This function plots mean and sem of calculated power
        # (uses cached data from the getPow_3d function, so need to run that first)
        if ax==None:
            fig = figure()
            ax = subplot(111)

        if self.pow3d_evQuery==None:
            lbl = 'all trials'
        else:
            lbl = self.pow3d_evQuery

        # plot mean pow
        cdat = ax.matshow(self.pow3d_mean, aspect='auto', cmap='RdBu_r');
        cdat.set_clim(clim)
        ax.invert_yaxis()
        xlabel('Time (ms) from '+self.pow3d_evType, fontsize=fsize_lbl); ylabel('Frequency (Hz)', fontsize=fsize_lbl)

        # set yticks and labels
        yticks(np.arange(0, len(self.pow3d_freqs)), np.round(self.pow3d_freqs), fontsize=fsize_tick)

        # set x ticks and labels
        if self.pow3d_apply_time_bins == False:
            xticks(linspace(0,np.shape(self.pow3d_mean)[1]-1,5),
               np.round(linspace(self.samples_to_ms(self.pow3d_xval[0]),
                                 self.samples_to_ms(self.pow3d_xval[-1]),5)).astype('str'),fontsize=fsize_tick)
        else:
            xticks(linspace(0,np.shape(self.pow3d_mean)[1]-1,5),
               np.round(linspace(self.pow3d_xval[0],self.pow3d_xval[-1],5)),fontsize=fsize_tick)

        # v line
        labels = ax.get_xticklabels()
        xtl = np.zeros((len(labels)))
        for i in arange(0,len(labels)):
            xtl[i] = float(labels[i].get_text())

        ax.vlines(x=ax.get_xticks()[np.argmin(np.abs(xtl-0))],
                  ymin=ax.get_ylim()[0],
                  ymax=ax.get_ylim()[1],
                  linestyles='--',alpha=0.5)

        # Now adding the colorbar
        if add_cbar == True:
            fig = gcf()
            cax = fig.add_axes(cbar_loc)
            cb = colorbar(mappable=cdat, cax = cax)
            cb.set_label('z-score power',fontsize=fsize_lbl)
            cb.ax.tick_params(labelsize=fsize_lbl)


    def getERP(self,erp_evType = 'RESPONSE',erp_evQuery = None):
        # self must contain MNE Epochs object for each event type (run self.toMNE() to get this)

        # parse erp_evQuery
        if erp_evQuery == '':
            erp_evQuery = None

         # store params
        self.erp_evType = erp_evType
        self.erp_evQuery= erp_evQuery

        # get erp data as a 2d matrix - trials x time
        Epochs = getattr(self,'Epochs_'+erp_evType)
        erp = np.copy(Epochs.get_data().squeeze())

        # remove buffer
        buff_idx = int((self.params['buffer_ms']/1000)*self.samplerate)
        erp = erp[:,buff_idx:-buff_idx]

        # filter events if this is a real event type
        if erp_evType != 'RANDOM':

            # run bad trials functions if we did not at startup
            if (erp_evQuery!=None):
                if ('badTrial' in erp_evQuery)&(self.do_init==False):
                    self.run_init_functions()

            #get matching events for erp_evType
            # .resetindex() is to get matching indices with erp matrix
            erp_ev_filt = self.ev_df.query('type==@erp_evType').reset_index()

            # error check- make sure we have the same number of events and erp trials
            if erp_ev_filt.shape[0]!=erp.shape[0]:
                raise NameError('mismatch between events and erp data')

            # apply filter to erp data
            if erp_evQuery!=None:
                erp_ev_filt = erp_ev_filt.query(erp_evQuery)
                filt_idx = erp_ev_filt.index
                erp = erp[filt_idx,:,:]

            # save filtered events
            self.erp_ev = erp_ev_filt

        # mean and SEM of erp
        samp_min = (self.params['tmin_ms']/1000)*self.samplerate
        samp_max= ((self.params['tmax_ms']/1000)*self.samplerate)+1


        #erp meta data
        self.erp = erp
        self.erp_xval = np.arange(samp_min,samp_max)
        self.erp_mean = np.nanmean(self.erp,axis=0)
        self.erp_sem = stats.sem(erp,axis=0)

    def plotERP(self,ax=None):
        # must run self.getERP() first

        #plot it
        if ax==None:
            fig = figure()
            ax = subplot(111)

        if self.erp_evQuery==None:
            lbl = 'all trials'
        else:
            lbl = self.erp_evQuery


        ax.plot(self.erp_xval,self.erp_mean,label=lbl,alpha=0.6)
        ax.fill_between(self.erp_xval,self.erp_mean+self.erp_sem,self.erp_mean-self.erp_sem,alpha=0.4)

        xt = np.array([self.erp_xval[0],0,0.5*self.samplerate,self.erp_xval[-1]])
        ax.set_xticks(xt)
        ax.set_xticklabels(np.round(1000*(xt/self.samplerate)).astype('int'))
        ax.set_xlabel('Time from '+self.erp_evType+' (ms)')
        ax.set_ylabel('uV')

    # Power Report - by delay and RT
    def plotReport_powByDelay(self,
                              pow_frange_lbl = 'HFA',
                              filt_lbl = 'error==0&fastResponse==0&badTrial==0&',
                              do_zscore = True,
                              apply_time_bins=True,
                              time_bin_size_ms=100,
                              apply_gauss_smoothing = True,
                              gauss_sd_scaling =0.1,
                              pow_method = 'wave',
                              plot_tf = True,
                              alpha = 0.6,
                              cL = (-2,2),
                              yL = (-2,2),
                              fsize_lbl = 10,
                              fsize_tick = 10,
                              figsize=(10,5)):
        #This function plots 2d power and TF power for each delay condition
        # It also plots 2d power separated by RT

        # make figure
        fig = figure(figsize=figsize,facecolor='w')

        # get ev type list
        evType_list = ['FIX_START','CC','RESPONSE']

        # get delay_list
        delay_list = np.unique(self.ev_df['delay'].to_numpy())

        evQuery_list = []
        rtTresh_dict = {}
        for d in delay_list:
            # generate evQuery list
            evQuery_list.append('delay=='+str(d))

            # identify median RT for this condition
            rtTresh_dict['delay=='+str(d)] = np.median(self.ev_df.query(filt_lbl+'delay=='+str(d))['RT'].to_numpy())

        #loop through delay conditions
        count = 0
        for q in evQuery_list:

            # loop thorugh event types
            for e in evType_list:

                # update counter (for subplot, 1 index)
                count+=1

                # create a subplot for 2d pow
                if plot_tf == True:
                    # if we are plotting TF, need to double the rows
                    ax=fig.add_subplot((len(evQuery_list)*2),
                                    len(evType_list),
                                    count)
                else:
                    ax=fig.add_subplot((len(evQuery_list)),
                                    len(evType_list),
                                    count)

                ## set title
                if count == 2:
                    ax.set_title(self.eLbl+'-'+pow_frange_lbl,
                                 fontsize = fsize_lbl+4)


                # plot 2d pow data
                # fast RTs
                self.getPow_2d(pow_evType=e,
                               pow_frange_lbl = pow_frange_lbl,
                               pow_method = pow_method,
                               pow_evQuery = filt_lbl+q+'& RT <= '+str(rtTresh_dict[q]),
                               do_zscore = do_zscore,
                               apply_gauss_smoothing = apply_gauss_smoothing,
                               gauss_sd_scaling = gauss_sd_scaling,
                               apply_time_bins=apply_time_bins,
                               time_bin_size_ms=time_bin_size_ms)
                self.plotPow_2d(ax = ax,
                                fsize_lbl=fsize_lbl,
                                fsize_tick=fsize_tick,
                                yL=yL,add_vline=True,lbl = q+' / Fast RT',alpha = alpha,color = 'C0')

                # slow RTs
                self.getPow_2d(pow_evType=e,
                               pow_frange_lbl = pow_frange_lbl,
                               pow_method = pow_method,
                               pow_evQuery = filt_lbl+q+'& RT > '+str(rtTresh_dict[q]),
                               do_zscore = do_zscore,
                               apply_gauss_smoothing = apply_gauss_smoothing,
                               gauss_sd_scaling = gauss_sd_scaling,
                               apply_time_bins=apply_time_bins,
                               time_bin_size_ms=time_bin_size_ms)
                self.plotPow_2d(ax = ax,
                                fsize_lbl=fsize_lbl,
                                fsize_tick=fsize_tick,
                                yL=yL,add_vline=True,lbl = q+' / Slow RT',alpha = alpha,color = 'C1')

                # add legend
                if count==1:
                    ax.legend(fontsize=fsize_lbl-4)

                # create subplot for TF
                if plot_tf == True:
                    ax = fig.add_subplot((len(evQuery_list)*2),
                                    len(evType_list),
                                    (len(evQuery_list)*len(evType_list))+count)

                    # plot 3d data
                    # parse cbar
                    if (count == (len(evQuery_list)*len(evType_list))):
                        add_cbar = True
                    else:
                        add_cbar = False


                    self.getPow_3d(pow_evType=e,
                                   pow_evQuery = filt_lbl+q,
                                   do_zscore = do_zscore,
                                   apply_time_bins=apply_time_bins,
                                   time_bin_size_ms=time_bin_size_ms)
                    self.plotPow_3d(ax = ax,add_cbar=add_cbar,
                                    cbar_loc=[1, 0.05, 0.03, 0.1],
                                    clim=cL,fsize_lbl = fsize_lbl,
                                    fsize_tick = fsize_tick)

        # fig layout
        fig.tight_layout()


    # Power Report - BY TARGET LOCATION
    def plotReport_powByTargetLoc(self,
                              pow_frange_lbl = 'HFA',
                              filt_lbl = 'error==0&fastResponse==0&badTrial==0&',
                              do_zscore = True,
                              apply_time_bins=True,
                              time_bin_size_ms=100,
                              apply_gauss_smoothing = True,
                              gauss_sd_scaling =0.1,
                              pow_method = 'wave',
                              plot_tf = True,
                              alpha = 0.7,
                              cL = (-2,2),
                              yL = (-2,2),
                              fsize_lbl = 10,
                              fsize_tick = 10,
                              figsize=(10,5)):
        #This function plots 2d power and TF power for each delay condition
        # It also plots 2d power separated by RT

        # make figure
        fig = figure(figsize=figsize,facecolor='w')

        # get ev type list
        evType_list = ['FIX_START','CC','RESPONSE']

        # make a subplot for each ev type
        ax_list=[]
        count = 0
        for i in evType_list:
            count+=1
            ax = fig.add_subplot(1,len(evType_list),count)
            ax_list.append(ax)


        # get delay_list
        targLoc_list = np.unique(self.ev_df['targetLoc_lbl'].to_numpy())

        evQuery_list = []
        rtTresh_dict = {}
        for t in targLoc_list:
            # generate evQuery list
            evQuery_list.append('targetLoc_lbl=="'+t+'"')


        #loop through delay conditions
        count = 0
        for q in evQuery_list:

            # loop thorugh event types
            for e in evType_list:

                # update counter (for subplot, 1 index)
                count+=1

                # parse plot_tf flag
                if plot_tf == False:

                    ax=ax_list[evType_list.index(e)]

                    #choose a color based on a LOCATION

                    # parse y axis
                    if ('north' in q): # choose a blue
                        cmap = get_cmap('Blues')
                        colors = cmap(np.arange(100))

                    elif ('south' in q): # choose a red
                        cmap = get_cmap('Reds')
                        colors = cmap(np.arange(100))

                    else: # choose a green
                        cmap = get_cmap('Greens')
                        colors = cmap(np.arange(100))

                    # parse x axis
                    if ('east' in q):
                        color = colors[70]
                    elif ('west' in q):
                        color = colors[80]
                    else:
                        color = colors[90]

                    # plot 2d pow data
                    self.getPow_2d(pow_evType=e,
                                   pow_frange_lbl = pow_frange_lbl,
                                   pow_method = pow_method,
                                   pow_evQuery = filt_lbl+q,
                                   do_zscore = do_zscore,
                                   apply_gauss_smoothing = apply_gauss_smoothing,
                                   gauss_sd_scaling = gauss_sd_scaling,
                                   apply_time_bins=apply_time_bins,
                                   time_bin_size_ms=time_bin_size_ms)
                    self.plotPow_2d(ax = ax,
                                    fsize_lbl=fsize_lbl,
                                    fsize_tick=fsize_tick,
                                    yL=yL,add_vline=True,lbl = q,alpha = alpha,color = color)



                else:
                    # if we are plotting TF, need a separate subplot for each query
                    ax=fig.add_subplot((len(evQuery_list)),
                                    len(evType_list),
                                    count)

                    # plot 3d data
                    # parse cbar
                    if (count == (len(evQuery_list)*len(evType_list))):
                        add_cbar = True
                    else:
                        add_cbar = False


                    self.getPow_3d(pow_evType=e,
                                   pow_evQuery = filt_lbl+q,
                                   do_zscore = do_zscore,
                                   apply_time_bins=apply_time_bins,
                                   time_bin_size_ms=time_bin_size_ms)
                    self.plotPow_3d(ax = ax,add_cbar=add_cbar,
                                    cbar_loc=[1, 0.05, 0.03, 0.1],
                                    clim=cL,fsize_lbl = fsize_lbl,
                                    fsize_tick = fsize_tick)

            ## set title
            ax_list[1].set_title(self.eLbl+'-'+pow_frange_lbl,
                                 fontsize = fsize_lbl+4)

            # add legend
            ax_list[0].legend(fontsize=fsize_lbl-4)

        # fig layout
        fig.tight_layout()

class SubjElectrode(Electrode):
    # This class inherits from Electrode. It concatenates data across multiple sessions. The main strategy is to maintain a list of Electrode objects. The reason this shouldn't be in Subject is because you want to iherit all the funcitonality from Electrode.
    # and perform functions by interating through the list. It also will populate attributes on an ad-hoc manner (e.g., pow3d_mean etc when getSubjPow is called). Once these attributes are populated, can run Electrode methods (e.g. plotPow3d

    # Constructor
    def __init__(self,subj,sess_idx_list=None,elec1_lbl=None,elec2_lbl=None,paramsDict=None,do_init=True):

        # initialize Subject object
        Subject.__init__(self,subj,paramsDict)

        # parse session list (self.sess_list)
        if sess_idx_list==None:
            self.sess_idx_list = np.arange(0,len(self.sess_list))
        else:
            self.sess_idx_list = sess_idx_list

        # make a list of Electrode objects by looping through sessions

        # initialize list to store Electrode objects
        self.E = []
        isBadElec = []

        # loop through sessions and store objects
        for i in self.sess_idx_list:
            self.E.append(Electrode(subj = subj,
                               sess_idx =self.sess_idx_list[i],
                              elec1_lbl=elec1_lbl,
                              elec2_lbl=elec2_lbl,
                              paramsDict=paramsDict,do_init=do_init))
            isBadElec.append(self.E[i].isBadElectrode)


            # collect ev_df (because it also includes bad trial data)
            # and EEG data
            if i == self.sess_idx_list[0]:
                self.ev_df = self.E[i].ev_df
            else:
                self.ev_df.append(self.E[i].ev_df,ignore_index=True)

        # additional attributes
        self.eLbl = self.E[0].eLbl
        self.isBadElectrode = np.any(isBadElec)
        self.anat = self.E[0].anat
        self.anat_dict = self.E[0].anat_dict

    # update params dictionary
    def updateParamsDict(self,key,val):
        # overwrites similar function in Subject class
        # update SE params dictionary
        self.params[key] = val

        # update each E params dict
        for i in self.sess_idx_list:
            self.E[i].params[key] = val




    # calcTFWrapper
    def calcTFWrapper(self,do_hilb = True,hilb_frange_lbl = 'HFA',do_wave=True):
        # This is a wrapper around calcTFWrapper from Electrode. It loops through all electrode objects and runs this functions
        # loop through list of Electrode objects
        for i in self.sess_idx_list:
            self.E[i].calcTFWrapper(do_hilb = do_hilb,hilb_frange_lbl = 'HFA',do_wave=do_wave)

    # getPow2d
    def getPow_2d(self,pow_evType='CC',pow_frange_lbl = 'HFA',pow_method = 'wave',pow_evQuery = None, do_zscore = True,apply_gauss_smoothing = True,gauss_sd_scaling = 0.075,apply_time_bins=False,time_bin_size_ms=100):
        # This is a wrapper around getPow_2d Electrode function. It can return power data that is z-scored within session etc


        # loop through list of Electrode objects and get pow2d
        for i in self.sess_idx_list:
            self.E[i].getPow_2d(pow_evType=pow_evType,
                                pow_frange_lbl = pow_frange_lbl,
                                pow_method = pow_method,
                                pow_evQuery = pow_evQuery,
                                do_zscore = do_zscore,
                                apply_gauss_smoothing = apply_gauss_smoothing,
                                gauss_sd_scaling =gauss_sd_scaling,
                                apply_time_bins=apply_time_bins,
                                time_bin_size_ms=time_bin_size_ms)


            # concatonate data as we go.
            # for the first session, just copy over all the attributes
            #(except mean and sem that we compute below)
            if i == 0:
                attr_list = ['powMat','pow_evType','pow_frange_lbl','pow_evQuery',
                            'pow_apply_gauss_smoothing','pow_do_zscore',
                            'pow_gauss_sd_scaling','pow_apply_time_bins',
                            'pow_time_bin_size_ms','pow_xval','pow_ev_filt']

                #loop through attr_list and store  vars
                for attr in attr_list:
                    #copy data to self in self
                    setattr(self,attr,getattr(self.E[i],attr))

            #for additional sessions, concatonate powmat and pow_ev_filt
            else:
                self.powMat = np.concatenate(
                    (self.powMat,self.E[i].powMat),
                    axis=0)

                self.pow_ev_filt = self.pow_ev_filt.append(self.E[i].pow_ev_filt)


        # outside the loop, recompute mean and sem
        self.powMat_mean = np.nanmean(self.powMat,axis=0)
        self.powMat_sem = stats.sem(self.powMat,axis=0,nan_policy='omit')

    # getPhase2d
    def getPhase_2d(self,phs_evType='CC',phs_frange_lbl = 'HFA',phs_method = 'hilb',phs_evQuery = None, apply_time_bins=True,time_bin_size_ms=100):
        # This is a wrapper around getPhase_2d Electrode function. It returns phase data across sessions

        # loop through list of Electrode objects and get pow2d
        for i in self.sess_idx_list:
            self.E[i].getPhase_2d(phs_evType=phs_evType,
                                phs_frange_lbl = phs_frange_lbl,
                                phs_method = phs_method,
                                phs_evQuery = phs_evQuery,
                                apply_time_bins=apply_time_bins,
                                time_bin_size_ms=time_bin_size_ms)


            # concatonate data as we go.
            # for the first session, just copy over all the attributes
            #(except mean and sem that we compute below)
            if i == 0:
                attr_list = ['phsMat','phs_evType','phs_frange_lbl','phs_evQuery',
                            'phs_apply_time_bins',
                            'phs_time_bin_size_ms','phs_xval','phs_ev_filt']

                #loop through attr_list and store  vars
                for attr in attr_list:
                    #copy data to self in self
                    setattr(self,attr,getattr(self.E[i],attr))

            #for additional sessions, concatenate phsmat and phs_ev_filt
            else:
                self.phsMat = np.concatenate(
                    (self.phsMat,self.E[i].phsMat),
                    axis=0)

                self.phs_ev_filt = self.phs_ev_filt.append(self.E[i].phs_ev_filt)


        # outside the loop, recompute mean and sem
        self.phsMat_rvl = circ.resultant_vector_length(self.phsMat,axis=0)
        self.phsMat_prefPhase = circ.mean(self.phsMat,axis=0)
    def getAmp_2d(self,amp_evType='CC',amp_frange_lbl = 'HFA',amp_method = 'hilb',amp_evQuery = None, do_zscore = True,apply_gauss_smoothing = True,gauss_sd_scaling = 0.1, apply_time_bins=True,time_bin_size_ms=100):
        # This is a wrapper around getPhase_2d Electrode function. It returns phase data across sessions

        # loop through list of Electrode objects and get pow2d
        for i in self.sess_idx_list:
            self.E[i].getAmp_2d(amp_evType=amp_evType,
                                amp_frange_lbl = amp_frange_lbl,
                                amp_method = amp_method,
                                amp_evQuery = amp_evQuery,
                                do_zscore = do_zscore,
                                apply_gauss_smoothing = apply_gauss_smoothing,
                                gauss_sd_scaling = gauss_sd_scaling,
                                apply_time_bins=apply_time_bins,
                                time_bin_size_ms=time_bin_size_ms)


            # concatonate data as we go.
            # for the first session, just copy over all the attributes
            #(except mean and sem that we compute below)
            if i == 0:
                attr_list = ['ampMat','amp_evType','amp_frange_lbl','amp_evQuery',
                            'amp_apply_gauss_smoothing','amp_do_zscore',
                            'amp_gauss_sd_scaling',
                            'amp_apply_time_bins',
                            'amp_time_bin_size_ms','amp_xval','amp_ev_filt']

                #loop through attr_list and store  vars
                for attr in attr_list:
                    #copy data to self in self
                    setattr(self,attr,getattr(self.E[i],attr))

            #for additional sessions, concatenate ampmat and amp_ev_filt
            else:
                self.ampMat = np.concatenate(
                    (self.ampMat,self.E[i].ampMat),
                    axis=0)

                self.amp_ev_filt = self.amp_ev_filt.append(self.E[i].amp_ev_filt)


        # outside the loop, recompute mean and sem
        self.ampMat_mean = np.nanmean(self.ampMat,axis=0)
        self.ampMat_sem = stats.sem(self.ampMat,axis=0,nan_policy='omit')
    def getEEG(self,eeg_evType='CC',eeg_evQuery = None, eeg_apply_filt=True,eeg_filt_fRange=(3,8)):
        # This is a wrapper around getEEG Electrode function. It returns phase data across sessions

        # loop through list of Electrode objects and get pow2d
        for i in self.sess_idx_list:
            self.E[i].getEEG(eeg_evType=eeg_evType,
                                eeg_evQuery = eeg_evQuery,
                                eeg_apply_filt=eeg_apply_filt,
                                eeg_filt_fRange=eeg_filt_fRange)


            # concatonate data as we go.
            # for the first session, just copy over all the attributes
            #(except mean and sem that we compute below)
            if i == 0:
                attr_list = ['eeg','eeg_filt','eeg_evType','eeg_evQuery',
                            'eeg_apply_filt',
                            'eeg_filt_fRange','eeg_xval','eeg_ev_filt']

                #loop through attr_list and store  vars
                for attr in attr_list:
                    #copy data to self in self
                    setattr(self,attr,getattr(self.E[i],attr))

            #for additional sessions, concatenate eeg and eeg_ev_filt
            else:
                self.eeg = np.concatenate(
                    (self.eeg,self.E[i].eeg),
                    axis=0)
                self.eeg_filt = np.concatenate(
                    (self.eeg_filt,self.E[i].eeg_filt),
                    axis=0)
                self.eeg_ev_filt = self.eeg_ev_filt.append(self.E[i].eeg_ev_filt)


    # getPow3d
    def getPow_3d(self,pow_evType='CC',pow_evQuery = None, do_zscore = True,apply_time_bins=False,time_bin_size_ms=100):
    # This is a wrapper around getPow_3d Electrode function. It can return power data that is z-scored within session etc

    # loop through list of Electrode objects and get pow2d
        for i in self.sess_idx_list:
            self.E[i].getPow_3d(pow_evType=pow_evType,
                                pow_evQuery = pow_evQuery,
                                do_zscore = do_zscore,
                                apply_time_bins=apply_time_bins,
                                time_bin_size_ms=time_bin_size_ms)

            # concatonate data as we go.
            # for the first session, just copy over all the attributes
            #(except mean and sem that we compute below)
            if i == 0:
                attr_list = ['pow3d','pow3d_evType','pow3d_evQuery',
                            'pow3d_do_zscore','pow3d_apply_time_bins',
                            'pow3d_time_bin_size_ms','pow3d_freqs',
                             'pow3d_xval','pow3d_ev_filt']

                #loop through attr_list and store  vars
                for attr in attr_list:
                    #copy data to self in self
                    setattr(self,attr,getattr(self.E[i],attr))

            #for additional sessions, concatonate pow3d nad pow_ev_filt
            else:
                self.pow3d = np.concatenate(
                    (self.pow3d,self.E[i].pow3d),
                    axis=2)

                self.pow3d_ev_filt = self.pow3d_ev_filt.append(self.E[i].pow3d_ev_filt)


        # outside the loop, recompute mean and sem
        self.pow3d_mean = np.nanmean(self.pow3d,axis=2)
        self.pow3d_sem = stats.sem(self.pow3d,axis=2)

    # function to delete Electrode objects for each session to save RAM
    def dropE(self):
        # this function will delete E objects from the SubjElectrode class
        delattr(self,'E')

    # TASK STATS 2d
    #[] consider moving this to the Electrode class
    def doTaskStats_2d(self,pow_frange_lbl = 'HFA',
                        pow_method = 'wave',
                        pow_evQuery = 'error==0&fastResponse==0&badTrial==0', 
                        do_zscore = True,
                        apply_gauss_smoothing = True,
                        gauss_sd_scaling = 0.075,
                        apply_time_bins=True,
                        time_bin_size_ms=100,num_iters=20,
                        regress_yvar_lbl = 'zrrt', 
                        overwriteFlag = False,feat_list_beh = None):
        # This function is a wrapper around various stats that relate a
        # 2d signal (e.g., HFA) to task events and RT
        #if bad electrode, fill with nans (this is incorporated in the subfunctions)


        
        # parse inputs
        self.taskstats2d_pow_frange_lbl = pow_frange_lbl
        self.taskstats2d_pow_method=pow_method
        self.taskstats2d_pow_evQuery=pow_evQuery
        self.taskstats2d_do_zscore=do_zscore
        self.taskstats2d_apply_gauss_smoothing=apply_gauss_smoothing
        self.taskstats2d_gauss_sd_scaling=gauss_sd_scaling
        self.taskstats2d_apply_time_bins=apply_time_bins
        self.taskstats2d_time_bin_size_ms=time_bin_size_ms
        self.taskstats2d_regress_yvar_lbl=regress_yvar_lbl
        self.taskstats2d_num_iters=num_iters
        # parse feat_list
        if feat_list_beh is None:
            self.taskstats2d_feat_list_beh = '' # this is to make it backward compatible
        else:
            self.taskstats2d_feat_list_beh = feat_list_beh
        

        # parse apply_time_bins to generate trange_periEvent_xval and trange_periEventShort_xval. These values are used in functions below to clip specific segments of task-related activity
        if apply_time_bins == True: # then pow_xval is in ms
            self.taskstats2d_trange_periEvent_xval = self.params['trange_periEvent']
            self.taskstats2d_trange_periEventShort_xval = self.params['trange_periEventShort']
            self.taskstats2d_trange_periEventLong_xval = self.params['trange_periEventLong']
        else: # convert to samples because pow_xval is in samples
            self.taskstats2d_trange_periEvent_xval = self.ms_to_samples(self.params['trange_periEvent'])
            self.taskstats2d_trange_periEventShort_xval = self.ms_to_samples(self.params['trange_periEventShort'])
            self.taskstats2d_trange_periEventLong_xval = self.ms_to_samples(self.params['trange_periEventLong'])


        # set filename based on inputs
        self.taskstats2d_fname = (('taskstats2d-'
                                  +self.subj
                                  +'-'
                                   +self.eLbl
                                   +'-'
                                   +self.taskstats2d_pow_frange_lbl
                                   +self.taskstats2d_pow_method
                                   +self.taskstats2d_pow_evQuery
                                   +str(self.taskstats2d_do_zscore)
                                   +str(self.taskstats2d_apply_gauss_smoothing)
                                   +str(self.taskstats2d_gauss_sd_scaling)
                                   +str(self.taskstats2d_apply_time_bins)
                                   +str(self.taskstats2d_time_bin_size_ms))+'num_iters'
                                   +str(self.taskstats2d_num_iters)+str(self.taskstats2d_feat_list_beh))


        # look for saved file
        if (os.path.exists(self.params_dir+self.taskstats2d_fname)==True)&(overwriteFlag==False):

            #load file if it exists
            self.taskstats2d = (self.load_pickle(self.params_dir+
                                                 self.taskstats2d_fname))
        else:

            # initialize dictionary
            self.taskstats2d = {}

            # parse if self.isBadElectrode (return a nan)
            if self.isBadElectrode == True:
                self.taskstats2d=np.nan
 
            else: 
                # update samplerate and anat data
                self.taskstats2d['samplerate'] = self.samplerate
                self.taskstats2d.update(self.anat_dict)

                # model response function
                # model average responses and generate trial by trial data that has been cleaned of "average evoked" responses. This will also generate color change and repsonse locked features (that can be used to extract cc-locked and response locked features)
                self.taskstats_modelTimeCourseAndGetResponseFeatures()

                # set feature list (behavioral features to relate to neural activity)
                if feat_list_beh is None:
                    self.taskstats2d['feat_list_beh'] = ['zrrt']
                    # rt_ms
                    #['zrrt','zrrt_pred', 'zrrt_resid_slow','zrrt_resid_fast','errorMemFast','shortDelayMem']
                else:
                    self.taskstats2d['feat_list_beh'] = feat_list_beh

                # this function orchestrates non-parametric stats between behavioral features and neural activity
                self.taskstats_doStats(num_iters = self.taskstats2d_num_iters)
  
                # run correlate features
                #self.taskstats_correlateFeatures()

                # run event selectivity (anticipatory time analysis,target location ANOVA, error selectivity, feedback selectivity)
                self.taskstats_selectivityAnova()

                # remove fields to save space
                keys_to_del = ['postFix_timeCourseShort_trials','postFix_timeCourseLong_trials','postResponse_timeCourseShort_trials','postResponse_timeCourseLong_trials','postCC_timeCourseShort_trials','postCC_timeCourseLong_trials','response_dt','response','responseL','responseS','responseS_dt','responseL_dt','response_clean','responseS_prestimRemoved_respLocked']
                for k in keys_to_del:
                    del self.taskstats2d[k]

                # Save pickle
                self.save_pickle(obj=self.taskstats2d,fpath=self.params_dir+self.taskstats2d_fname)


    def taskstats_modelTimeCourseAndGetResponseFeatures(self):


        ########## GET TARG LOCKED POWER DATA #################

        # get power for target locked data (we will model this and use this to generate CC locked and response locked time courses)
        # Intiantiate power for the current ev_type
        self.getPow_2d(pow_evType='FIX_START',
                       pow_frange_lbl = self.taskstats2d_pow_frange_lbl,
                       pow_method = self.taskstats2d_pow_method,
                       pow_evQuery = self.taskstats2d_pow_evQuery,
                       do_zscore = self.taskstats2d_do_zscore,
                       apply_gauss_smoothing = self.taskstats2d_apply_gauss_smoothing,
                       gauss_sd_scaling = self.taskstats2d_gauss_sd_scaling,
                       apply_time_bins=self.taskstats2d_apply_time_bins,
                       time_bin_size_ms=self.taskstats2d_time_bin_size_ms)


        # get short and long rt functions on the same time scale. We can plot these (for reference to other modelled functions). Also will calculate offset of gaussian peaks relative to median RTs (in samples)
        shortTrials_bool = self.pow_ev_filt.eval('delay==500').to_numpy() 
        longTrials_bool = self.pow_ev_filt.eval('delay==1500').to_numpy()

        # get xval 
        self.taskstats2d['timeCourse_xval_postFix'] = self.pow_xval

        # time series (for plotting and generating cleaned trial by trial data)
        self.taskstats2d['postFix_timeCourseShort_trials'] = self.powMat[shortTrials_bool,:]
        self.taskstats2d['postFix_timeCourseShort_mean'] = np.nanmean(self.powMat[shortTrials_bool,:],axis=0)
        self.taskstats2d['postFix_timeCourseShort_sem'] = stats.sem(self.powMat[shortTrials_bool,:],axis=0,nan_policy='omit')


        self.taskstats2d['postFix_timeCourseLong_trials'] = self.powMat[longTrials_bool,:]
        self.taskstats2d['postFix_timeCourseLong_mean'] = np.nanmean(self.powMat[longTrials_bool,:],axis=0)
        self.taskstats2d['postFix_timeCourseLong_sem'] = stats.sem(self.powMat[longTrials_bool,:],axis=0,nan_policy='omit')


        ########## IDENTIFY KEY TIME POINTS ################
        mod_xval = self.taskstats2d['timeCourse_xval_postFix']

        # target on
        targOn_idx = np.argmin(np.absolute(mod_xval-0)) 

        # 1000 ms prior to target on (start of pre-event interval)
        preStart_idx = np.argmin(np.absolute(mod_xval+self.taskstats2d_trange_periEventLong_xval))

        # 500 ms prior to target on (end of pre-event interval)
        preEnd_idx = np.argmin(np.absolute(mod_xval+self.taskstats2d_trange_periEvent_xval))

        # first CC 500 ms post
        ccS_idx = np.argmin(np.absolute(mod_xval-self.taskstats2d_trange_periEvent_xval))

        # 1000 ms post 
        ccS_pst_idx = np.argmin(np.absolute(mod_xval-(self.taskstats2d_trange_periEvent_xval+self.taskstats2d_trange_periEvent_xval)))

        # second CC 1500 ms post
        ccL_idx = np.argmin(np.absolute(mod_xval-(self.taskstats2d_trange_periEvent_xval+self.taskstats2d_trange_periEventLong_xval)))

        # second CC 2000ms ms post
        ccL_pst_idx = np.argmin(np.absolute(mod_xval-(self.taskstats2d_trange_periEvent_xval+self.taskstats2d_trange_periEvent_xval+self.taskstats2d_trange_periEventLong_xval)))

        # 2500ms (start of post-event interval)
        pstStart_idx = np.argmin(np.absolute(mod_xval-(self.taskstats2d_trange_periEvent_xval+self.taskstats2d_trange_periEventLong_xval+self.taskstats2d_trange_periEventLong_xval)))

        # 3000 ms (end of post-event interval)
        pstEnd_idx = np.argmin(np.absolute(mod_xval-(self.taskstats2d_trange_periEvent_xval+self.taskstats2d_trange_periEventLong_xval+self.taskstats2d_trange_periEventLong_xval+self.taskstats2d_trange_periEvent_xval)))





        ########## INITIALIZE CONTAINER ################


        # initialize dictionary to hold model resposne features
        responseModel_dict = {}

        # get response functions and xval (default response function is target onset locked)
        responseL_trials = self.taskstats2d['postFix_timeCourseLong_trials']
        responseS_trials = self.taskstats2d['postFix_timeCourseShort_trials']


        responseL = self.taskstats2d['postFix_timeCourseLong_mean']
        responseS = self.taskstats2d['postFix_timeCourseShort_mean']
        responseL_ci = self.taskstats2d['postFix_timeCourseLong_sem']*1.96
        responseS_ci = self.taskstats2d['postFix_timeCourseShort_sem']*1.96


        # mean response function (for model evaluation)
        response = np.nanmean((responseL,responseS),axis=0)
        response_ci = np.nanmean((responseL_ci,responseS_ci),axis=0)
       

        ########## GET RT DATA #################
        
        # get RT, anatomy etc
        # extract zrrt, RT (uses power data instantiated in getResponseFeatures())
        self.taskstats2d['rt'] = self.pow_ev_filt['RT'].to_numpy()
        self.taskstats2d['zrrt'] = stats.zscore(-1/self.taskstats2d['rt'])
        self.taskstats2d['pow_ev_filt'] = self.pow_ev_filt
        self.taskstats2d['shortTrials_bool'] = shortTrials_bool
        self.taskstats2d['longTrials_bool'] = longTrials_bool


        # calculate 'zrrtStoch' (stochastic zrrt)
        zrrtStoch = np.zeros(len(self.taskstats2d['zrrt']))
        zrrtStoch[:] = np.nan
        zrrtStoch[shortTrials_bool] = stats.zscore(self.taskstats2d['zrrt'][shortTrials_bool])
        zrrtStoch[longTrials_bool] = stats.zscore(self.taskstats2d['zrrt'][longTrials_bool])
        self.taskstats2d['zrrtStoch'] = zrrtStoch


        # fit memory model (to get model-based estimates of RTs)
        self.fitMemoryRegression(ev_df = self.taskstats2d['pow_ev_filt'],decay_model='best')

        # add other cogn. vars for for regression analysis
        feat_to_add = ['zrrt_pred','zrrt_resid','zrrt_resid_fast','zrrt_resid_slow','delayCondIsLong','errorMemFast','errorMemSlow','fastResponseMem', 'shortDelayMem','delayMem_conflict']
        for f in feat_to_add:
            self.taskstats2d[f] = self.memReg_dict[f]


        # store rt parameters (assumes memReg has been run). This is for the regression analysis later. Save short and long trials bool so you can filter separately if needed
        responseModel_dict['shortTrials_bool'] = shortTrials_bool
        responseModel_dict['longTrials_bool'] = longTrials_bool


        # get RT data (in samples or ms depending on whether we are binning data). 
        # rts_targLocked indicates indicates the sample when the reaction happens on each trial RELATIVE to target onset (in terms of mod_val). 
        # rtsIdx_targ locked indicates the absolute sample idxfrom start of the time series when response happened on each trial. use this to identify response locked signals
        if self.taskstats2d_apply_time_bins==False:
            responseModel_dict['rts_targLocked_S'] = self.ms_to_samples(self.taskstats2d['rt'][shortTrials_bool]) + mod_xval[ccS_idx]
            responseModel_dict['rts_targLocked_L'] = self.ms_to_samples(self.taskstats2d['rt'][longTrials_bool]) + mod_xval[ccL_idx]

            responseModel_dict['rtsIdx_targLocked_S'] = self.ms_to_samples(self.taskstats2d['rt'][shortTrials_bool]) + ccS_idx
            responseModel_dict['rtsIdx_targLocked_L'] = self.ms_to_samples(self.taskstats2d['rt'][longTrials_bool]) + ccL_idx

        else:
            responseModel_dict['rts_targLocked_S'] = self.taskstats2d['rt'][shortTrials_bool] + mod_xval[ccS_idx]
            responseModel_dict['rts_targLocked_L'] = self.taskstats2d['rt'][longTrials_bool] + mod_xval[ccL_idx]

            responseModel_dict['rtsIdx_targLocked_S'] = self.taskstats2d['rt'][shortTrials_bool] + ccS_idx
            responseModel_dict['rtsIdx_targLocked_L'] = self.taskstats2d['rt'][longTrials_bool] + ccL_idx


        responseModel_dict['rts_targLocked_S_median'] = np.median(responseModel_dict['rts_targLocked_S'])
        responseModel_dict['rts_targLocked_L_median'] = np.median(responseModel_dict['rts_targLocked_L'])




        ################# DEFINE SUBFUNCTIONS ###########
        # define functions to fit (for curve fitting)
        def gaussian_local(x, amp, cen, wid):
            return amp * np.exp(-(x-cen)**2 /((2*wid)**2))
        def line_local(x, m,b):
            return m*x + b

        def clean_trial_data(x,to_rem):
            # x .. 2d array (trials,time)
            # to_rem ... avg function to remove from each trial. Expects 1d array (time).

            # transform into (1,time) 
            to_rem = to_rem[np.newaxis,:]

            # tile to match size of x
            to_rem = np.tile(to_rem,reps=(np.shape(x)[0],1))

            # subtract
            x_new = x - to_rem

            #return cleaned data
            return x_new 
        def toCC(x,cc_idx):
            #transforms target locked trial by trial data to color change locked trial by trial data
            # x .. 2d array (trials,time). Target locked data
            # rt_idx .. int. indicates absolue sample index (from start of time series where color change occured)
            # priorMS and postMS are inferred from self.params['trange_periEventLong'] (default 1000 ms pre and 1000 ms post)
            # identify pre and post (in )
            if self.taskstats2d_apply_time_bins==False:
                pre_offset = self.ms_to_samples(self.params['trange_periEventLong'])
                pst_offset = self.ms_to_samples(self.params['trange_periEventLong'])
            else:
                pre_offset = self.params['trange_periEventLong']
                pst_offset = self.params['trange_periEventLong']

             # calculate xval_respLocked (in samp)
            xval_ccLocked = np.arange(-pre_offset,pst_offset+1)               

            # initialize container
            x_ccLocked = np.zeros((np.shape(x)[0],len(xval_ccLocked)))

            # populate container
            x_ccLocked = x[:,np.floor(cc_idx-pre_offset).astype('int'):np.floor(cc_idx+pst_offset+1).astype('int')]

            #return data
            return x_ccLocked,xval_ccLocked

        def toResponseLocked(x,rt_idx):
            # transforms target-locked trial by trial data to response locked trial by trial data
            # x .. 2d array (trials,time). Target locked data
            # rt_idx .. 1d array(trials). indicates absolue sample index (from start of time series where response was generated). 
            # priorMS and postMS are inferred from self.params['trange_periEventLong'] (default 1000 ms pre and 1000 ms post)

            # identify pre and post (in )
            if self.taskstats2d_apply_time_bins==False:
                pre_offset = self.ms_to_samples(self.params['trange_periEventLong'])
                pst_offset = self.ms_to_samples(self.params['trange_periEventLong'])
            else:
                pre_offset = self.params['trange_periEventLong']
                pst_offset = self.params['trange_periEventLong']                
            # calculate xval_respLocked (in samp)
            xval_respLocked = np.arange(-pre_offset,pst_offset+1)

            # initialize container
            x_respLocked = np.zeros((np.shape(x)[0],len(xval_respLocked)))

            # loop through trials and populate resp locked matrix
            for i in np.arange(0,np.shape(x)[0]):
                x_respLocked[i,:] = x[i,np.floor(rt_idx[i]-pre_offset).astype('int'):np.floor(rt_idx[i]+pst_offset+1).astype('int')]

            #return data
            return x_respLocked,xval_respLocked

        ################# GENERATE RESPONSE AND CC LOCKED POWER DATA #################

        # short delay response locked
        responseS_trials_respLocked,self.taskstats2d['timeCourse_xval_postResponse']=toResponseLocked(responseS_trials,responseModel_dict['rtsIdx_targLocked_S'])
        self.taskstats2d['postResponse_timeCourseShort_trials'] = responseS_trials_respLocked 
        self.taskstats2d['postResponse_timeCourseShort_mean'] = np.nanmean(responseS_trials_respLocked,axis=0)
        self.taskstats2d['postResponse_timeCourseShort_sem'] = stats.sem(responseS_trials_respLocked,axis=0,nan_policy='omit')

        # long delay response locked
        responseL_trials_respLocked,xval_respLocked=toResponseLocked(responseL_trials,responseModel_dict['rtsIdx_targLocked_L'])
        self.taskstats2d['postResponse_timeCourseLong_trials'] = responseL_trials_respLocked
        self.taskstats2d['postResponse_timeCourseLong_mean'] = np.nanmean(responseL_trials_respLocked,axis=0)
        self.taskstats2d['postResponse_timeCourseLong_sem'] = stats.sem(responseL_trials_respLocked,axis=0,nan_policy='omit')

        # short delay CC locked
        responseS_trials_ccLocked,self.taskstats2d['timeCourse_xval_postCC']=toCC(responseS_trials,ccS_idx)
        self.taskstats2d['postCC_timeCourseShort_trials'] = responseS_trials_ccLocked
        self.taskstats2d['postCC_timeCourseShort_mean'] = np.nanmean(responseS_trials_ccLocked,axis=0)
        self.taskstats2d['postCC_timeCourseShort_sem'] = stats.sem(responseS_trials_ccLocked,axis=0,nan_policy='omit')

        # long delay CC locked
        responseL_trials_ccLocked,xval_ccLocked=toCC(responseL_trials,ccL_idx)
        self.taskstats2d['postCC_timeCourseLong_trials'] = responseL_trials_ccLocked
        self.taskstats2d['postCC_timeCourseLong_mean'] = np.nanmean(responseL_trials_ccLocked,axis=0)
        self.taskstats2d['postCC_timeCourseLong_sem'] = stats.sem(responseL_trials_ccLocked,axis=0,nan_policy='omit')



        ######### BASIC STATS TO ASSESS TASK_RESPONSIVE #####

        ## only includes short (500 ms)) and long delay trials (1500 ms); ignores 1000 ms delay in rare cases that it was included
        # baseline (-500 ms to targ on)
        thisIdx_start = preEnd_idx
        thisIdx_end = targOn_idx
        pre_fix_baseline = np.concatenate((np.nanmean(self.powMat[shortTrials_bool,thisIdx_start:thisIdx_end],axis= 1),np.nanmean(self.powMat[longTrials_bool,thisIdx_start:thisIdx_end],axis= 1)))



        #post targ on (-500 ms to targ on)
        thisIdx_start = targOn_idx
        thisIdx_end = ccS_idx
        post_targOn = np.concatenate((np.nanmean(self.powMat[shortTrials_bool,thisIdx_start:thisIdx_end],axis= 1),np.nanmean(self.powMat[longTrials_bool,thisIdx_start:thisIdx_end],axis= 1)))


        # post CC (0-500 post CC)
        thisIdx_start = targOn_idx
        thisIdx_end = ccS_idx
        post_ccOn = np.concatenate((np.nanmean(responseS_trials_ccLocked[:,thisIdx_start:thisIdx_end],axis= 1),np.nanmean(responseL_trials_ccLocked[:,thisIdx_start:thisIdx_end],axis= 1)))



        # post Response (0 - 1000 ms after response)
        thisIdx_start = targOn_idx
        thisIdx_end = ccS_pst_idx
        post_response = np.concatenate((np.nanmean(responseS_trials_respLocked[:,thisIdx_start:thisIdx_end],axis= 1),np.nanmean(responseL_trials_respLocked[:,thisIdx_start:thisIdx_end],axis= 1)))



        # post Fix
        tstat,pval = stats.ttest_1samp(post_targOn-pre_fix_baseline,popmean=0)
        self.taskstats2d['postFix_tstat'] = tstat
        self.taskstats2d['postFix_pval'] = pval

        # post CC
        tstat,pval = stats.ttest_1samp(post_ccOn-pre_fix_baseline,popmean=0)
        self.taskstats2d['postCC_tstat'] = tstat
        self.taskstats2d['postCC_pval'] = pval

        # post Response
        tstat,pval = stats.ttest_1samp(post_response-pre_fix_baseline,popmean=0)
        self.taskstats2d['postResponse_tstat'] = tstat
        self.taskstats2d['postResponse_pval'] = pval



        ################ BEGIN ITERATIVE GAUSSIAN FITTING ########

        ###### step 1 #### - fit line to beginning and end of response function to get linear trend
        idx = np.hstack((np.arange(preStart_idx,preEnd_idx),np.arange(pstStart_idx,pstEnd_idx)))

        # get x and y values
        x = mod_xval[idx]
        y = responseS[idx] # we are using short response function here so we are not biased by event-related changes that occur late

        # do lin regression to get params and stats
        m0,b0,rvalue,pvalue,serr = stats.linregress(x,y)

        # initial guess for line params
        #m0 = (y[-1]-y[0])/(x[-1]-x[0]) # slope
        #b0 = y[np.argmin(x)] # y int
        init_vals = [m0,b0]

        # fit a line
        best_vals,covar = optimize.curve_fit(f=line_local,xdata=x,ydata=y,p0=init_vals)

        # get response trend
        response_trend = line_local(mod_xval,best_vals[0],best_vals[1])

        # detrend the response functions
        responseS_trials_dt = clean_trial_data(responseS_trials,response_trend)
        responseL_trials_dt = clean_trial_data(responseL_trials,response_trend)

        responseL_dt = responseL-response_trend
        responseS_dt = responseS-response_trend
        response_dt = response-response_trend

        # determine threshold (based on prestimulus interval of detrended data)
        dtThresh_pos = np.nanmean(response_dt[preStart_idx:preEnd_idx]) + np.nanmean(response_ci[preStart_idx:preEnd_idx])
        dtThresh_neg = np.nanmean(response_dt[preStart_idx:preEnd_idx]) - np.nanmean(response_ci[preStart_idx:preEnd_idx])

        # update dict
        responseModel_dict['mod_xval'] = mod_xval
        responseModel_dict['response'] = response
        responseModel_dict['responseS'] = responseS
        responseModel_dict['responseL'] = responseL
        responseModel_dict['responseS_trials'] = responseS_trials
        responseModel_dict['responseL_trials'] = responseL_trials


        responseModel_dict['modParams_responseTrend_slope'] = best_vals[0]
        responseModel_dict['modParams_responseTrend_yint'] = best_vals[1]
        responseModel_dict['modParams_responseTrend_rval'] = rvalue
        responseModel_dict['modParams_responseTrend_zval'] = self.fisher_z(rvalue)
        responseModel_dict['mod_responseTrend'] = response_trend

        # detreded repsonse function that we're fitting
        responseModel_dict['response_dt'] = response_dt
        responseModel_dict['responseS_dt'] = responseS_dt
        responseModel_dict['responseL_dt'] = responseL_dt
        responseModel_dict['responseS_trials_dt'] = responseS_trials_dt
        responseModel_dict['responseL_trials_dt'] = responseL_trials_dt

        # threshold values
        responseModel_dict['dtThresh_pos'] = dtThresh_pos # de-trended thresh
        responseModel_dict['dtThresh_neg'] = dtThresh_neg # de-trended thresh

        # clean response functions (these will be interatively updated by removing event-related gaussians and reflect the residual response function after removing all event related phenomena)
        responseModel_dict['response_clean'] = response_dt
        responseModel_dict['responseS_clean'] = responseS_dt
        responseModel_dict['responseL_clean'] = responseL_dt
        responseModel_dict['responseS_trials_clean'] = responseS_trials_dt
        responseModel_dict['responseL_trials_clean'] = responseL_trials_dt


        # define subfunction to fit a gaussian to a response funciton
        def fitGaussWrapper(x,y,responseModel_dict,lbl,respFields_to_update,use_respLocked = False):

            if np.any(y > responseModel_dict['dtThresh_pos']) | np.any(y < responseModel_dict['dtThresh_neg']):

                # parse whether the dominant deflection is positive or negative

                # if both positive, estimate intial parameters based on the larger deflection
                if np.any(y > responseModel_dict['dtThresh_pos']) & np.any(y < responseModel_dict['dtThresh_neg']):

                    # is positive theshold higher or neg threshold higher?
                    pos_dev = np.max(y-responseModel_dict['dtThresh_pos'])
                    neg_dev = np.abs(np.min(y-responseModel_dict['dtThresh_neg']))
                    if pos_dev>=neg_dev:
                        isPos=True
                    else:
                        isPos=False
                elif np.any(y > responseModel_dict['dtThresh_pos']):
                    isPos = True
                else:
                    isPos = False

                #guess initial params
                if isPos == True:
                    amp0 = np.max(y)
                    cen0 = x[np.argmax(y)]
                else:
                    amp0 = np.min(y)
                    cen0 = x[np.argmin(y)]
                
                wid0 = 1
                init_vals = [amp0,cen0,wid0]

                # fit a gaussian
                try:
                    best_vals,covar = optimize.curve_fit(f=gaussian_local,xdata=x,ydata=y,p0=init_vals)
                    # predict this function over the entire course of the recording (modeled response). Parse use_respLocked here

                    if use_respLocked == True:
                        mod_resp = gaussian_local(responseModel_dict['mod_xval_respLocked'],best_vals[0],best_vals[1],best_vals[2])
                    else:
                        mod_resp = gaussian_local(responseModel_dict['mod_xval'],best_vals[0],best_vals[1],best_vals[2])

                    # update dictionary with params and modeled response function (these are in samples)
                    responseModel_dict['modParams_'+lbl+'_amp'] = best_vals[0]
                    responseModel_dict['modParams_'+lbl+'_cen'] = best_vals[1]
                    responseModel_dict['modParams_'+lbl+'_wid'] = best_vals[2]
                    responseModel_dict['mod_'+lbl] = mod_resp

                    # convert center and width to ms (so we can compare across subjects (w/ different sampling rates)) 
                    responseModel_dict['modParams_'+lbl+'_cen_ms'] = self.samples_to_ms(best_vals[1])
                    responseModel_dict['modParams_'+lbl+'_wid_ms'] = self.samples_to_ms(best_vals[2])

                    # calculate center parameters in terms of median rt offset
                    if lbl in ['postCCS','periResponseS']:
                        responseModel_dict['modParams_'+lbl+'_cen_rtoffset_s'] = self.samples_to_ms(best_vals[1]-np.median(responseModel_dict['rts_targLocked_S']))/1000
                    elif lbl in ['postCCL','periResponseL']:
                        responseModel_dict['modParams_'+lbl+'_cen_rtoffset_s'] = self.samples_to_ms(best_vals[1]-np.median(responseModel_dict['rts_targLocked_L']))/1000

                    # update _clean response function
                    for f in respFields_to_update:
                        # parse if trial by trial data

                        if 'trials' in f:
                            responseModel_dict[f] = clean_trial_data(responseModel_dict[f],mod_resp)
                        else:
                            responseModel_dict[f] = responseModel_dict[f]-mod_resp

                except RuntimeError:# handle cases where we could not find a good fit

                    #update dict with nans
                    responseModel_dict['modParams_'+lbl+'_amp'] = 0
                    responseModel_dict['modParams_'+lbl+'_cen'] = np.nan
                    responseModel_dict['modParams_'+lbl+'_wid'] = np.nan
                    responseModel_dict['modParams_'+lbl+'_cen_ms'] = np.nan
                    responseModel_dict['modParams_'+lbl+'_wid_ms'] = np.nan
                    if use_respLocked == True:
                        responseModel_dict['mod_'+lbl] = np.zeros(len(responseModel_dict['mod_xval_respLocked']))
                    else:
                        responseModel_dict['mod_'+lbl] = np.zeros(len(responseModel_dict['mod_xval']))
                    if lbl in ['postCCS','periResponseS']:
                        responseModel_dict['modParams_'+lbl+'_cen_rtoffset_s'] = np.nan
                    elif lbl in ['postCCL','periResponseL']:
                        responseModel_dict['modParams_'+lbl+'_cen_rtoffset_s'] = np.nan
            else:
                #update dict with nans
                responseModel_dict['modParams_'+lbl+'_amp'] = 0
                responseModel_dict['modParams_'+lbl+'_cen'] = np.nan
                responseModel_dict['modParams_'+lbl+'_wid'] = np.nan
                responseModel_dict['modParams_'+lbl+'_cen_ms'] = np.nan
                responseModel_dict['modParams_'+lbl+'_wid_ms'] = np.nan
                if use_respLocked == True:
                    responseModel_dict['mod_'+lbl] = np.zeros(len(responseModel_dict['mod_xval_respLocked']))
                else:
                    responseModel_dict['mod_'+lbl] = np.zeros(len(responseModel_dict['mod_xval']))


                if lbl in ['postCCS','periResponseS']:
                    responseModel_dict['modParams_'+lbl+'_cen_rtoffset_s'] = np.nan
                elif lbl in ['postCCL','periResponseL']:
                    responseModel_dict['modParams_'+lbl+'_cen_rtoffset_s'] = np.nan
            return responseModel_dict

        ##### step 2 #####  fit targ onset gaussian based on 500 ms activity from (target onset to post CC short)
        idx = np.arange(targOn_idx,ccS_idx)

        # fit a gaussian based on this response curve and update the modelResponse dict
        responseModel_dict = fitGaussWrapper(x=mod_xval[idx],y=response_dt[idx],responseModel_dict=responseModel_dict,lbl='postTargOn',respFields_to_update=['response_clean','responseS_clean','responseL_clean','responseS_trials_clean','responseL_trials_clean'])

        ##### step 3 #### 
        # generate response-locked data (dt and clean) for short delay trials after removing target onset effects. Fit response locked gaussians (pre and post) and clean responseLocked data. Note that postCC responses are still here but will be modeled relative to response here. 

        # generate response-locked clean trial by trial data for short delay trial responses and mod_xval_resp. Also save a snapshot of respLocked avg response function before continuing to clean it so we can evaluate our model

        responseModel_dict['responseS_trials_clean_respLocked'],mod_xval_respLocked = toResponseLocked(responseModel_dict['responseS_trials_clean'],responseModel_dict['rtsIdx_targLocked_S'])

        responseModel_dict['mod_xval_respLocked'] = mod_xval_respLocked
        responseModel_dict['responseS_prestimRemoved_respLocked'] = np.nanmean(responseModel_dict['responseS_trials_clean_respLocked'],axis=0)
        responseModel_dict['responseS_clean_respLocked'] = np.nanmean(responseModel_dict['responseS_trials_clean_respLocked'],axis=0)

        #fit peri-response gaussians to response locked data. First to pre-move interval (500 ms prior: zero_idx)
        preIdx_respLocked = np.argmin(np.absolute(mod_xval_respLocked+self.taskstats2d_trange_periEvent_xval))
        zeroIdx_respLocked = np.argmin(np.abs(mod_xval_respLocked)) 
        pstIdx_respLocked = np.argmin(np.absolute(mod_xval_respLocked-self.taskstats2d_trange_periEvent_xval))
          
        idx = np.arange(preIdx_respLocked,zeroIdx_respLocked)

        responseModel_dict = fitGaussWrapper(x=mod_xval_respLocked[idx],y=responseModel_dict['responseS_clean_respLocked'][idx],responseModel_dict=responseModel_dict,lbl='preResponseS_respLocked',respFields_to_update=['responseS_clean_respLocked','responseS_trials_clean_respLocked'],use_respLocked=True)


        # and then to post-move interval
        pstIdx_respLocked = np.argmin(np.absolute(mod_xval_respLocked-self.taskstats2d_trange_periEvent_xval))
        idx = np.arange(zeroIdx_respLocked,pstIdx_respLocked)

        responseModel_dict = fitGaussWrapper(x=mod_xval_respLocked[idx],y=responseModel_dict['responseS_clean_respLocked'][idx],responseModel_dict=responseModel_dict,lbl='postResponseS_respLocked',respFields_to_update=['responseS_clean_respLocked','responseS_trials_clean_respLocked'],use_respLocked=True)

        ###### step 4 #########
        # model target locked short delay trials (after having removed target onset effects, response locked effects are still here but will be modeled relative to CC onset here
        idx =np.arange(ccS_idx,ccS_pst_idx)

        # fit a gaussian and update the modelResponse dict
        # here we use clean short response function with target onset gaussian removed
        responseModel_dict = fitGaussWrapper(x=mod_xval[idx],y=responseModel_dict['responseS_clean'][idx],responseModel_dict=responseModel_dict,lbl='postCCS',respFields_to_update=['responseS_clean','responseS_trials_clean'])

        
        # fit a gaussian to capture any residual periResponse activity
        # here we use clean short response function with target onset gaussian removed. Use a 1000 ms interval post-color change
        idx =np.arange(ccS_idx,ccL_idx)
        responseModel_dict = fitGaussWrapper(x=mod_xval[idx],y=responseModel_dict['responseS_clean'][idx],responseModel_dict=responseModel_dict,lbl='periResponseS',respFields_to_update=['responseS_clean','responseS_trials_clean'])
        
        ###### step 5 #########
        # fit long delay response - short color change expectation
        idx =np.arange(ccS_idx,ccL_idx) # 500ms - 1500 ms (to capture any ramping activity)
        responseModel_dict = fitGaussWrapper(x=mod_xval[idx],y=responseModel_dict['responseL_clean'][idx],responseModel_dict=responseModel_dict,lbl='postNoCCS',respFields_to_update=['responseL_clean','responseL_trials_clean'])

        ###### step 6 #########
        # generate response-locked dt clean trial by trial data for long delay trial responses and mod_xval_resp. any expectation related efffects that emerge after the absence of the short delay cue have been removied

        responseModel_dict['responseL_trials_clean_respLocked'],mod_xval_respLocked = toResponseLocked(responseModel_dict['responseL_trials_clean'],responseModel_dict['rtsIdx_targLocked_L']  )
    
        responseModel_dict['responseL_prestimRemoved_respLocked'] = np.nanmean(responseModel_dict['responseL_trials_clean_respLocked'],axis=0)    
        responseModel_dict['responseL_clean_respLocked'] = np.nanmean(responseModel_dict['responseL_trials_clean_respLocked'],axis=0)


        #fit pre-response gaussians to response locked data. First to pre-move interval (500 ms prior: zero_idx)
        idx = np.arange(preIdx_respLocked,zeroIdx_respLocked)

        responseModel_dict = fitGaussWrapper(x=mod_xval_respLocked[idx],y=responseModel_dict['responseL_clean_respLocked'][idx],responseModel_dict=responseModel_dict,lbl='preResponseL_respLocked',respFields_to_update=['responseL_clean_respLocked','responseL_trials_clean_respLocked'],use_respLocked=True)

        # and then to post-move interval
        idx = np.arange(zeroIdx_respLocked,pstIdx_respLocked)

        responseModel_dict = fitGaussWrapper(x=mod_xval_respLocked[idx],y=responseModel_dict['responseL_clean_respLocked'][idx],responseModel_dict=responseModel_dict,lbl='postResponseL_respLocked',respFields_to_update=['responseL_clean_respLocked','responseL_trials_clean_respLocked'],use_respLocked=True)

        ###### step 7 #########
        # fit long delay - color change gaussian. Again, this includes any potential response locked signals that will be modeled relative to color change (as in the case of short delay trials)

        idx =np.arange(ccL_idx,ccL_pst_idx)

        # fit a gaussian and update the modelResponse dict
        # here we use clean long response function with target onset gaussian removed
        responseModel_dict = fitGaussWrapper(x=mod_xval[idx],y=responseModel_dict['responseL_clean'][idx],responseModel_dict=responseModel_dict,lbl='postCCL',respFields_to_update=['responseL_clean','responseL_trials_clean'])

        # fit any residual peri-response signals (to 1000 ms post CC)
        idx =np.arange(ccL_idx,pstStart_idx)
        responseModel_dict = fitGaussWrapper(x=mod_xval[idx],y=responseModel_dict['responseL_clean'][idx],responseModel_dict=responseModel_dict,lbl='periResponseL',respFields_to_update=['responseL_clean','responseL_trials_clean'])

        # generate CC-locked data (dt and trial by trial)
        # short delay trials
        responseModel_dict['responseS_trials_dt_ccLocked'],mod_xval_ccLocked = toCC(responseModel_dict['responseS_trials_dt'],ccS_idx)
        responseModel_dict['responseS_trials_clean_ccLocked'],mod_xval_ccLocked = toCC(responseModel_dict['responseS_trials_clean'],ccS_idx)

        responseModel_dict['mod_xval_ccLocked'] = mod_xval_ccLocked
        responseModel_dict['responseS_dt_ccLocked'] = np.nanmean(responseModel_dict['responseS_trials_dt_ccLocked'],axis=0)
        responseModel_dict['responseS_clean_ccLocked'] = np.nanmean(responseModel_dict['responseS_trials_clean_ccLocked'],axis=0)

        # long delay trials
        responseModel_dict['responseL_trials_dt_ccLocked'],mod_xval_ccLocked = toCC(responseModel_dict['responseL_trials_dt'],ccL_idx)
        responseModel_dict['responseL_trials_clean_ccLocked'],mod_xval_ccLocked = toCC(responseModel_dict['responseL_trials_clean'],ccL_idx)


        responseModel_dict['responseL_dt_ccLocked'] = np.nanmean(responseModel_dict['responseL_trials_dt_ccLocked'],axis=0)
        responseModel_dict['responseL_clean_ccLocked'] = np.nanmean(responseModel_dict['responseL_trials_clean_ccLocked'],axis=0)

        ###### step 8######### evaluate model fits to response function
        # predict response functions (z-scored)
        responseModel_dict['responseS_z']=stats.zscore(responseModel_dict['responseS'])
        responseModel_dict['responseL_z']=stats.zscore(responseModel_dict['responseL'])
        responseModel_dict['responseS_respLocked_z']=stats.zscore(responseModel_dict['responseS_prestimRemoved_respLocked'])
        responseModel_dict['responseL_respLocked_z']=stats.zscore(responseModel_dict['responseL_prestimRemoved_respLocked'])



        # model predictions of each function (z-scored)
        responseModel_dict['mod_fullS'] = np.nanmean(np.vstack((responseModel_dict['mod_responseTrend'],responseModel_dict['mod_postTargOn'],responseModel_dict['mod_postCCS'],responseModel_dict['mod_periResponseS'])),axis=0)
        if np.all(responseModel_dict['mod_fullS']==0)==False:
            responseModel_dict['mod_fullS'] = stats.zscore(responseModel_dict['mod_fullS'])

        responseModel_dict['mod_fullL'] = np.nanmean(np.vstack((responseModel_dict['mod_responseTrend'],responseModel_dict['mod_postTargOn'],responseModel_dict['mod_postNoCCS'],responseModel_dict['mod_postCCL'],responseModel_dict['mod_periResponseL'])),axis=0)
        if np.all(responseModel_dict['mod_fullL']==0)==False:
            responseModel_dict['mod_fullL'] = stats.zscore(responseModel_dict['mod_fullL'])

        responseModel_dict['mod_fullS_respLocked'] = np.nanmean(np.vstack((responseModel_dict['mod_preResponseS_respLocked'],responseModel_dict['mod_postResponseS_respLocked'])),axis=0)
        if np.all(responseModel_dict['mod_fullS_respLocked']==0)==False:
            responseModel_dict['mod_fullS_respLocked'] = stats.zscore(responseModel_dict['mod_fullS_respLocked'])

        responseModel_dict['mod_fullL_respLocked'] = np.nanmean(np.vstack((responseModel_dict['mod_preResponseL_respLocked'],responseModel_dict['mod_postResponseL_respLocked'])),axis=0)
        if np.all(responseModel_dict['mod_fullL_respLocked']==0)==False:
            responseModel_dict['mod_fullL_respLocked'] = stats.zscore(responseModel_dict['mod_fullL_respLocked'])

        # calculate r-square
        responseModel_dict['mod_fullS_rsq'] = r2_score(y_true=responseModel_dict['responseS_z'],y_pred=responseModel_dict['mod_fullS'])
        responseModel_dict['mod_fullL_rsq'] = r2_score(y_true=responseModel_dict['responseL_z'],y_pred=responseModel_dict['mod_fullL'])        
        responseModel_dict['mod_fullS_respLocked_rsq'] = r2_score(y_true=responseModel_dict['responseS_respLocked_z'],y_pred=responseModel_dict['mod_fullS_respLocked'])
        responseModel_dict['mod_fullL_respLocked_rsq'] = r2_score(y_true=responseModel_dict['responseL_respLocked_z'],y_pred=responseModel_dict['mod_fullL_respLocked'])


        ################ FEATURE EXTRACTION ON TRIAL BY TRIAL DATA ################ 
        # Both for clean trial by trial data and raw trial by trial data


        def combineTrialsVec(shortTrials_bool,longTrials_bool,shortTrials_vec, longTrials_vec):

            # initialize containers
            x = np.zeros((len(shortTrials_bool)))
            x[:] = np.nan

            # fill in short delay (500) and long delay trials (1500). in rare cases where there are additional delay conditions (delay = 1000), then it fills in nans
            x[shortTrials_bool] = shortTrials_vec
            x[longTrials_bool] = longTrials_vec

            return x
        def combineTrialsMat(shortTrials_bool,longTrials_bool,shortTrials_mat, longTrials_mat):
            # combines short and long delay trials in original interleaved order that they were presented

            # initialize containers
            x = np.zeros((len(shortTrials_bool),np.shape(shortTrials_mat)[1]))
            x[:] = np.nan

            # fill in short delay (500) and long delay trials (1500). in rare cases where there are additional delay conditions (delay = 1000), then it fills in nans
            x[shortTrials_bool,:] = shortTrials_mat
            x[longTrials_bool,:] = longTrials_mat

            return x


        ########### get key peri-event segments ##########

        # target-locked segment (500 ms prior to 500 ms post target onset)
        responseModel_dict['mod_xval_periTarg'] = responseModel_dict['mod_xval'][preEnd_idx:ccS_idx]
        responseModel_dict['responseS_trials_clean_periTarg'] =  responseModel_dict['responseS_trials_clean'][:,preEnd_idx:ccS_idx]
        responseModel_dict['responseL_trials_clean_periTarg'] =  responseModel_dict['responseL_trials_clean'][:,preEnd_idx:ccS_idx]
        responseModel_dict['response_trials_clean_periTarg'] = combineTrialsMat(shortTrials_bool, longTrials_bool,responseModel_dict['responseS_trials_clean_periTarg'],responseModel_dict['responseL_trials_clean_periTarg'])
       
        # pre CC segment (1000 ms prior to color change)
        responseModel_dict['mod_xval_ccLocked_preCC'] = responseModel_dict['mod_xval_ccLocked'][preStart_idx:targOn_idx]
        responseModel_dict['responseS_trials_clean_ccLocked_preCC'] =  responseModel_dict['responseS_trials_clean_ccLocked'][:,preStart_idx:targOn_idx]
        responseModel_dict['responseL_trials_clean_ccLocked_preCC'] =  responseModel_dict['responseL_trials_clean_ccLocked'][:,preStart_idx:targOn_idx]
        responseModel_dict['response_trials_clean_ccLocked_preCC'] = combineTrialsMat(shortTrials_bool, longTrials_bool,responseModel_dict['responseS_trials_clean_ccLocked_preCC'],responseModel_dict['responseL_trials_clean_ccLocked_preCC'])

        # post CC segment (color change to 1000 ms post)
        responseModel_dict['mod_xval_ccLocked_postCC'] = responseModel_dict['mod_xval_ccLocked'][targOn_idx:ccS_pst_idx]
        responseModel_dict['responseS_trials_clean_ccLocked_postCC'] =  responseModel_dict['responseS_trials_clean_ccLocked'][:,targOn_idx:ccS_pst_idx]
        responseModel_dict['responseL_trials_clean_ccLocked_postCC'] =  responseModel_dict['responseL_trials_clean_ccLocked'][:,targOn_idx:ccS_pst_idx]
        responseModel_dict['response_trials_clean_ccLocked_postCC'] = combineTrialsMat(shortTrials_bool, longTrials_bool,responseModel_dict['responseS_trials_clean_ccLocked_postCC'],responseModel_dict['responseL_trials_clean_ccLocked_postCC'])


        # pre response segment (1000 ms prior to response)
        responseModel_dict['mod_xval_respLocked_preResp'] = responseModel_dict['mod_xval_respLocked'][preStart_idx:targOn_idx]
        responseModel_dict['responseS_trials_clean_respLocked_preResp'] =  responseModel_dict['responseS_trials_clean_respLocked'][:,preStart_idx:targOn_idx]
        responseModel_dict['responseL_trials_clean_respLocked_preResp'] =  responseModel_dict['responseL_trials_clean_respLocked'][:,preStart_idx:targOn_idx]
        responseModel_dict['response_trials_clean_respLocked_preResp'] = combineTrialsMat(shortTrials_bool, longTrials_bool,responseModel_dict['responseS_trials_clean_respLocked_preResp'],responseModel_dict['responseL_trials_clean_respLocked_preResp'])


        # post response segment (response to 1000 ms post)
        responseModel_dict['mod_xval_respLocked_postResp'] = responseModel_dict['mod_xval_respLocked'][targOn_idx:ccS_pst_idx]
        responseModel_dict['responseS_trials_clean_respLocked_postResp'] =  responseModel_dict['responseS_trials_clean_respLocked'][:,targOn_idx:ccS_pst_idx]
        responseModel_dict['responseL_trials_clean_respLocked_postResp'] =  responseModel_dict['responseL_trials_clean_respLocked'][:,targOn_idx:ccS_pst_idx]
        responseModel_dict['response_trials_clean_respLocked_postResp'] = combineTrialsMat(shortTrials_bool, longTrials_bool,responseModel_dict['responseS_trials_clean_respLocked_postResp'],responseModel_dict['responseL_trials_clean_respLocked_postResp'])

        ########### get key peri-event features ##########
        #S0f - baseline activity prior to target onset (-500 to 0)

        # raw
        responseModel_dict['S0fS'] = np.nanmean(responseModel_dict['responseS_trials'][:,preEnd_idx:targOn_idx],axis=1)
        responseModel_dict['S0fL'] = np.nanmean(responseModel_dict['responseL_trials'][:,preEnd_idx:targOn_idx],axis=1)
        responseModel_dict['S0f'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['S0fS'], responseModel_dict['S0fL'])

        # clean
        responseModel_dict['S0fS_clean'] = np.nanmean(responseModel_dict['responseS_trials_clean'][:,preEnd_idx:targOn_idx],axis=1)
        responseModel_dict['S0fL_clean'] = np.nanmean(responseModel_dict['responseL_trials_clean'][:,preEnd_idx:targOn_idx],axis=1)
        responseModel_dict['S0f_clean'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['S0fS_clean'], responseModel_dict['S0fL_clean'])


        #S0c -  activity prior to color change (-500 ms to 0). Will include post target onset response for short delay trials
        # raw
        responseModel_dict['S0cS'] = np.nanmean(self.taskstats2d['postCC_timeCourseShort_trials'][:,preEnd_idx:targOn_idx],axis=1)
        responseModel_dict['S0cL'] = np.nanmean(self.taskstats2d['postCC_timeCourseLong_trials'][:,preEnd_idx:targOn_idx],axis=1)
        responseModel_dict['S0c'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['S0cS'], responseModel_dict['S0cL'])


        # clean
        responseModel_dict['S0cS_clean'] = np.nanmean(responseModel_dict['responseS_trials_clean_ccLocked'][:,preEnd_idx:targOn_idx],axis=1)
        responseModel_dict['S0cL_clean'] = np.nanmean(responseModel_dict['responseL_trials_clean_ccLocked'][:,preEnd_idx:targOn_idx],axis=1)
        responseModel_dict['S0c_clean'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['S0cS_clean'], responseModel_dict['S0cL_clean'])


        def getBuildupRate(preResponseMat):
            # calculate slope for each trial. Can also input a postCC mat to compute post stimulus rate of rise

            x = np.zeros((np.shape(preResponseMat)[0]))

            for i in np.arange(0,x.shape[0]):x[i], intercept, r_value, p_value, std_err = stats.linregress(np.linspace(0,preResponseMat.shape[1],preResponseMat.shape[1]),preResponseMat[i,:])

                
            return x

        #postCC - activity following color change (0 - 250 ms)
        # define post short window (250 ms post)
        pstShort_idx = np.argmin(np.absolute(self.pow_xval-self.taskstats2d_trange_periEventShort_xval))

        #raw 
        postCCMatS = self.taskstats2d['postCC_timeCourseShort_trials'][:,targOn_idx:pstShort_idx]
        postCCMatL = self.taskstats2d['postCC_timeCourseLong_trials'][:,targOn_idx:pstShort_idx]
        responseModel_dict['postCCS'] = np.nanmean(postCCMatS,axis=1)
        responseModel_dict['postCCL'] = np.nanmean(postCCMatL,axis=1)
        responseModel_dict['postCC'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['postCCS'], responseModel_dict['postCCL'])
        # get postcc build up rate (on Raw Data)
        responseModel_dict['postCC_burS'] = getBuildupRate(postCCMatS)
        responseModel_dict['postCC_burL'] = getBuildupRate(postCCMatL)
        responseModel_dict['postCC_bur'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['postCC_burS'], responseModel_dict['postCC_burL'])


        #clean 
        postCCMatS_clean = responseModel_dict['responseS_trials_clean_ccLocked'][:,targOn_idx:pstShort_idx]
        postCCMatL_clean = responseModel_dict['responseL_trials_clean_ccLocked'][:,targOn_idx:pstShort_idx]
        responseModel_dict['postCCS_clean'] = np.nanmean(postCCMatS_clean,axis=1)
        responseModel_dict['postCCL_clean'] = np.nanmean(postCCMatL_clean,axis=1)
        responseModel_dict['postCC_clean'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['postCCS_clean'], responseModel_dict['postCCL_clean'])
        # get postcc build up rate (on clean Data)
        responseModel_dict['postCC_burS_clean'] = getBuildupRate(postCCMatS_clean)
        responseModel_dict['postCC_burL_clean'] = getBuildupRate(postCCMatL_clean)
        responseModel_dict['postCC_bur_clean'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['postCC_burS_clean'], responseModel_dict['postCC_burL_clean'])


        # define short time windows (which we need here; default 250 ms)
        preShort_idx = np.argmin(np.absolute(self.pow_xval+self.taskstats2d_trange_periEventShort_xval))
        pstShort_idx = np.argmin(np.absolute(self.pow_xval-self.taskstats2d_trange_periEventShort_xval))

        # generate response-locked short and long delay data 


        # raw
        preResponseMatS = self.taskstats2d['postResponse_timeCourseShort_trials'][:,preShort_idx:targOn_idx]
        preResponseMatL =self.taskstats2d['postResponse_timeCourseLong_trials'][:,preShort_idx:targOn_idx]

        responseModel_dict['preResponseS'] = np.nanmean(preResponseMatS,axis=1)
        responseModel_dict['preResponseL'] = np.nanmean(preResponseMatL,axis=1)
        responseModel_dict['preResponse'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['preResponseS'], responseModel_dict['preResponseL'])
        # get pre response build up rate
        responseModel_dict['preResponse_burS'] = getBuildupRate(preResponseMatS)
        responseModel_dict['preResponse_burL'] = getBuildupRate(preResponseMatL)
        responseModel_dict['preResponse_bur'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['preResponse_burS'], responseModel_dict['preResponse_burL'])


        # clean
        # option 1: use  clean target locked data (so we are removing all visual evoked signals that can confound pre-build up rate and preRepsonse_peak measures); but this is a worse model of response related effects and can lead to more residual effects in some cases

        # option 2: use clean response locked data (models the peri-response function better, but potentilally includes more visual-locked evoked signals)

        # strategy: if model fit of target locked data > 0.7 then use it, but if it is a really bad fit, then use response locked data
        if responseModel_dict['mod_fullS_rsq'] > 0.7:
            responseS_ccCleaned,mod_xval_respLocked = toResponseLocked(responseModel_dict['responseS_trials_clean'],responseModel_dict['rtsIdx_targLocked_S'])
            preResponseMatS_clean = responseS_ccCleaned[:,preShort_idx:targOn_idx]
        else:
            preResponseMatS_clean = responseModel_dict['responseS_trials_clean_respLocked'][:,preShort_idx:targOn_idx]


        if responseModel_dict['mod_fullL_rsq'] > 0.7:
            responseL_ccCleaned,mod_xval_respLocked = toResponseLocked(responseModel_dict['responseL_trials_clean'],responseModel_dict['rtsIdx_targLocked_L'])
            preResponseMatL_clean =responseL_ccCleaned[:,preShort_idx:targOn_idx]
        else:
            preResponseMatL_clean =responseModel_dict['responseL_trials_clean_respLocked'][:,preShort_idx:targOn_idx]

        responseModel_dict['preResponseS_clean'] = np.nanmean(preResponseMatS_clean,axis=1)
        responseModel_dict['preResponseL_clean'] = np.nanmean(preResponseMatL_clean,axis=1)
        responseModel_dict['preResponse_clean'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['preResponseS_clean'], responseModel_dict['preResponseL_clean'])
        # get pre response build up rate (on clean)
        responseModel_dict['preResponse_burS_clean'] = getBuildupRate(preResponseMatS_clean)
        responseModel_dict['preResponse_burL_clean'] = getBuildupRate(preResponseMatL_clean)
        responseModel_dict['preResponse_bur_clean'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['preResponse_burS_clean'], responseModel_dict['preResponse_burL_clean'])




        # get response peaks (activity at time of response). Last index is response. (ON RAW DATA)
        responseModel_dict['preResponse_peaksS'] =preResponseMatS[:,-1]
        responseModel_dict['preResponse_peaksL'] = preResponseMatL[:,-1]
        responseModel_dict['preResponse_peaks'] = combineTrialsVec(shortTrials_bool,longTrials_bool,responseModel_dict['preResponse_peaksS'], responseModel_dict['preResponse_peaksL'])

        ########## SAVE DATA IN SELF AND TASKSTATS##########
        self.responseModel_dict = {}
        self.responseModel_dict = responseModel_dict.copy()

        # get key list to include in task stats (ignore trial by trial data))
        key_list = list(responseModel_dict.keys())
        for k in key_list:
            #print(k)
            if 'trials' in k:
                del responseModel_dict[k]
   
        # update task stats with responseModel_dict (but do not include trial by trial data to save space)
        self.taskstats2d.update(responseModel_dict)

    def taskstats_doStats(self,num_iters = 20):
        # This function orchestrates non-parameteric stats relating model-free and model-based behavioral variables to neural activity

        # subfunctions
        def getBehData(apply_circ_shift = False):
            beh_feat_dict = {}

            # collect behavioral data from taskstats
            for f in self.taskstats2d['feat_list_beh']:
                # get beh feature
                x = np.copy(self.taskstats2d[f])

                # z-score (to help interpret parameters)
                x = (x - np.nanmean(x))/np.nanstd(x)

                if apply_circ_shift == True:
                    # randomly generate an amount to circularly shift data
                    shift_idx = np.random.randint(low=0, high=len(x))

                    # circ shift
                    x = np.roll(x,shift_idx)

                    # label this as a null iteration
                    beh_feat_dict['is_null'] = True
                else:
                    beh_feat_dict['is_null'] = False


                # update beh_feat_dict
                beh_feat_dict[f] = x 

            return beh_feat_dict
                    
        def getStatsDict(beh_feat_dict,output_light = False):
            # returns true stats if you feed in true behavioral data, but returns null stats if you feed in circ shifted behavioral data

            # init stats dict
            stats_dict = {}
            stats_dict['isBad_list'] = []

            # pop out is null
            is_null = beh_feat_dict.pop('is_null', None)


            # get reg data
            for k in beh_feat_dict.keys():

                if np.all(beh_feat_dict[k]==0)|np.all(np.isnan(beh_feat_dict[k])):
                    stats_dict[k+'_isBad'] = True
                    stats_dict['isBad_list'].append(True)
                    continue
 
                # else, contiue
                stats_dict[k+'_isBad'] = False
                stats_dict['isBad_list'].append(False)

                regStats_dict = self.taskstats_rtRegress(beh_var = beh_feat_dict[k],beh_var_lbl=k,is_null = is_null)
                stats_dict[k+'_keys_rtRegress'] = list(regStats_dict.keys())
                stats_dict.update(regStats_dict)

            # parse return_light
            key_list = list(stats_dict.keys())
            if output_light == True:
                for k in key_list:
                    if ('SSE' in k) | ('tstat' in k) | ('rval' in k):
                        continue
                    else:
                        del stats_dict[k]

            return stats_dict 
        
        def update_pvals(stats_dict_true,stats_null_df):
            # updates true stats dict with non-parametric p-values based on circular sift of that procedure. generate pval_np fields ("non parametric")

            def get_p_two_tailed(null_vals,true_val):

                # parse true_val
                # note we use median here instead of zero in case there are asymmetries in the null distribution
                if true_val>=np.median(null_vals):
                    # probability of observing a null value greater than true value
                    pval_np = np.count_nonzero(null_vals >= true_val)/len(null_vals)
                else:
                    # prob of observing a null value less than true value
                    pval_np = np.count_nonzero(null_vals <= true_val)/len(null_vals)
                # multiply by 2 to make it a two-tailed p-value (ceiling val of 1)
                pval_np = np.min((1,pval_np * 2))

                # get 95% confidence intervals
                ci_neg,ci_pos = np.percentile(null_vals,[2.5,97.5]) 


                # calculate zstat np (non parametric z stat)
                zstatnp = (true_val - np.mean(null_vals))/np.std(null_vals)
        
                return pval_np,ci_pos,ci_neg,zstatnp

            # loop through beh_feat fields
            for k in self.taskstats2d['feat_list_beh']:
                if stats_dict_true[k+'_isBad'] == True:
                    continue
                
                # do we have more predictive power than expected by chance? (one-tailed pvalue for sum squared error)
                null_sse = stats_null_df['rtRegress_multivar_'+k+'_SSE'].to_numpy()
                true_sse = stats_dict_true['rtRegress_multivar_'+k+'_SSE']

                # calc false pos rate (prob of observing a null sse that is smaller than or equal to true sse))
                pval_np = np.count_nonzero(null_sse <= true_sse)/len(null_sse)

                stats_dict_true['rtRegress_multivar_'+k+'_SSE_pvalnp'] = pval_np
                # save null dist in this case for plotting
                stats_dict_true['rtRegress_multivar_'+k+'_SSE_null'] =  null_sse

                for f in stats_dict_true[k+'_keys_rtRegress']:
                    if 'tstat' in f:
                        if 'multivar' in f:
                            # use largest effect for null dist when evaluating t-stats associated with multivariate regression
                            null_vals= stats_null_df['rtRegress_multivar_'+k+'_largestEffect_tstat'].to_numpy()
                        else:
                            null_vals= stats_null_df[f].to_numpy()

                        # save null
                        stats_dict_true[f.split('tstat')[0]+'null'] = null_vals

                        #get two tailed p-value
                        stats_dict_true[f.split('tstat')[0]+'pvalnp'],stats_dict_true[f.split('tstat')[0]+'cipos'],stats_dict_true[f.split('tstat')[0]+'cineg'],stats_dict_true[f.split('tstat')[0]+'zstatnp'] = get_p_two_tailed(null_vals= null_vals, true_val= stats_dict_true[f])
                    elif 'rval' in f:
                        null_vals = stats_null_df[f]

                        # save null
                        #stats_dict_true[f.split('rval')[0]+'null'] = null_vals

                        #get two tailed p-value
                        stats_dict_true[f.split('rval')[0]+'pvalnp'],stats_dict_true[f.split('rval')[0]+'cipos'],stats_dict_true[f.split('rval')[0]+'cineg'],stats_dict_true[f.split('rval')[0]+'zstatnp'] = get_p_two_tailed(null_vals= null_vals, true_val= stats_dict_true[f])


            return stats_dict_true

        # get true stats
        beh_dict_true = getBehData(apply_circ_shift=False)
        stats_dict_true = getStatsDict(beh_dict_true)

        # run through circular shifts
        stats_dict_null_list = []
        for i in np.arange(num_iters):

            # get null stats using circularly shifted behavioral data
            beh_dict_null = getBehData(apply_circ_shift=True)
            stats_dict_null = getStatsDict(beh_dict_null,output_light=True)

            # update list 
            stats_dict_null_list.append(stats_dict_null)

            print(i,'/',num_iters)

        # get nullStats_df
        stats_null_df = pd.DataFrame(stats_dict_null_list)


        # update stats dict with non-parametric p-values
        stats_dict_true = update_pvals(stats_dict_true,stats_null_df)


        # # if there are any bad features, we need to fill in taskstats with the appropriate fields
        # if np.any(stats_dict_true['isBad_list']):
        #     # find first beh var for which isBad is False
        #     good_feat = self.taskstats2d['feat_list_beh'][stats_dict_true['isBad_list'].index(False)]

        #     # get key list
        #     key_list = list(stats_dict_true[good_feat+'_keys_rtRegress'])+list(stats_dict_true[good_feat+'_keys_corrStats'])

        #     # get bad feat list 
        #     bad_feat_list= list(np.array(self.taskstats2d['feat_list_beh'])[np.array(stats_dict_true['isBad_list'])])

        #     for bad_feat in bad_feat_list:

        #         #loop through key list
        #         for k in key_list:
        #             k_split = k.split(good_feat)
        #             if len(k_split) == 1: 
        #                 # skip keys that are not specific to this feature
        #                 continue
        #             stats_dict_true[k_split[0]+bad_feat+k_split[1]] =  np.nan

        #         stats_dict_true['rtRegress_multivar_'+bad_feat+'_SSE_pvalnp'] = np.nan
        #         stats_dict_true['rtRegress_multivar_'+bad_feat+'_SSE_null'] = np.nan
                

        #         for f in stats_dict_true[good_feat+'_keys_rtRegress']:
        #             if 'tstat' in f:
        #                 stats_dict_true[f.split('tstat')[0]+'pvalnp']=np.nan
        #                 stats_dict_true[f.split('tstat')[0]+'cipos']=np.nan
        #                 stats_dict_true[f.split('tstat')[0]+'cineg']=np.nan
                # for f in stats_dict_true[good_feat+'_keys_corrStats']:
                #     if 'corrVec' in f:
                #         stats_dict_true[f+'_pvalsnp'] = np.nan
                #         stats_dict_true[f+'_cipos'] = np.nan
                #         stats_dict_true[f+'_cineg'] = np.nan


        # update task stats
        self.taskstats2d.update(stats_dict_true)



        #self.stats_null_df = []
        #self.stats_null_df = stats_null_df


    def taskstats_rtRegress(self,beh_var,beh_var_lbl,is_null):
        # This function performs the regression framework to relate neural activity with RT

        # is_null indicates whether we are dealing with shuffled data. 

        #fit various regressions using statsmodels.formula
        
        rtRegress = {}
        regStats_dict  = {}

        # create data_dict (for regression. These are not shuffled)
        data_dict ={}
        feat_to_copy = ['longTrials_bool','S0f','S0c','postCC','postCC_bur','preResponse','preResponse_bur','preResponse_peaks','S0f_clean','S0c_clean','postCC_clean','postCC_bur_clean','preResponse_clean','preResponse_bur_clean']

        for f in feat_to_copy:
            data_dict[f] = np.copy(self.taskstats2d[f])

        # update w behavioral var (done separately to allow for circularly shifted data)
        data_dict[beh_var_lbl] = beh_var

        # OMNIBUS TEST (compare SSE vs. null distribution to assess predicitive power)

        # multivar_neuralOnly (without build up rates)
        #reg_neuOnly = smf.ols(beh_var_lbl+' ~ S0f_clean + S0c_clean + postCC_clean + preResponse_clean', data = data_dict).fit()

        # multivar_neuralOnly (including build up rates)
        reg_neuOnly = smf.ols(beh_var_lbl+' ~ S0f_clean + S0c_clean + postCC_clean + postCC_bur_clean  + preResponse_clean + preResponse_bur_clean', data = data_dict).fit()

        # save sum of squared error (for null stats purposes. The larger this value, the worse the fit). Need to obtain a one-tailed measure asking whether this neural signal has any predictive power of RT?
        regStats_dict['rtRegress_multivar_'+beh_var_lbl+'_SSE'] =  reg_neuOnly.ssr


        # run spearman corr to assess specific effects (separately for long and short delay, and separately for clean and not clean)
        def run_corr(data_dict,regStats_dict,neu_var_lbl,apply_clean = True):

            ### all trials 
            lbl = 'rtCorr_'+beh_var_lbl+'_'+neu_var_lbl
            lbl_clean = lbl + '_clean'

            # raw
            regStats_dict[lbl+'_rval'], regStats_dict[lbl+'_pval'] = stats.spearmanr(data_dict[neu_var_lbl],data_dict[beh_var_lbl],nan_policy='omit')

            # clean
            if apply_clean == True:
                regStats_dict[lbl_clean+'_rval'], regStats_dict[lbl_clean+'_pval']    = stats.spearmanr(data_dict[neu_var_lbl+'_clean'],data_dict[beh_var_lbl],nan_policy='omit')

            ### short delay trials only
            lbl = 'rtCorrS_'+beh_var_lbl+'_'+neu_var_lbl
            lbl_clean = lbl + '_clean'

            # raw
            regStats_dict[lbl+'_rval'], regStats_dict[lbl+'_pval'] = stats.spearmanr(data_dict[neu_var_lbl][self.taskstats2d['shortTrials_bool']],data_dict[beh_var_lbl][self.taskstats2d['shortTrials_bool']],nan_policy='omit')

            # clean
            if apply_clean == True:
                regStats_dict[lbl_clean+'_rval'], regStats_dict[lbl_clean+'_pval'] = stats.spearmanr(data_dict[neu_var_lbl+'_clean'][self.taskstats2d['shortTrials_bool']],data_dict[beh_var_lbl][self.taskstats2d['shortTrials_bool']],nan_policy='omit')


            ### long delay trials only
            lbl = 'rtCorrL_'+beh_var_lbl+'_'+neu_var_lbl
            lbl_clean = lbl + '_clean'

            # raw
            regStats_dict[lbl+'_rval'], regStats_dict[lbl+'_pval'] = stats.spearmanr(data_dict[neu_var_lbl][self.taskstats2d['longTrials_bool']],data_dict[beh_var_lbl][self.taskstats2d['longTrials_bool']],nan_policy='omit')

            # clean
            if apply_clean == True:
                regStats_dict[lbl_clean+'_rval'], regStats_dict[lbl_clean+'_pval'] = stats.spearmanr(data_dict[neu_var_lbl+'_clean'][self.taskstats2d['longTrials_bool']],data_dict[beh_var_lbl][self.taskstats2d['longTrials_bool']],nan_policy='omit')

            return regStats_dict
        # S0f (pre target baseline)
        regStats_dict = run_corr(data_dict,regStats_dict,'S0f')

        # S0c (pre CC)
        regStats_dict = run_corr(data_dict,regStats_dict,'S0c')

        # postCC (post stim)
        regStats_dict = run_corr(data_dict,regStats_dict,'postCC')

        # postCC buildup rate
        regStats_dict = run_corr(data_dict,regStats_dict,'postCC_bur')

        # preResponse 
        regStats_dict = run_corr(data_dict,regStats_dict,'preResponse')

        # preResponse buildup rate
        regStats_dict = run_corr(data_dict,regStats_dict,'preResponse_bur')

        # preResponse_peaks
        regStats_dict = run_corr(data_dict,regStats_dict,'preResponse_peaks',apply_clean = False)



        #### Calculate delay-related differences in neural features
        delay_feat_list = ['S0c','postCC','postCC_bur','preResponse','preResponse_bur']

        # get long and short bool idx

        for d in delay_feat_list:
            lbl = 'delayDiff_'+d
            lbl_clean = lbl+'_clean'
            if is_null == False:
            
                # compute true stat (raw)
                regStats_dict[lbl+'_tstat'],regStats_dict[lbl+'_pval'] = stats.ttest_ind(data_dict[d][self.taskstats2d['longTrials_bool']],data_dict[d][self.taskstats2d['shortTrials_bool']])

                # compute true stat (clean)
                regStats_dict[lbl_clean+'_tstat'],regStats_dict[lbl+'_pval'] = stats.ttest_ind(data_dict[d+'_clean'][self.taskstats2d['longTrials_bool']],data_dict[d+'_clean'][self.taskstats2d['shortTrials_bool']])


            else:
                # compute null stat (by randomly assigning long/short delay labels matched to the number of long and short delay trials)

                # shuffle in place using a copy of the delay labels from events
                shuf_delay_lbls =(np.copy(self.taskstats2d['pow_ev_filt']['delay'].to_numpy()))
                np.random.shuffle(shuf_delay_lbls)

                shuf_longTrials_bool = (shuf_delay_lbls==1500)
                shuf_shortTrials_bool = shuf_delay_lbls==500

                # compute shuffle stat (raw)
                regStats_dict[lbl+'_tstat'],regStats_dict[lbl+'_pval'] = stats.ttest_ind(data_dict[d][shuf_longTrials_bool],data_dict[d][shuf_shortTrials_bool],nan_policy='omit')

                # compute shuffle stat (clean)
                regStats_dict[lbl_clean+'_tstat'],regStats_dict[lbl+'_pval'] = stats.ttest_ind(data_dict[d+'_clean'][shuf_longTrials_bool],data_dict[d+'_clean'][shuf_shortTrials_bool],nan_policy='omit')

                

        return regStats_dict


    def taskstats_correlateFeatures(self):
        #This function performs correlations between various neural response features 

        # sub-routine
        def runcorr(self,lbl1,lbl2):
            x = self.taskstats2d[lbl1]
            y = self.taskstats2d[lbl2]
            drop_idx = np.isnan(x)|np.isnan(y)


            r,p = stats.spearmanr(x[~drop_idx], y[~drop_idx])
            self.taskstats2d['corr_'+lbl1+'_'+lbl2+'_r'] = r
            self.taskstats2d['corr_'+lbl1+'_'+lbl2+'_p'] = p


        # S0F vs. postFix
        runcorr(self,'S0f','S0c')
        runcorr(self,'S0c','postCC')
        runcorr(self,'postCC','preResponse')
        runcorr(self,'preResponse','preResponse_peaks')
        runcorr(self,'preResponse','preResponse_bur')
        runcorr(self,'preResponse_bur','preResponse_peaks')

    def taskstats_selectivityAnova(self):
        # This function performs various analyses on trials (including those not included in evQuery). ANOVAs to assess whether postFix activity varies by target location; and post response activity varies with error and feedback. Here, we use all trials, not just queried trials

        # get post fix data (including all good quality trials)
        self.getPow_2d(pow_evType='FIX_START',
                       pow_frange_lbl = self.taskstats2d_pow_frange_lbl,
                       pow_method = self.taskstats2d_pow_method,
                       pow_evQuery = 'badTrial==0',
                       do_zscore = self.taskstats2d_do_zscore,
                       apply_gauss_smoothing = self.taskstats2d_apply_gauss_smoothing,
                       gauss_sd_scaling = self.taskstats2d_gauss_sd_scaling,
                       apply_time_bins=self.taskstats2d_apply_time_bins,
                       time_bin_size_ms=self.taskstats2d_time_bin_size_ms)

        # time intervals of interest 0 (targ on) to 500 ms (periEvent)
        zero_idx = np.argmin(np.absolute(self.pow_xval-0))
        pst_idx = np.argmin(np.absolute(self.pow_xval-self.taskstats2d_trange_periEvent_xval))

        # 250 ms post target on (end of pre-event interval)
        pstShort_idx = np.argmin(np.absolute(self.pow_xval-self.taskstats2d_trange_periEventShort_xval))


        # 500 ms prior to target on (end of pre-event interval)
        preEnd_idx = np.argmin(np.absolute(self.pow_xval+self.taskstats2d_trange_periEvent_xval))


        # identify trials where participants made a premature response (do not count really early premature responses to avoid confounds w post-target activity)
        premature_bool = self.pow_ev_filt.eval('RT<0&RT_targ>500')

        # prediction times (time of premature response from target onset)
        pts = self.pow_ev_filt['RT_targ'].to_numpy()[premature_bool]


        # get power in various time interval
        # pre-target onset power 
        preTarg_pow = np.nanmean(self.powMat[:,preEnd_idx:zero_idx],axis=1)
        # post target onset power
        postTarg_pow = np.nanmean(self.powMat[:,zero_idx:pstShort_idx],axis=1)

        # calculate spearman corr relating pre targ activity and prediction times
        rval,pval = stats.spearmanr(preTarg_pow[premature_bool],pts)

        # calculate spearman corr
        self.taskstats2d['ptCorr_S0f_rval'] = rval
        self.taskstats2d['ptCorr_S0f_pval'] = pval

        # calculate spearman corr relating post targ activity and prediction times
        rval,pval = stats.spearmanr(postTarg_pow[premature_bool],pts)

        # calculate spearman corr
        self.taskstats2d['ptCorr_postTarg_rval'] = rval
        self.taskstats2d['ptCorr_postTarg_pval'] = pval

        # # calculate power and build up rate prior to antcipatory response (prediction)
        # prePT_pow = np.zeros(np.shape(pts))
        # prePT_bur = np.zeros(np.shape(pts))

        # # loop through all premature bool trials
        # p_count=-1
        # for pt_idx in np.where(premature_bool)[0]:
        #     # 
        #     p_count+=1

        #     # this prediction time in samples
        #     thisPT_samp = self.ms_to_samples(pts[p_count])

        #     # start and stop of power segment relative to this prediction time
        #     #(-250 ms to 0)
        #     this_start = int(thisPT_samp-self.taskstats2d_trange_periEventShort_xval) # 250 ms prior
        #     this_stop = int(thisPT_samp)
            
        #     # calc segment of power prior to prediction time
        #     thisPow = self.powMat[pt_idx,this_start:this_stop]

        #     f = plt.figure()
        #     plt.plot(thisPow)


        #     # get mean pow
        #     prePT_pow[p_count] = np.nanmean(thisPow)

        #     # get build up rate
        #     prePT_bur[p_count], intercept, r_value, p_value, std_err = stats.linregress(np.linspace(0,len(thisPow),len(thisPow)),thisPow)


        # # calculate spearman corr relating pre prediction time activity and prediction times
        # rval,pval = stats.spearmanr(prePT_pow,pts)

        # # calculate spearman corr
        # self.taskstats2d['ptCorr_preResp_rval'] = rval
        # self.taskstats2d['ptCorr_preResp_pval'] = pval


        # # calculate spearman corr relating pre prediction time activity and prediction times
        # rval,pval = stats.spearmanr(prePT_bur,pts)

        # # calculate spearman corr
        # self.taskstats2d['ptCorr_preResp_bur_rval'] = rval
        # self.taskstats2d['ptCorr_preResp_bur_pval'] = pval


        ### target location ANOVA


        # get power data
        tl_pow = np.nanmean(self.powMat[:,zero_idx:pst_idx],axis=1)
        tl_lbls = self.pow_ev_filt['targetLoc_lbl'].to_numpy()  

        tl_df = pd.DataFrame({'zPow':pd.Series(tl_pow),'targLoc':pd.Series(tl_lbls)}) 

        # set up one-way anova OLS model 
        #https://www.pythonfordatascience.org/anova-python/#anova_statsmodels
        olsmod = smf.ols('zPow ~ C(targLoc)',data = tl_df).fit()

        # fit one-way anova (using type 2 method to measure sum of squares)
        aov_table = sm.stats.anova_lm(olsmod, typ=2)

        self.taskstats2d['tlAnova_fstat'] = aov_table.loc['C(targLoc)']['F']
        self.taskstats2d['tlAnova_pval'] = aov_table.loc['C(targLoc)']['PR(>F)'] 
        self.taskstats2d['tlAnova_zstat'] = stats.norm.ppf(1-self.taskstats2d['tlAnova_pval'])
        # find target location with maximum mean zpow 
        tl_meanPow = tl_df.groupby('targLoc').mean()
        maxResponse = tl_meanPow.max().to_numpy()
        maxResponse_loc = tl_meanPow.query('zPow==@maxResponse').index.to_numpy()[0]  

        self.taskstats2d['tlAnova_maxResponse'] = maxResponse
        self.taskstats2d['tlAnova_maxResponse_loc'] = maxResponse_loc

        # get post response data (including all good trials)
        self.getPow_2d(pow_evType='RESPONSE',
                       pow_frange_lbl = self.taskstats2d_pow_frange_lbl,
                       pow_method = self.taskstats2d_pow_method,
                       pow_evQuery = 'badTrial==0',
                       do_zscore = self.taskstats2d_do_zscore,
                       apply_gauss_smoothing = self.taskstats2d_apply_gauss_smoothing,
                       gauss_sd_scaling = self.taskstats2d_gauss_sd_scaling,
                       apply_time_bins=self.taskstats2d_apply_time_bins,
                       time_bin_size_ms=self.taskstats2d_time_bin_size_ms)
        
        # time intervals of interest 0 (targ on) to 1000 ms (postLong)
        zero_idx = np.argmin(np.absolute(self.pow_xval-0))
        pstLong_idx = np.argmin(np.absolute(self.pow_xval-self.taskstats2d_trange_periEventLong_xval))

        # get fb power data 
        fb_pow =  np.nanmean(self.powMat[:,zero_idx:pstLong_idx],axis=1) 

        # t-test error vs. no error
        error_bool = self.pow_ev_filt['error']
        self.taskstats2d['errSel_tstat'],self.taskstats2d['errSel_pval'] = stats.ttest_ind(fb_pow[error_bool],fb_pow[error_bool==False])


        # t-test post fb vs. no fb
        reward_bool = self.pow_ev_filt.eval('RT<300')
        self.taskstats2d['rewSel_tstat'],self.taskstats2d['rewSel_pval'] = stats.ttest_ind(fb_pow[reward_bool],fb_pow[reward_bool==False])




    def plotTaskStats2d_sse(self,ax=None,beh_var_lbl = 'zrrt',fsize_lbl=12,fsize_tick=12):
        # set ax
        if ax == None:
            f = plt.figure(figsize=(10,5))
            ax = plt.subplot(111)

        # plot null dist
        plt.hist(self.taskstats2d['rtRegress_multivar_'+beh_var_lbl+'_SSE_null'],color = '0.5',bins=100)

        # plot true sse
        plt.vlines(self.taskstats2d['rtRegress_multivar_'+beh_var_lbl+'_SSE'],ax.get_ylim()[0],ax.get_ylim()[1],color ='red',linewidth=3)

        # title
        ax.set_title(label=('Null distribution (circular shift) '+beh_var_lbl+'  p-val: '+str(np.round(self.taskstats2d['rtRegress_multivar_'+beh_var_lbl+'_SSE_pvalnp'],3))),fontsize=fsize_lbl)
        ax.set_ylabel('Count',fontsize=fsize_lbl)
        ax.set_xlabel('Sum of squared error in residuals',fontsize=fsize_lbl)



    def plotTaskStats2d_timeCourse(self,ax = None, lbl=None,evType = 'FIX_START', yL = None, xL_ms = None,add_vline=True,fsize_lbl=16,fsize_tick=16,alpha = 0.6,color_short = None,color_long = None,add_legend = False,add_title=False,use_clean_data = False, plot_short=True,plot_long=True):
        # This function plots the power time course for the signal that was used to compute taskstats. Separately plots long and short delay trials

        # set ax
        if ax == None:
            f = plt.figure()
            ax = plt.subplot(111)

        # parse inputs
        if lbl == None:
            lbl_short = 'short delay'
            lbl_long = 'long delay'
        else:
            lbl_short = lbl+' short delay'
            lbl_long = lbl+' long delay'

        if color_short == None:
            color_short  = 'C0'
        if color_long == None:
            color_long = 'C1'

        # get pow data
        if evType == 'FIX_START':
            fieldLbl = 'postFix'
            xLbl = 'target onset'
            fieldLbl2 = ''
        elif evType == 'CC':
            fieldLbl = 'postCC'
            xLbl = 'color change'
            fieldLbl2 = '_ccLocked'
        elif evType == 'RESPONSE':
            fieldLbl = 'postResponse'
            xLbl = 'response'
            fieldLbl2 = '_respLocked'



        # get x values
        pow_xval = self.taskstats2d['timeCourse_xval'+'_'+fieldLbl]

        # parse xlim
        if xL_ms == None:
            if self.taskstats2d_apply_time_bins == False:
                xL_ms = (self.samples_to_ms(pow_xval[0]),self.samples_to_ms(pow_xval[-1]))
            else:
                xL_ms = (pow_xval[0],pow_xval[-1])

        if use_clean_data == False:
            powMat_short_mean = self.taskstats2d[fieldLbl+'_timeCourseShort_mean']
            powMat_long_mean = self.taskstats2d[fieldLbl+'_timeCourseLong_mean']
        else:
            powMat_short_mean = self.taskstats2d['responseS_clean'+fieldLbl2]
            powMat_long_mean = self.taskstats2d['responseL_clean'+fieldLbl2]

        powMat_short_sem = self.taskstats2d[fieldLbl+'_timeCourseShort_sem']
        powMat_long_sem = self.taskstats2d[fieldLbl+'_timeCourseLong_sem']


        # plot short delay
        if plot_short == True:
            ax.plot(pow_xval,powMat_short_mean,label=lbl_short,alpha=alpha,color = color_short)
            ax.fill_between(pow_xval,powMat_short_mean+powMat_short_sem,powMat_short_mean-powMat_short_sem,alpha=alpha-.2,color = color_short)

        # plot long delay
        if plot_long == True:
            ax.plot(pow_xval,powMat_long_mean,label=lbl_long,alpha=alpha,color = color_long)
            ax.fill_between(pow_xval,powMat_long_mean+powMat_long_sem,powMat_long_mean-powMat_long_sem,alpha=alpha-.2,color = color_long)


        # if x val are in samples, then covert tick labels
        if self.taskstats2d_apply_time_bins == False:
            ax.set_xlim((self.ms_to_samples(xL_ms[0]),self.ms_to_samples(xL_ms[1])))
            xt = np.array([self.ms_to_samples(xL_ms[0]),0,0.5*self.samplerate,self.ms_to_samples(xL_ms[1])])
            ax.set_xticks(xt)
            ax.set_xticklabels(np.round(1000*(xt/self.samplerate)).astype('int'),fontsize=fsize_tick)
        else:
            ax.set_xlim((xL_ms[0],xL_ms[1]))
            xt = np.array([xL_ms[0],0,pow_xval[np.argmin(np.abs(pow_xval-500))],xL_ms[-1]])
            ax.set_xticks(xt)
            ax.set_xticklabels(xt,fontsize=fsize_tick)

        ax.set_xlabel('Time from '+xLbl+' (ms)',fontsize=fsize_lbl)
        if self.taskstats2d_do_zscore == True:
            ax.set_ylabel('z-score '+self.taskstats2d_pow_frange_lbl,fontsize=fsize_lbl)
        else:
            ax.set_ylabel('Power (a.u.)',fontsize=fsize_lbl)

        # set ylim
        if yL != None:
            ax.set_ylim(yL)
        else:
            yL = ax.get_ylim()


        # set yticklabels
        plt.yticks(np.linspace(yL[0], yL[1],5), np.round(np.linspace(yL[0], yL[1],5),2), fontsize=fsize_tick)

        # v line
        if add_vline==True:
            # get vL_ticks
            if self.taskstats2d_apply_time_bins == False:
                if evType=='FIX_START':
                    if (plot_short == True)&(plot_long == True):
                        vL_ticks = [0,int(0.5*self.samplerate),int(1.5*self.samplerate)]
                    elif plot_short == True:
                        vL_ticks = [0,int(0.5*self.samplerate)]
                    elif plot_long == True:
                        vL_ticks = [0,int(1.5*self.samplerate)]

                else:
                    vL_ticks = [0]

            else:
                if evType=='FIX_START':
                    vL_ticks= [pow_xval[np.argmin(np.abs(pow_xval-0))],pow_xval[np.argmin(np.abs(pow_xval-500))],pow_xval[np.argmin(np.abs(pow_xval-1500))]]
                else:
                    vL_ticks= [pow_xval[np.argmin(np.abs(pow_xval-0))]]

            for v in vL_ticks:
                if (v > xL_ms[0]) & (v < xL_ms[1]):
                    ax.vlines(x=v,
                      ymin=ax.get_ylim()[0],
                      ymax=ax.get_ylim()[1],
                      linestyles='--',alpha=0.5)
        # legend
        if add_legend == True:
            ax.legend(fontsize=fsize_tick)

        #title
        if add_title == True:
            if 'White Matter' in self.taskstats2d['anat_native']:
                ax.set_title(self.anat_dict['uElbl']+'-'+self.taskstats2d['anat_wm'],fontsize=fsize_tick)
            else:
                ax.set_title(self.anat_dict['uElbl']+'-'+self.anat_dict['anat_native'],fontsize=fsize_tick)
            #ax.set_title(self.anat_dict['uElbl']+'-'+self.anat_dict['anat_native'],fontsize=fsize_tick)
    def plotTaskStats2d_timeCourseModel_fits(self,ax=None, delay_str = 'S',use_respLocked=False,add_legend=True,add_vline=True,plot_rts=False,fsize_tick = 14,fsize_lbl = 14):
        #This function plots model
        if ax == None:
            f = plt.figure()
            ax = plt.subplot(111)

        if use_respLocked==True:
            r_str = '_respLocked'
        else:
            r_str = ''

        # plot actual response function
        ax.plot(self.taskstats2d['mod_xval'+r_str],self.taskstats2d['response'+delay_str+r_str+'_z'],label='observed response')
        # plot predicted response
        ax.plot(self.taskstats2d['mod_xval'+r_str],self.taskstats2d['mod_full'+delay_str+r_str],label='predicted response',linestyle = '--')

        # get x values
        pow_xval = self.taskstats2d['mod_xval'+r_str]

        # parse xlim
        if self.taskstats2d_apply_time_bins == False:
            xL_ms = (self.samples_to_ms(pow_xval[0]),self.samples_to_ms(pow_xval[-1]))
        else:
            xL_ms = (pow_xval[0],pow_xval[-1])

        # if x val are in samples, then covert tick labels
        if self.taskstats2d_apply_time_bins == False:
            ax.set_xlim((self.ms_to_samples(xL_ms[0]),self.ms_to_samples(xL_ms[1])))
            xt = np.array([self.ms_to_samples(xL_ms[0]),0,0.5*self.samplerate,1.5*self.samplerate,self.ms_to_samples(xL_ms[1])])
            ax.set_xticks(xt)
            ax.set_xticklabels(np.round(1000*(xt/self.samplerate)).astype('int'),fontsize=fsize_tick)
        else:
            ax.set_xlim((xL_ms[0],xL_ms[1]))
            xt = np.array([xL_ms[0],0,pow_xval[np.argmin(np.abs(pow_xval-500))],xL_ms[-1]])
            ax.set_xticks(xt)
            ax.set_xticklabels(xt,fontsize=fsize_tick)


        if use_respLocked==True:
            ax.set_xlabel('Time from response (ms)',fontsize=fsize_lbl)
        else:
            ax.set_xlabel('Time from target onset (ms)',fontsize=fsize_lbl)
        ax.set_ylabel('z-score '+self.taskstats2d_pow_frange_lbl,fontsize=fsize_lbl)


        if delay_str == 'S':
            lbl = 'Short delay response'+r_str
            # option to plot RTs
            if (plot_rts == True) & (use_respLocked==False):
                ax2 = ax.twinx()
                ax2.hist(self.taskstats2d['rts_targLocked_S'],bins=10,alpha=.3,color='0.5')
                ax2.axis('off')

        elif delay_str == 'L':
            lbl = 'Long delay response'+r_str
            # option to plot RTs
            if (plot_rts == True)& (use_respLocked==False):
                ax2 = ax.twinx()
                ax2.hist(self.taskstats2d['rts_targLocked_L'],bins=10,alpha=.3,color='0.5')
                ax2.axis('off')

        ax.set_title(lbl+' r-sq = {}'.format(np.round(self.taskstats2d['mod_full'+delay_str+r_str+'_rsq'],4)))    
        if add_legend == True:
            ax.legend(fontsize=fsize_tick)

        # v line
        if add_vline==True:
            # get vL_ticks
            if self.taskstats2d_apply_time_bins == False:
                if use_respLocked==False:
                    vL_ticks = [0,int(0.5*self.samplerate),int(1.5*self.samplerate)]
                else:
                    vL_ticks = [0]
                
            else:
                if use_respLocked==False:
                    vL_ticks= [pow_xval[np.argmin(np.abs(pow_xval-0))],pow_xval[np.argmin(np.abs(pow_xval-500))],pow_xval[np.argmin(np.abs(pow_xval-1500))]]
                else:
                    vL_ticks= [pow_xval[np.argmin(np.abs(pow_xval-0))]]

            for v in vL_ticks:
                if (v > xL_ms[0]) & (v < xL_ms[1]):
                    ax.vlines(x=v,
                      ymin=ax.get_ylim()[0],
                      ymax=ax.get_ylim()[1],
                      linestyles='--',alpha=0.5)

    def plotTaskStats2d_timeCourseModel_components(self,ax=None, delay_str = 'S',use_respLocked = False, add_legend=True,add_vline=True,plot_rts=True,fsize_tick = 14,fsize_lbl = 14):
        #This function plots model
        if ax == None:
            f = plt.figure()
            ax = plt.subplot(111)

        if use_respLocked==True:
            r_str = '_respLocked'
        else:
            r_str = ''


        if delay_str == 'S':
            lbl = 'Short delay response'+r_str

            if use_respLocked==True:
                ax.plot(self.taskstats2d['mod_xval'+r_str],self.taskstats2d['mod_preResponse'+delay_str+r_str],label='pre response',color = '0.5')
                ax.plot(self.taskstats2d['mod_xval'+r_str],self.taskstats2d['mod_postResponse'+delay_str+r_str],label='post response',color = '0.5')
            else:
                ax.plot(self.taskstats2d['mod_xval'],self.taskstats2d['mod_responseTrend'],color = '0.5',label='response trend')
                ax.plot(self.taskstats2d['mod_xval'],self.taskstats2d['mod_postTargOn'],color = '0.5',label='targetOn')
                ax.plot(self.taskstats2d['mod_xval'],self.taskstats2d['mod_postCCS'],color = '0.5',label='post CC')
                ax.plot(self.taskstats2d['mod_xval'],self.taskstats2d['mod_periResponseS'],color = '0.5',label='periResponseS')
                # option to plot RTs
                if plot_rts == True:
                    ax2 = ax.twinx()
                    ax2.hist(self.taskstats2d['rts_targLocked_S'],bins=10,alpha=.3,color='0.5')
                    ax2.hist(self.taskstats2d['rts_targLocked_L'],bins=10,alpha=.3,color='0.5')
                    ax2.axis('off')

        elif delay_str == 'L':
            lbl = 'Long delay response'+r_str
            if use_respLocked==True:
                ax.plot(self.taskstats2d['mod_xval'+r_str],self.taskstats2d['mod_preResponse'+delay_str+r_str],color = '0.5',label='pre response')
                ax.plot(self.taskstats2d['mod_xval'+r_str],self.taskstats2d['mod_postResponse'+delay_str+r_str],color = '0.5',label='post response')
            else:
                ax.plot(self.taskstats2d['mod_xval'],self.taskstats2d['mod_responseTrend'],color = '0.5',label='response trend')
                ax.plot(self.taskstats2d['mod_xval'],self.taskstats2d['mod_postTargOn'],color = '0.5',label='targetOn')
                ax.plot(self.taskstats2d['mod_xval'],self.taskstats2d['mod_postNoCCS'],color = '0.5',label='no CC S')
                ax.plot(self.taskstats2d['mod_xval'],self.taskstats2d['mod_postCCL'],color = '0.5',label='post CC')            
                ax.plot(self.taskstats2d['mod_xval'],self.taskstats2d['mod_periResponseL'],color = '0.5',label='peri response')            
                # option to plot RTs
                if plot_rts == True:
                    ax2 = ax.twinx()
                    ax2.hist(self.taskstats2d['rts_targLocked_S'],bins=10,alpha=.3,color='0.5')
                    ax2.hist(self.taskstats2d['rts_targLocked_L'],bins=10,alpha=.3,color='0.5')
                    ax2.axis('off')
        ax.set_title(lbl+' components')

        # get x values
        pow_xval = self.taskstats2d['mod_xval'+r_str]

        # parse xlim
        if self.taskstats2d_apply_time_bins == False:
            xL_ms = (self.samples_to_ms(pow_xval[0]),self.samples_to_ms(pow_xval[-1]))
        else:
            xL_ms = (pow_xval[0],pow_xval[-1])

        # if x val are in samples, then covert tick labels
        if self.taskstats2d_apply_time_bins == False:
            ax.set_xlim((self.ms_to_samples(xL_ms[0]),self.ms_to_samples(xL_ms[1])))
            xt = np.array([self.ms_to_samples(xL_ms[0]),0,0.5*self.samplerate,1.5*self.samplerate,self.ms_to_samples(xL_ms[1])])
            ax.set_xticks(xt)
            ax.set_xticklabels(np.round(1000*(xt/self.samplerate)).astype('int'),fontsize=fsize_tick)
        else:
            ax.set_xlim((xL_ms[0],xL_ms[1]))
            xt = np.array([xL_ms[0],0,pow_xval[np.argmin(np.abs(pow_xval-500))],xL_ms[-1]])
            ax.set_xticks(xt)
            ax.set_xticklabels(xt,fontsize=fsize_tick)

        if use_respLocked==True:
            ax.set_xlabel('Time from response (ms)',fontsize=fsize_lbl)
        else:
            ax.set_xlabel('Time from target onset (ms)',fontsize=fsize_lbl)
        ax.set_ylabel('z-score predicted',fontsize=fsize_lbl)



        if add_legend == True:
            ax.legend(fontsize=fsize_tick)

        # v line
        if add_vline==True:
            # get vL_ticks
            if self.taskstats2d_apply_time_bins == False:

                if use_respLocked==False:
                    vL_ticks = [0,int(0.5*self.samplerate),int(1.5*self.samplerate)]
                else:
                    vL_ticks = [0]
                
            else:

                if use_respLocked==False:
                    vL_ticks= [pow_xval[np.argmin(np.abs(pow_xval-0))],pow_xval[np.argmin(np.abs(pow_xval-500))],pow_xval[np.argmin(np.abs(pow_xval-1500))]]
                else:
                    vL_ticks= [pow_xval[np.argmin(np.abs(pow_xval-0))]]
            for v in vL_ticks:
                if (v > xL_ms[0]) & (v < xL_ms[1]):
                    ax.vlines(x=v,
                      ymin=ax.get_ylim()[0],
                      ymax=ax.get_ylim()[1],
                      linestyles='--',alpha=0.5)

    def plotTaskStats2d_fingerprint(self,ax = None,field_list=None,width = 0.4,fsize_tick = 12,yL = None):
        # This function plots bar graphs summarizing the task-related "fingerprint" of the signal

        # set ax
        if ax == None:
            f = plt.figure()
            ax = plt.subplot(111)

        # list of fields to plot
        if field_list == None:
            field_list = ['postFix_tstat','postCC_tstat','postResponse_tstat','modParams_responseTrend_rval','modParams_postTargOn_amp','modParams_preResponseS_respLocked_amp','modParams_postResponseS_respLocked_amp','modParams_postNoCCS_amp','modParams_preResponseL_respLocked_amp','modParams_postResponseL_respLocked_amp']
            #,'tlAnova_zstat','errSel_tstat','rewSel_tstat'
            #,'vmIndex_tstat','mvIndex_tstat', (depreciated)
            
        count = -1
        lbl_list = []
        for f in field_list:
            if 'modParams' in f:
                lbl_list.append(f.split('_')[1])
            else:
                lbl_list.append(f.split('_')[0])
            count+=1
            plt.bar(count,self.taskstats2d[f],color='0.5',edgecolor='k',width = width)

        # set ticks
        ax.set_xticks(np.arange(0,len(field_list)))
        ax.set_xticklabels(lbl_list,fontsize=fsize_tick,rotation=90)
        ax.set_ylabel('t-statistic',fontsize=fsize_tick)
        ax.set_xlim([-.5,len(field_list)-.5])
        if yL != None:
            ax.set_ylim(yL)

    def plotTaskStats2d_rtCorrResults(self,ax = None,beh_feat = 'zrrt',field_list=['S0f','S0c','postCC','postCC_bur','preResponse_bur','preResponse'],width = 0.6,fsize_tick = 16,fsize_lbl = 16,yL = None,use_clean = False,delay_str = '',plot_pvals = True):
        # This function plots bar graphs summarizing the rt-related "fingerprint" of the signal

        # set ax
        if ax == None:
            f = plt.figure()
            ax = plt.subplot(111)

        # list of fields to plot
        field_list_lbls = []

        # parse clean str
        if use_clean==True:
            clean_str = '_clean'
            clean_lbl = 'residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'raw z HFA'

        if delay_str == '':
            delay_lbl = 'all trials'
        elif delay_str == 'S':
            delay_lbl = 'short delay trials'
        elif delay_str == 'L':
            delay_lbl = 'long delay trials'



        count = -1
        xtick_lbl = []
        for f in field_list:
            count+=1

            xtick_lbl.append(f)

            #plt.bar(count,self.taskstats2d['rtCorr'+delay_str+'_'+beh_feat+'_'+f+clean_str+stat_lbl],color='0.5',edgecolor='k',width = width)

            # true stat
            x = self.taskstats2d['rtCorr'+delay_str+'_'+beh_feat+'_'+f+clean_str+'_rval']

            pvalnp = self.taskstats2d['rtCorr'+delay_str+'_'+beh_feat+'_'+f+clean_str+'_pvalnp']

            # ci pos
            cipos = self.taskstats2d['rtCorr'+delay_str+'_'+beh_feat+'_'+f+clean_str+'_cipos']
            # ci neg
            cineg = self.taskstats2d['rtCorr'+delay_str+'_'+beh_feat+'_'+f+clean_str+'_cineg']


            # plot error bar (manually)
            ax.plot((count,count),(cineg,cipos),color = '0.5')
            ax.plot((count-.25,count+.25),(cineg,cineg),color = '0.5')
            ax.plot((count-.25,count+.25),(cipos,cipos),color = '0.5')

            # plot true tstat
            ax.plot(count,x,'*k',markersize=10)

            # write pval?
            if plot_pvals == True:
                ax.text(count+.1,x,s='p = '+str(np.round(pvalnp,3)),fontsize=fsize_tick-2)

        ax.set_xlim((-.5,count+.5))
        ax.set_xticks(np.arange(0,count+1))
        ax.set_xticklabels(xtick_lbl,fontsize=fsize_tick,rotation=45)
        ax.set_ylabel('Spearman $r$',fontsize=fsize_lbl)
        ax.set_title(beh_feat+' '+delay_lbl+' use_clean = '+str(use_clean),fontsize=fsize_lbl)

        if yL != None:
            ax.set_ylim(yL)

        plt.tight_layout()

    def plotTaskStats2d_delayDiffResults(self,ax = None,beh_feat = 'zrrt',field_list=['S0c','postCC','postCC_bur','preResponse','preResponse_bur'],width = 0.6,fsize_tick = 16,fsize_lbl = 16,yL = None,use_clean = False,plot_pvals=True):
        # This function plots bar graphs summarizing the rt-related "fingerprint" of the signal

        # set ax
        if ax == None:
            f = plt.figure()
            ax = plt.subplot(111)

        # list of fields to plot
        field_list_lbls = []

        # parse clean str
        if use_clean==True:
            clean_str = '_clean'
            clean_lbl = 'residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'raw z HFA'

        count = -1
        for f in field_list:
            count+=1

            #plt.bar(count,self.taskstats2d['rtCorr'+delay_str+'_'+beh_feat+'_'+f+clean_str+stat_lbl],color='0.5',edgecolor='k',width = width)

            # true stat
            x = self.taskstats2d['delayDiff'+'_'+f+clean_str+'_tstat']

            pvalnp = self.taskstats2d['delayDiff'+'_'+f+clean_str+'_pvalnp']

            # ci pos
            cipos = self.taskstats2d['delayDiff'+'_'+f+clean_str+'_cipos']
            # ci neg
            cineg = self.taskstats2d['delayDiff'+'_'+f+clean_str+'_cineg']


            # plot error bar (manually)
            ax.plot((count,count),(cineg,cipos),color = '0.5')
            ax.plot((count-.25,count+.25),(cineg,cineg),color = '0.5')
            ax.plot((count-.25,count+.25),(cipos,cipos),color = '0.5')

            # plot true tstat
            ax.plot(count,x,'*k',markersize=10)

            # write pval?
            if plot_pvals==True:
                ax.text(count+.1,x,s='p = '+str(np.round(pvalnp,3)),fontsize=fsize_tick-2)

        ax.set_xlim((-.5,count+.5))
        ax.set_xticks(np.arange(0,count+1))
        ax.set_xticklabels(field_list,fontsize=fsize_tick,rotation = 45)
        ax.set_ylabel('$t$ stat',fontsize=fsize_lbl)
        ax.set_title(beh_feat+' use_clean = '+str(use_clean),fontsize=fsize_lbl)

        if yL != None:
            ax.set_ylim(yL)

        plt.tight_layout()

                
class Collection(SubjElectrode):
    # This Class inherits from Subject Electrode.
    # Its purposes is to build group-level objects across subjects where each electrode is an observaation
    # Input should be a list of  uElbls ("subj - elec1 - elec2") and a collection_lbl (for saving purposes). if list of uElbls is empty, will process the label (e.g., allsubj-allelecs)
    # Main functions should include:
    # collecting data by looping through list of u-eLbls - should be able to collect anatomical data, 2d power, 3d power or phase, or statistical tests implemented by SubjectElectrode
    # for each object, saving/loading group level object
    # build customFeatureMatrix (combination of above data)
    # Filtering electrodes by some statistical test (write this functionality into SubjElectrode)
    # Dimensionality Reduction
    # Clustering based on feature matrix
    # Making summary plots (avg power, avg stats, or anatomical data)
    # collapsing within subjects for across subj stats etc

    # CONSTRUCTOR
    def __init__(self,uElbl_list = None, collection_lbl = 'all',paramsDict = None):

        # basic attributes
        self.collection_lbl = collection_lbl
        self.uElbl_list = uElbl_list


        # PARSE INPUTS
        # if a list of electrodes is not provide it, we construct it based on the label
        if uElbl_list == None:

            # load ccdt_info JSON file to get directories and subjInfor
            with open("ccdt_info.json", "r") as read_file:
                ccdt_info = json.load(read_file)

            # construct uElbl list by looping through subjects, and anatomical data
            self.uElbl_list = []
            self.subj_list = []
            self.mni_x =[]
            self.mni_y =[]
            self.mni_z =[]
            self.anat_list=[]
            self.anat_list_wm=[]

            for s in np.arange(0,len(ccdt_info['subjects'])):
                S = Subject(subj=ccdt_info["subjects"][s]['subj'])

                # this is all subjects, all electrodes  (except exclude flag)
                if collection_lbl == 'all':
                    if ccdt_info["subjects"][s]['exclude']==False:
                        self.uElbl_list.extend(S.tal_df['uElbl'].to_numpy())
                        self.subj_list.extend([S.subj]*len(S.tal_df))
                        self.mni_x.extend(S.tal_df['x'].to_numpy())
                        self.mni_y.extend(S.tal_df['y'].to_numpy())
                        self.mni_z.extend(S.tal_df['z'].to_numpy())
                        self.anat_list.extend(S.tal_df['anat_native'].to_numpy())
                        self.anat_list_wm.extend(S.tal_df['anat_wm'].to_numpy())

            # save master list (use this for collecting gro5up dat, this will not get filtered)
            self.uElbl_list_master = self.uElbl_list

            # this is a flag for wheter or not to apply filter when loading group data
            # gets set to True when filter is applied
            self.filtE_applyFilt = False
            self.filtE_lbl = ''


            # Initialize subj electrode with the first electrode in the list
            SubjElectrode.__init__(self,
                               subj=self.uElbl_list[0].split('-')[0],
                               sess_idx_list=None,
                               elec1_lbl=self.uElbl_list[0].split('-')[1],
                               elec2_lbl=self.uElbl_list[0].split('-')[2],
                               paramsDict=paramsDict,do_init=True)

            # Init function
            self.getBadElectrodes()
            # this will not get filtered (and can be used to get
            self.isBadElectrode_list_master = self.isBadElectrode_list.copy()


    # Get badElectrodes_list()
    def getBadElectrodes(self,overwriteFlag=False):

        # make a label for the save file
        self.groupLbl_isBadElectrode_list = ('GROUP-isBadElectrode-' +
                               self.collection_lbl)

        # look for file
        if (os.path.exists(self.params_dir+self.groupLbl_isBadElectrode_list)==True)&(overwriteFlag==False):
            #load file if it exists
            self.isBadElectrode_list = (self.load_pickle(self.params_dir+
                                                 self.groupLbl_isBadElectrode_list)
                               )

            print('Loaded Group isBadElectrode_list  ')


            if self.filtE_applyFilt == True:
                self.applyFiltE(attr_list=['isBadElectrode_list'])


        # if not, loop through and collect pow data (and save)
        else:
            # initialize
            self.isBadElectrode_list = []

            # loop
            for e in arange(0,len(self.uElbl_list_master)):
                print(e,self.uElbl_list_master[e],
                      np.round(100*(e/len(self.uElbl_list_master)),2))

                S = SubjElectrode(subj=self.uElbl_list_master[e].split('-')[0],
                                  sess_idx_list=None,
                                  elec1_lbl=self.uElbl_list_master[e].split('-')[1],
                                  elec2_lbl=self.uElbl_list_master[e].split('-')[2],
                                  paramsDict=None,do_init=True)


                # isBadElectrode
                self.isBadElectrode_list.append(S.isBadElectrode)

            #save isBadElectrode_list
            self.save_pickle(obj = self.isBadElectrode_list,
                                 fpath = self.params_dir+self.groupLbl_isBadElectrode_list)

            # apply filter
            if self.filtE_applyFilt == True:
                    self.applyFiltE(attr_list=['isBadElectrode_list'])


    def getBehXSubj(self,evQuery='error==0&fastResponse==0'):
        # This function will loop through all subjects in self.subjList and computes various behavioral measures. It then creates a dataframe in self "behXSubj_df" that can be queried as needed

        subj_list = list(np.unique(self.subj_list))
        beh_dict_list = []

        for s in subj_list:

            beh_dict = {}


            # init subj
            S = Subject(s)

            beh_dict['subj'] = S.subj

            # get RT difference
            # # std
            # rts short delay
            rts_s = S.getRTs(evQuery = evQuery+'&delay==500', rt_dist_type = 'standard')
            zrrt_s = S.getRTs(evQuery = evQuery+'&delay==500', rt_dist_type = 'zrrt')

            # rts long delay 
            rts_l = S.getRTs(evQuery = evQuery+'&delay==1500', rt_dist_type = 'standard')
            zrrt_l = S.getRTs(evQuery = evQuery+'&delay==1500', rt_dist_type = 'zrrt')

            beh_dict['rtDiff_mean'] = np.mean(rts_l)-np.mean(rts_s)
            beh_dict['zrrtDiff_mean'] = np.mean(zrrt_l)-np.mean(zrrt_s)
            beh_dict['rtDiff_std'] = np.std(rts_l)-np.std(rts_s)
            beh_dict['zrrtDiff_std'] = np.std(zrrt_l)-np.std(zrrt_s)


            # get error rate and fast response rate 
            choiceEv = S.ev_df.query('type=="RESPONSE"')
            beh_dict['error_rateL'] = np.sum(choiceEv.eval('RT<0&delay==1500'))/np.sum(choiceEv.eval('delay==1500'))
            beh_dict['error_rateS'] = np.sum(choiceEv.eval('RT<0&delay==500'))/np.sum(choiceEv.eval('delay==500'))
            beh_dict['error_diff'] = beh_dict['error_rateL'] - beh_dict['error_rateS']
            beh_dict['guess_rateL'] = np.sum(choiceEv.eval('RT<-500&delay==1500'))/np.sum(choiceEv.eval('delay==1500'))


            # count total number of too fast errors
            beh_dict['n_tooFast_errors'] = np.sum(choiceEv.eval('RT<0'))

            # fit LATER 2
            rts_A,rts_B,pred_idx_A,pred_idx_B = S.getRTs_for_LATER2()

            # FIT LATER 2
            beh_dict.update(S.fitLATER2_byCondition(rts_A,rts_B,pred_idx_A, pred_idx_B,model_type = 'std_bias'))#

            # append
            beh_dict_list.append(beh_dict)

        # convert to data frame
        behXSubj_df = pd.DataFrame(beh_dict_list,index = subj_list)

        # save in self
        self.behXSubj_df = behXSubj_df




    # COLLECTION - GET POW 2d
    def getPow_2d(self,pow_evType='FIX_START',pow_frange_lbl = 'HFA',pow_method = 'wave',pow_evQuery = None, do_zscore = True,apply_gauss_smoothing = True,gauss_sd_scaling = 0.075,apply_time_bins = True,time_bin_size_ms=50,overwriteFlag=False,paramsDict = {},num_bins = 10):
        # num_bins sets number of percentile bins for rt
        # HARD CODED Param - Need to apply_time_bins because we cannot concatonate matrixes with different sizes (due to differing number of samples)
        apply_time_bins=True
        #print('Note: Apply_time_bins is hard coded to be true when collecting power data across multiple subjects')

        # update params for self with new paramsDict inputs
        self.params.update(paramsDict) 

        #make label for group file
        # convention for pow_evQuery should be filtLbl + query
        # eg(error==0&fastResponse==0&badTrial==0&delay==500)
        self.groupLbl_pow2d = ('GROUP-POW2d-' +
                               self.collection_lbl +
                               pow_evType +
                               pow_frange_lbl +
                               pow_method +
                               str(pow_evQuery) +
                               str(do_zscore) +
                               str(apply_gauss_smoothing) +
                               str(gauss_sd_scaling) +
                               str(apply_time_bins) +
                               str(time_bin_size_ms)+str(paramsDict)
                               )

        # look for file
        if (os.path.exists(self.params_dir+self.groupLbl_pow2d)==True)&(overwriteFlag==False):
            #load file if it exists
            powMat_dict = (self.load_pickle(self.params_dir+
                                                 self.groupLbl_pow2d)
                               )

            print('Loaded Group 2d Power dictionary')

            #get e (value of iter) based on number of rows in powMat array 
            e = np.shape(powMat_dict['powMat'])[0]


        # if not, initialize powMat dict and set e to 0
        else:

            # powMat will be num electrodes x time; binPow_500 and _1500 will be 3 d - num bins X num time X num electrodes
            powMat_dict = {'powMat':np.array([]),'binPow_500':np.array([]),'binPow_1500':np.array([])}

            # set iter to 0
            e = 0

        # check if e has counted all the way through
        if (e == len(self.uElbl_list_master)):

            # This means we have counted through and updated the dictionary appropriately. We are done, no need to collect more data
            pass
            print('power group dictionary is complete')

        else:
            # this means that we are still running through the loop, continue running through the loop

            # set variable to figure out where to start, if this is the first time we are entering the loop eStart will be set to 0
            eStart = e

            for e in arange(eStart,len(self.uElbl_list_master)):
                print(e,self.uElbl_list_master[e],
                      np.round(100*(e/len(self.uElbl_list_master)),2))
                S = SubjElectrode(subj=self.uElbl_list_master[e].split('-')[0],
                                  sess_idx_list=None,
                                  elec1_lbl=self.uElbl_list_master[e].split('-')[1],
                                  elec2_lbl=self.uElbl_list_master[e].split('-')[2],
                                  paramsDict=paramsDict,do_init=True)

                # get pow 2 d
                S.getPow_2d(pow_evType=pow_evType,
                            pow_frange_lbl = pow_frange_lbl,
                            pow_method = pow_method,
                            pow_evQuery = pow_evQuery,
                            do_zscore = do_zscore,
                            apply_gauss_smoothing = apply_gauss_smoothing,
                            gauss_sd_scaling = gauss_sd_scaling,
                            apply_time_bins=apply_time_bins,
                            time_bin_size_ms=time_bin_size_ms)



                # get pow mat
                powMat = S.powMat


                # update dict with pow Mat

                if e == 0:
                    powMat_dict['powMat'] = np.nanmean(powMat,axis=0,keepdims=True)
                else:
                    powMat_dict['powMat'] = (np.append(powMat_dict['powMat'],
                                                  np.nanmean(powMat,
                                                             axis=0,
                                                             keepdims=True),
                                                  axis=0))


                # get rts
                rts = S.pow_ev_filt['RT'].to_numpy()
                delays = S.pow_ev_filt['delay'].to_numpy() 
                delay_list = [500,1500] # hard coded, because these are the conditions we are interested in. 


                # bin pow by delay condition
                for d in delay_list:

                    if (d in delays) == True:
                        # get binned pow
                        binPow_mean, binPow_sem, rts_bin_idx, bins = S.binPowByRT_2d(powMat=powMat[delays==d,:],rts = rts[delays==d],num_bins=num_bins)

                    else:
                        # create a nan matrix
                        binPow_mean = np.empty((num_bins,powMat.shape[1]))
                        binPow_mean[:] = nan

                    # update dict with binPow

                    if e == 0:
                        powMat_dict['binPow_'+str(d)] = binPow_mean[:,:,np.newaxis]
                    else:
                        powMat_dict['binPow_'+str(d)] = (np.append(powMat_dict['binPow_'+str(d)],binPow_mean[:,:,np.newaxis],axis=2))

                #save dict (within electrode loop)
                self.save_pickle(obj = powMat_dict,
                                 fpath = self.params_dir+self.groupLbl_pow2d)

            # At this point, we should have a complete dictionary and are done with the for loop

        # add x_vals etc
        self.powMat = powMat_dict['powMat']
        self.powMat_mean = np.nanmean(self.powMat,axis=0)
        self.powMat_sem = stats.sem(self.powMat,axis=0,nan_policy='omit')
        self.binPow_500 = powMat_dict['binPow_500']
        self.binPow_1500 = powMat_dict['binPow_1500']
        self.num_bins = num_bins
        self.pow_evType=pow_evType
        self.pow_frange_lbl = pow_frange_lbl
        self.pow_evQuery = pow_evQuery
        self.pow_apply_gauss_smoothing = apply_gauss_smoothing
        self.pow_do_zscore = do_zscore
        self.pow_gauss_sd_scaling = 0.1
        self.pow_apply_time_bins=apply_time_bins
        self.pow_time_bin_size_ms=time_bin_size_ms


        if apply_time_bins==True:
            # create time bins dict (for plotting purposes)
            self.makeTimeBins(time_bin_size_ms = time_bin_size_ms)
            self.pow_xval = self.time_bin_dict['lbl']
        else:
            self.pow_xval = np.linspace(self.ms_to_samples(self.params['tmin_ms']),
                                         self.ms_to_samples(self.params['tmax_ms']),
                                         np.shape(powMat)[1])

        #apply filter
        if self.filtE_applyFilt == True:
            self.applyFiltE(attr_list=['powMat','binPow_500','binPow_1500'])


    # COLLECT_POW3d
    def getPow_3d(self,pow_evType='CC',pow_evQuery = None, do_zscore = True,apply_time_bins=True,time_bin_size_ms=100,overwriteFlag = False):
        #make label for group file
        # convention for pow_evQuery should be filtLbl + query
        # eg(error==0&fastResponse==0&badTrial==0&delay==500)
        self.groupLbl_pow3d = ('GROUP-POW3d-' +
                               self.collection_lbl +
                               pow_evType +
                               str(pow_evQuery) +
                               str(do_zscore) +
                               str(apply_time_bins) +
                               str(time_bin_size_ms)
                         )
        # look for file
        if (os.path.exists(self.params_dir+self.groupLbl_pow3d)==True)&(overwriteFlag==False):
            #load file if it exists
            self.pow3d = (self.load_pickle(self.params_dir+
                                                 self.groupLbl_pow3d)
                               )

            print('Loaded Group 3d Power  ')


            if self.filtE_applyFilt == True:
                self.applyFiltE(attr_list=['pow3d'])

        # if not, loop through and collect pow data (and save)
        else:
            self.pow3d = np.array([])

            for e in arange(0,len(self.uElbl_list_master)):
                print(e,self.uElbl_list_master[e],
                      np.round(100*(e/len(self.uElbl_list_master)),2))
                S = SubjElectrode(subj=self.uElbl_list_master[e].split('-')[0],
                                  sess_idx_list=None,
                                  elec1_lbl=self.uElbl_list_master[e].split('-')[1],
                                  elec2_lbl=self.uElbl_list_master[e].split('-')[2],
                                  paramsDict=None,do_init=True)

                # get pow 2 d
                S.getPow_3d(pow_evType=pow_evType,
                            pow_evQuery = pow_evQuery,
                            do_zscore = do_zscore,
                            apply_time_bins=apply_time_bins,
                            time_bin_size_ms=time_bin_size_ms)

                if e == 0:
                    self.pow3d = np.nanmean(S.pow3d,axis=2,keepdims=True)
                else:
                    self.pow3d = (np.append(self.pow3d,
                                                  np.nanmean(S.pow3d,
                                                             axis=2,
                                                             keepdims=True),
                                                  axis=2))

            #save power
            self.save_pickle(obj = self.pow3d,
                             fpath = self.params_dir+self.groupLbl_pow3d)

            # apply filter
            if self.filtE_applyFilt == True:
                self.applyFiltE(attr_list=['pow3d'])

        # store  vars (outside if statement)
        self.pow3d_mean = np.nanmean(self.pow3d,axis=2)
        self.pow3d_sem = stats.sem(self.pow3d,axis=2)
        self.pow3d_evType=pow_evType
        self.pow3d_evQuery = pow_evQuery
        self.pow3d_do_zscore = do_zscore
        self.pow3d_apply_time_bins=apply_time_bins
        self.pow3d_time_bin_size_ms=time_bin_size_ms
        self.pow3d_freqs = self.getWaveFreqs(wave_frange=self.params['wave_frange'],
                                             wave_number_of_freq=self.params['wave_number_of_freq'])
        if apply_time_bins==True:
            # create time bins dict (for plotting purposes)
            self.makeTimeBins(time_bin_size_ms = time_bin_size_ms)
            self.pow3d_xval = self.time_bin_dict['lbl']
        else:
            self.pow3d_xval = np.linspace(self.ms_to_samples(self.params['tmin_ms']),
                                         self.ms_to_samples(self.params['tmax_ms']),
                                         np.shape(powMat)[1])

    #COLLECT_TASKSTATS2D
    def doTaskStats_2d(self,pow_frange_lbl = 'HFA',pow_method = 'wave',
    pow_evQuery = 'error==0&fastResponse==0&badTrial==0',
     do_zscore = True,apply_gauss_smoothing = True,
     gauss_sd_scaling = 0.1,apply_time_bins=True,
     time_bin_size_ms=100, num_iters = 20,regress_yvar_lbl = 'zrrt',
     overwriteFlag = False,feat_list_beh=None, run_light = True):

        if feat_list_beh is None:
            feat_list_beh_str = ''
        else:
            feat_list_beh_str = str(feat_list_beh)


        #make label for group file
        # convention for pow_evQuery should be filtLbl + query
        # eg(error==0&fastResponse==0&badTrial==0&delay==500)
        self.groupLbl_taskstats2d = ('GROUP-TASKSTATS2d-' +
                               self.collection_lbl +
                               pow_frange_lbl +
                               pow_method +
                               str(pow_evQuery) +
                               str(do_zscore) +
                               str(apply_gauss_smoothing) +
                               str(gauss_sd_scaling) +
                               str(apply_time_bins) +
                               str(time_bin_size_ms)+'num_iters'+str(num_iters)+feat_list_beh_str
                         )

        # look for file
        if (os.path.exists(self.params_dir+self.groupLbl_taskstats2d)==True)&(overwriteFlag==False):
            #load file if it exists
            self.taskstats2d = (self.load_pickle(self.params_dir+
                                                 self.groupLbl_taskstats2d)
                               )

            print('Loaded Group TaskStats2d dictionary ')

            #get e (value of iter) based on length of dict list
            e = len(self.taskstats2d)

        # else, initialize and set e to 0
        else:
            # initialize list to collect dictionaries
            self.taskstats2d = []

            # set iter to 0
            e = 0


        # check if e has counted all the way through
        if (e == len(self.uElbl_list_master)):

            # This means we have counted through and updated the dictionary appropriately. We are done, no need to collect more data
            print('dictionary is complete')
            pass

        # if not, loop through and collect task stats 2d data in a group dataframe
        else:
            
            # this means that we are still running through the loop, continue running through the loop

            # set variable to figure out where to start, if this is the first time we are entering the loop eStart will be set to 0
            eStart = e


            for e in arange(eStart,len(self.uElbl_list_master)):
                print(e,self.uElbl_list_master[e],
                      np.round(100*(e/len(self.uElbl_list_master)),2))

                if run_light == True:
                    do_init = False
                else:
                    do_init = True

                S = SubjElectrode(subj=self.uElbl_list_master[e].split('-')[0],
                                  sess_idx_list=None,
                                  elec1_lbl=self.uElbl_list_master[e].split('-')[1],
                                  elec2_lbl=self.uElbl_list_master[e].split('-')[2],
                                  paramsDict=None,do_init=True)

                # do taskstats2d
                S.doTaskStats_2d(pow_frange_lbl = pow_frange_lbl,
                            pow_method = pow_method,
                            pow_evQuery = pow_evQuery,
                            do_zscore = do_zscore,
                            apply_gauss_smoothing = apply_gauss_smoothing,
                            gauss_sd_scaling = gauss_sd_scaling,
                            apply_time_bins=apply_time_bins,
                            time_bin_size_ms=time_bin_size_ms,num_iters=num_iters,
                            regress_yvar_lbl = regress_yvar_lbl,
                            overwriteFlag=overwriteFlag,feat_list_beh=feat_list_beh)


                # append list of dict
                self.taskstats2d.append(S.taskstats2d)

                #save dictionary
                if run_light == False:
                    # save at each step
                    self.save_pickle(obj = self.taskstats2d,
                             fpath = self.params_dir+self.groupLbl_taskstats2d)

        
            # At this point, we should have a complete dictionary and are done with the for loop
            if run_light == True:
                # save at the end of the loop
                self.save_pickle(obj = self.taskstats2d,
                     fpath = self.params_dir+self.groupLbl_taskstats2d)
        # loop through the list and replace nans with a dict of nans with the correct fields
        first_good_elec_idx = np.nonzero(np.array(self.isBadElectrode_list_master)==False)[0][0]

        # get key list and make nan_dict
        key_list = self.taskstats2d[first_good_elec_idx].keys()
        nan_dict = {}
        for k in key_list:
            nan_dict[k]=np.nan
        
        for bad_idx in np.nonzero(np.array(self.isBadElectrode_list_master))[0]:
            self.taskstats2d[bad_idx] = nan_dict


        # construct a dataframe using the list of dictionaries
        self.taskstats2d_df = pd.DataFrame(self.taskstats2d,index = self.uElbl_list_master)

        # apply electrode filter
        if self.filtE_applyFilt == True:
            self.applyFiltE(attr_list=['taskstats2d_df'])


        # add params etc
        self.taskstats2d_pow_frange_lbl = pow_frange_lbl
        self.taskstats2d_pow_method = pow_method
        self.taskstats2d_pow_evQuery = pow_evQuery
        self.taskstats2d_do_zscore = do_zscore
        self.taskstats2d_apply_gauss_smoothing = apply_gauss_smoothing
        self.taskstats2d_gauss_sd_scaling = gauss_sd_scaling
        self.taskstats2d_apply_time_bins = apply_time_bins
        self.taskstats2d_time_bin_size_ms = time_bin_size_ms
        self.taskstats2d_num_iters = num_iters
        #also collect regression labels (these are not incorporated
        # into the savefilename)
        self.taskstats2d_regress_yvar_lbl = regress_yvar_lbl
        self.taskstats2d_feat_list_beh = feat_list_beh_str




    def groupElectrodesByTaskStats(self, pthresh_str = '0.05',vm_thresh_ms_str = '100',print_flag=False):
        # This function returns a dictionary with booleans for various electrode groups. Depends on task stats, Can add groups as needed. 


        def getRetIdx(C,str_list):
            # This sub-function returns a boolean to mask a group of electrodes by evaluating \
            # the str_list with "&" operators and evaluating the collection object "C"
    
            ret_str = '('
            count = 0
            for s in str_list:
                count+=1
                if count>1:
                    ret_str = ret_str+')&('+s
                else:
                    ret_str = ret_str+s
            ret_str = ret_str + ')'
    
            ret_idx = C.taskstats2d_df.eval(ret_str).to_numpy().astype('bool')
            return ret_idx

        # initialize dict
        ret_dict = {}
        
        # create bool masks
        ### rt effects ###
        #if 'zrrt' in self.taskstats2d_feat_list_beh:
        #    ret_dict['ret_idx_rtPredictive'] = getRetIdx(self,str_list =['rtRegress_multivar_zrrt_SSE_pvalnp<'+pthresh_str])


        ##### anat ####
        ret_dict['ret_idx_anat_left'] =getRetIdx(self,str_list = ['x<0'])

        ###### task-related activity #####
        # fix Inc
        ret_dict['ret_idx_fixInc'] = getRetIdx(self,str_list = ['postFix_tstat>0& postFix_pval<'+pthresh_str])
        
        # CC inc
        ret_dict['ret_idx_ccInc'] = getRetIdx(self,str_list = ['postCC_tstat>0& postCC_pval<'+pthresh_str])

        # response inc
        ret_dict['ret_idx_respInc'] = getRetIdx(self,str_list = ['postResponse_tstat>0& postResponse_pval<'+pthresh_str])

        # task increase
        ret_dict['ret_idx_taskInc'] = (ret_dict['ret_idx_fixInc'])|(ret_dict['ret_idx_ccInc'])|(ret_dict['ret_idx_respInc'])

        ret_dict['ret_idx_taskNull'] = getRetIdx(self,str_list = ['postFix_pval>'+pthresh_str+'& postCC_pval>'+pthresh_str+'& postResponse_pval>'+pthresh_str])

        # fix greatest
        ret_dict['ret_idx_fixGreatest'] = getRetIdx(self,str_list = ['(postFix_tstat>postCC_tstat)& (postFix_tstat>postResponse_tstat)'])
        # cc greatest
        ret_dict['ret_idx_ccGreatest'] = getRetIdx(self,str_list = ['(postCC_tstat>postFix_tstat)& (postCC_tstat>postResponse_tstat)'])
        # response greatest
        ret_dict['ret_idx_respGreatest'] = getRetIdx(self,str_list = ['(postResponse_tstat>postFix_tstat)& (postResponse_tstat>postCC_tstat)'])


        ###### stim vs. motor locking #####
        # stim locked
        ret_dict['ret_idx_stimLocked'] = getRetIdx(self,str_list = ['modParams_preResponseS_respLocked_amp>0& modParams_preResponseS_respLocked_cen_ms<-'+vm_thresh_ms_str,'modParams_preResponseL_respLocked_amp>0& modParams_preResponseL_respLocked_cen_ms<-'+vm_thresh_ms_str])
        # response locked
        ret_dict['ret_idx_respLocked'] = getRetIdx(self,str_list = ['modParams_preResponseS_respLocked_amp>0& modParams_preResponseS_respLocked_cen_ms>=-'+vm_thresh_ms_str,'modParams_preResponseL_respLocked_amp>0& modParams_preResponseL_respLocked_cen_ms>=-'+vm_thresh_ms_str])
               
        ###### spatial and feedback selectivity #####
        # stim locked
        ret_dict['ret_idx_spatialSel'] = getRetIdx(self,str_list = ['tlAnova_pval<'+pthresh_str])

        # error increase
        ret_dict['ret_idx_errInc'] = getRetIdx(self,str_list = ['errSel_tstat>0& errSel_pval<'+pthresh_str])  

        # error decrease
        ret_dict['ret_idx_errDec'] = getRetIdx(self,str_list = ['errSel_tstat<0& errSel_pval<'+pthresh_str])  

        # reward increase
        ret_dict['ret_idx_rewInc'] = getRetIdx(self,str_list = ['rewSel_tstat>0& rewSel_pval<'+pthresh_str])  
            
        # error decrease
        ret_dict['ret_idx_rewDec'] = getRetIdx(self,str_list = ['rewSel_tstat<0& rewSel_pval<'+pthresh_str])   

        ###### modelled gaussians #####
        ret_dict['ret_idx_modTargOn'] = getRetIdx(self,str_list = ['modParams_postTargOn_amp>0'])
        ret_dict['ret_idx_modPostCCS'] = getRetIdx(self,str_list = ['modParams_postCCS_amp>0'])
        ret_dict['ret_idx_modPreResponseS'] = getRetIdx(self,str_list = ['modParams_preResponseS_respLocked_amp>0'])
        ret_dict['ret_idx_modPostResponseS'] = getRetIdx(self,str_list = ['modParams_postResponseS_respLocked_amp>0'])
        ret_dict['ret_idx_modPostNoCCS'] = getRetIdx(self,str_list = ['modParams_postNoCCS_amp>0'])
        ret_dict['ret_idx_modPreResponseL'] = getRetIdx(self,str_list = ['modParams_preResponseL_respLocked_amp>0'])
        ret_dict['ret_idx_modPostResponseL'] = getRetIdx(self,str_list = ['modParams_postResponseL_respLocked_amp>0'])


        ###### define mutually-exclusive groups #####

        # RAMP
        ret_dict['ret_idx_grp_ramp'] = (ret_dict['ret_idx_taskInc']) & (ret_dict['ret_idx_modTargOn']==False)& (ret_dict['ret_idx_modPreResponseS']==False)& (ret_dict['ret_idx_modPostResponseS']==False)& (ret_dict['ret_idx_modPostNoCCS']==False)& (ret_dict['ret_idx_modPreResponseL']==False)& (ret_dict['ret_idx_modPostResponseL']==False)

        # convenience bool to track electrodes that have been assigned
        hasGrp = ret_dict['ret_idx_grp_ramp'];
        #print('num hasGroup',sum(hasGrp))


        # MOTOR
        ret_dict['ret_idx_grp_motor'] = (ret_dict['ret_idx_taskInc']) & (hasGrp==False) & (ret_dict['ret_idx_ccGreatest']) & (ret_dict['ret_idx_respLocked'])
        ret_dict['ret_idx_grp_motor_left'] = (ret_dict['ret_idx_grp_motor']&ret_dict['ret_idx_anat_left'])
        ret_dict['ret_idx_grp_motor_right'] = (ret_dict['ret_idx_grp_motor'])&(ret_dict['ret_idx_anat_left']==False)

        hasGrp = hasGrp | (ret_dict['ret_idx_grp_motor'])
        #print('num hasGroup',sum(hasGrp))

        # VISUAL
        ret_dict['ret_idx_grp_visual'] = (ret_dict['ret_idx_taskInc']) & (hasGrp==False) & (ret_dict['ret_idx_stimLocked'])

        hasGrp = hasGrp | (ret_dict['ret_idx_grp_visual'])

        # REWARD/TEXT
        ret_dict['ret_idx_grp_rewTex'] = (ret_dict['ret_idx_taskInc']) & (hasGrp==False) & (ret_dict['ret_idx_respGreatest']) & (ret_dict['ret_idx_modPostResponseS']) & (ret_dict['ret_idx_modPostResponseS']) & (ret_dict['ret_idx_modPostResponseL']) & (ret_dict['ret_idx_modPostResponseL'])

        hasGrp = hasGrp | (ret_dict['ret_idx_grp_rewTex'])
        #print('num hasGroup',sum(hasGrp))




        # EXPECTATION
        ret_dict['ret_idx_grp_exp'] = (ret_dict['ret_idx_taskInc']) & (hasGrp==False) &  (ret_dict['ret_idx_modPostNoCCS'])
        #print('num hasGroup',sum(hasGrp))
        hasGrp = hasGrp | (ret_dict['ret_idx_grp_exp'])


        # other
        ret_dict['ret_idx_grp_other'] = (ret_dict['ret_idx_taskInc']) & (hasGrp==False)
        hasGrp = hasGrp | (ret_dict['ret_idx_grp_other'])





        #ret_idx_m = getRetIdx(self,str_list = ['postCC_tstat>0& postCC_pval<'+pthresh_str,'vmIndex_tstat>0&vmIndex_pval<'+pthresh_str,'postCC_tstat>postFix_tstat)&(postCC_tstat>postResponse_tstat'])
        #ret_dict['ret_idx_m'] = ret_idx_m
        #print('num motor',sum(ret_idx_m))

        #ret_idx_r = (ret_idx_m==False)&getRetIdx(self,str_list = ['postResponse_tstat>0& postResponse_pval<'+pthresh_str,'postResponse_tstat>postFix_tstat','postResponse_tstat>postCC_tstat'])
        #ret_dict['ret_idx_r'] = ret_idx_r
        #print('num text/reward',sum(ret_idx_r))

        #ret_idx_v = (ret_idx_m==False)&(ret_idx_r==False)&getRetIdx(self,str_list = ['postCC_tstat>0& postCC_pval<'+pthresh_str,'mvIndex_tstat>0&mvIndex_pval<'+pthresh_str,'postFix_tstat>0& postFix_pval<'+pthresh_str,'postResponse_tstat>0& postResponse_pval<'+pthresh_str,'postFix_tstat>postResponse_tstat','postCC_tstat>postResponse_tstat'])
        #ret_dict['ret_idx_v'] = ret_idx_v
        #print('num visual',sum(ret_idx_v))

        if print_flag == True:
            for k in ret_dict.keys():
                print('num '+k.split('_')[-1]+'...'+str(sum(ret_dict[k]))+'...'+str(np.count_nonzero(self.collapseBySubj_1d(ret_dict[k]))))

        # return ret_dict
        return ret_dict

    #COLLECT_xPAC
    def pac_xPAC(self,eeg_evType = 'CC',eeg_evQuery = 'error==0&fastResponse==0&badTrial==0',eeg_apply_filt=False,
        cat_method = 'choice',idpac_findAmp = (2,0,0), idpac_xPAC = (2,3,4),
        frange_pha = None,frange_amp = None,frange_pha_lbl = 'LFO',frange_amp_lbl = 'HFO',n_bins = 50, f_width = 10,n_perm=1000,overwriteFlag = False):
        # This function collects group data related to PAC analyses on continous EEG data

        # generate file name (with parameters)
        #make label for group file
        # convention for pow_evQuery should be filtLbl + query
        # eg(error==0&fastResponse==0&badTrial==0&delay==500)
        self.groupLbl_xPAC = ('GROUP-xPAC' +
                               self.collection_lbl +
                               eeg_evType +
                               str(eeg_evQuery) +
                               str(eeg_apply_filt) +
                               str(cat_method) +
                               str(idpac_findAmp) +
                               str(idpac_xPAC) +
                               str(frange_pha) +
                               str(frange_amp) +
                               str(frange_pha_lbl) +
                               str(frange_amp_lbl) +
                               str(n_bins) +
                               str(f_width) +
                               str(n_perm)
                               )

        # look for saved file
        if (os.path.exists(self.params_dir+self.groupLbl_xPAC)==True)&(overwriteFlag==False):
            #load file if it exists
            self.xPAC = (self.load_pickle(self.params_dir+
                                                 self.groupLbl_xPAC)
                               )

            print('Loaded Group xPAC dictionary ')

            #get e (value of iter) based on length of dict 
            e = len(self.xPAC)

        # else, initialize and set e to 0
        else:
            # initialize list to collect dictionaries
            self.xPAC = []

            # set iter to 0
            e = 0

        # check if e has counted all the way through and xPAC has been updated
        if (e == len(self.uElbl_list_master)):

            # This means we have counted through and updated the dictionary appropriately. We are done, no need to collect more data
            print('the dictionary is complete')
            pass


  
        else:
            # this means that we are still running through the loop, continue running through the loop

            # set variable to figure out where to start, if this is the first time we are entering the loop eStart will be set to 0
            eStart = e

            # loop through electrodes
            for e in arange(eStart,len(self.uElbl_list_master)):
                print(e,self.uElbl_list_master[e],
                      np.round(100*(e/len(self.uElbl_list_master)),2))
                
                # initialize SE
                S = SubjElectrode(subj=self.uElbl_list_master[e].split('-')[0],
                                  sess_idx_list=None,
                                  elec1_lbl=self.uElbl_list_master[e].split('-')[1],
                                  elec2_lbl=self.uElbl_list_master[e].split('-')[2],
                                  paramsDict=None,do_init=True)

                # check if bad Electrode
                if S.isBadElectrode == True:
                    # populate with nans
                    # populate dictionary with  nan
                    xPAC_dict = {}

                    # fooof data
                    xPAC_dict['ff_LFO_frange'] = nan
                    xPAC_dict['ff_LFO_frange_mean'] = nan
                    xPAC_dict['ff_LFO_peak'] = nan
                    xPAC_dict['ff_r_sq'] = nan
                    xPAC_dict['ff_alpha'] = nan
                    xPAC_dict['ff_aperiodic'] = nan
                    xPAC_dict['ff_beta'] = nan
                    xPAC_dict['ff_gamma'] = nan
                    xPAC_dict['ff_theta'] = nan
                    
                    # xpac data
                    xPAC_dict['pac_xpac_frange_pha'] = nan # LFO
                    xPAC_dict['pac_xpac_frange_amp'] = nan # HFO
                    xPAC_dict['pac_xpac'] = nan
                    xPAC_dict['pac_xpac_pval'] = nan 
                    xPAC_dict['pac_xpac_shuf'] = nan


                else:

                    # do xPAC pipeline as follows

                    # get eeg for the trials that meet the evQuery
                    S.getEEG(eeg_evType=eeg_evType,eeg_evQuery = eeg_evQuery,
                    eeg_apply_filt=eeg_apply_filt)

                    # concatonate EEG to get continuous time series across trials
                    S.catEEG(cat_method = cat_method) # fixation on to Response

                    # Identify dominant Low frequency oscillation (LFO) using FOOOF
                    S.calcLFO()

                    # Apply PAC to find optimal amplitude bandwith and plot co-modulogram
                    S.pac_findAmpRange(idpac = idpac_findAmp,frange = frange_pha,frange_lbl = frange_pha_lbl, n_bins = n_bins,f_width = f_width)

                    # compute continuous PAC using LFO and HFO (optimal HF range)
                    S.pac_xPAC(idpac = idpac_xPAC, n_bins = n_bins,
                    n_perm = n_perm,frange_pha = frange_pha,frange_amp = frange_amp,
                    frange_pha_lbl = frange_pha_lbl,frange_amp_lbl = frange_amp_lbl)

                    # populate dictionary with key data
                    xPAC_dict = {}

                    # fooof data
                    xPAC_dict['ff_LFO_frange'] = S.ff_LFO_frange
                    xPAC_dict['ff_LFO_frange_mean'] = np.nanmean(S.ff_LFO_frange)
                    xPAC_dict['ff_LFO_peak'] = S.ff_LFO_peak
                    xPAC_dict['ff_r_sq'] = S.ff_r_sq
                    xPAC_dict['ff_alpha'] = S.ff_alpha
                    xPAC_dict['ff_aperiodic'] = S.ff_aperiodic
                    xPAC_dict['ff_beta'] = S.ff_beta
                    xPAC_dict['ff_gamma'] = S.ff_gamma
                    xPAC_dict['ff_theta'] = S.ff_theta
                    
                    # xpac data
                    xPAC_dict['pac_xpac_frange_pha'] = S.pac_xpac_frange_pha # LFO
                    xPAC_dict['pac_xpac_frange_amp'] = S.pac_xpac_frange_amp # HFO
                    xPAC_dict['pac_xpac'] = S.pac_xpac
                    xPAC_dict['pac_xpac_pval'] = S.pac_xpac_pval 
                    xPAC_dict['pac_xpac_shuf'] = S.pac_xpac_shuf

                # append list of dict
                self.xPAC.append(xPAC_dict)

                # save dict
                self.save_pickle(obj = self.xPAC,
                             fpath = self.params_dir+self.groupLbl_xPAC)


        # At this point, we should have a complete dictionary and are done with the for loo

        # construct a dataframe using the list of dictionaries
        self.xPAC_df = pd.DataFrame(self.xPAC,index = self.uElbl_list_master)

        # apply filter
        if self.filtE_applyFilt == True:
            self.applyFiltE(attr_list=['xPAC_df'])


        
        # add params etc
        self.xPAC_eeg_evType = eeg_evType
        self.xPAC_eeg_evQuery = eeg_evQuery
        self.xPAC_eeg_apply_filt = eeg_apply_filt
        self.xPAC_cat_method = cat_method
        self.xPAC_idpac_findAmp = idpac_findAmp
        self.xPAC_idpac_xPAC = idpac_xPAC
        self.xPAC_frange_pha = frange_pha
        self.xPAC_frange_amp = frange_amp
        self.xPAC_frange_pha_lbl = frange_pha_lbl
        self.xPAC_frange_amp_lbl = frange_amp_lbl
        self.xPAC_n_bins = n_bins
        self.xPAC_f_width = f_width
        self.xPAC_n_perm = n_perm

    ## FILTER ELECTRODES
    def filterElectrodes(self,filtE_bool = None, taskstats2dQuery = None, xpacQuery = None, talQuery = None, clusQuery = None):
        # This function filters electrodes within a collection based on a combinaation of electrode level data (e.g., taskstats, anatomy, cluster assignment) as provided in the input
        # It will filter all collected group data so far (e.g, 2d pow, 3d pow, anat, taskstats2d_df, xPac_df)
        # It will toggle apply filt to true such that any newly collected group data are filtered appropriatley
        # The workflow should be to initialize a collection object, load the group data that you want to use to filter electrodes (e.g., taskstats2d_df), then load additional group dataa that you want to analyze (stats and figures)
        # To revert to the entire electrode collection, it is best to initialize a new collection object


        #(this will be used by group data functions eg
        #getPow2d and 3d)
        self.filtE_applyFilt = True

        # parse inputs
        self.filtE_taskstats2dQuery = taskstats2dQuery
        self.filtE_xpacQuery = xpacQuery
        self.filtE_talQuery = talQuery
        self.filtE_clusQuery = clusQuery
        self.filtE_lbl = ''
        self.filtE_bool = filtE_bool


        #parse filtE bool
        # if filtE_bool is not provided, it generates a boolean mask based on queries
        if np.all(self.filtE_bool == None):

            # initialize bool_mask
            self.filtE_bool = np.ones(len(self.uElbl_list_master)).astype('bool')
            taskstats2d_bool =np.ones(len(self.uElbl_list_master)).astype('bool')
            xpac_bool= np.ones(len(self.uElbl_list_master)).astype('bool')
            tal_bool = np.ones(len(self.uElbl_list_master)).astype('bool')
            clus_bool = np.ones(len(self.uElbl_list_master)).astype('bool')

            if taskstats2dQuery!=None:
                taskstats2d_bool = self.taskstats2d_df.eval(taskstats2dQuery).to_numpy().astype('bool')
                self.filtE_lbl = self.filtE_lbl+taskstats2dQuery

                if xpacQuery!=None:
                    xpac_bool = self.xPAC_df.eval(xpacQuery).to_numpy().astype('bool')
                    self.filtE_lbl = self.filtE_lbl+xpacQuery

                # [] to do
                if talQuery!=None:
                    pass

                # [] to do
                if clusQuery!=None:
                    pass

                # get overall mask
                self.filtE_bool = taskstats2d_bool&xpac_bool&tal_bool&clus_bool
                self.filtE_bool = self.filtE_bool.astype('bool')

        # applyFiltE
        self.applyFiltE(attr_list=['uElbl_list',
                                   'subj_list',
                                   'mni_x',
                                   'mni_y',
                                   'mni_z',
                                  'anat_list','anat_list_wm',
                                  'isBadElectrode_list',
                                  'powMat','binPow_500','binPow_1500',
                                  'pow3d',
                                  'taskstats2d_df',
                                  'xPAC_df'])

    # apply filter
    def applyFiltE(self,attr_list = None):

        #
        for a in attr_list:

            if hasattr(self,a):
                x = getattr(self,a)

                if a == 'powMat':
                    x = x[self.filtE_bool,:]

                    # recalc mean and sem of power
                    self.powMat_mean = np.nanmean(x,axis=0)
                    self.powMat_sem = stats.sem(x,axis=0,nan_policy='omit')


                elif np.any(a in ['pow3d','binPow_500','binPow_1500']):
                    x = x[:,:,self.filtE_bool]

                    if a == 'pow3d':
                        self.pow3d_mean = np.nanmean(x,axis=2)
                        self.pow3d_sem = stats.sem(x,axis=2,nan_policy='omit')

                elif type(x)==list:
                    x = np.array(x)[self.filtE_bool]
                else:
                    x = x[self.filtE_bool]

                setattr(self,a,x)




    ## COLLAPSE BY SUBJ
    def collapseBySubj_wrapper(self,attr = 'powMat'):
        #This function collapses group data by subj
        # based on inputs. It calls various subfcn depending
        # on the attribute that we want to collapse
        if attr == 'powMat':

            self.powMat = self.collapseBySubj_2d(x = self.powMat)

            # recalc mean and sem of power
            self.powMat_mean = np.nanmean(self.powMat,axis=0)
            self.powMat_sem = stats.sem(self.powMat,axis=0,nan_policy='omit')


        elif attr == 'pow3d':
            self.pow3d = self.collapseBySubj_3d(x = self.pow3d)

            self.pow3d_mean = np.nanmean(self.pow3d,axis=2)
            self.pow3d_sem = stats.sem(self.pow3d,axis=2,nan_policy='omit')


    def collapseBySubj_1d(self,x, subj_list_ret_idx = None):
        #This function collapses a 1d array within subject (e.g., regress_t_zrrt_St). If x has been masked, then provide subj_list_ret_idx to apply the same mask to self.subj_list
 
        # apply ret_idx to subj_list 
        if subj_list_ret_idx is None:
            subj_list_ret_idx = np.ones(len(self.uElbl_list)).astype('bool') 

        subj_list = list(np.array(self.subj_list.copy())[subj_list_ret_idx])

        uSubj_list = np.unique(subj_list)
        x_grp = np.zeros((len(uSubj_list)))


        for i in arange(0,len(uSubj_list)):
            s_idx = np.array(subj_list)==uSubj_list[i]
            x_grp[i] = np.nanmean(x[s_idx])

        return x_grp


    def collapseBySubj_2d(self,x, subj_list_ret_idx = None):
        #This function collapses a 2d matrix within subject (e.g., powMat)
        # assumes that rows are electrodes and columns are features.  If x has been masked, then provide subj_list_ret_idx to apply the same mask to self.subj_list
        # apply ret_idx to subj_list 
        if subj_list_ret_idx is None:
            subj_list_ret_idx = np.ones(len(self.uElbl_list)).astype('bool') 
        subj_list = list(np.array(self.subj_list.copy())[subj_list_ret_idx])

        uSubj_list = np.unique(subj_list)
        x_grp = np.zeros((len(uSubj_list),x.shape[1]))

        for i in arange(0,len(uSubj_list)):
            s_idx = np.array(subj_list)==uSubj_list[i]
            x_grp[i,:] = np.nanmean(x[s_idx,:],axis = 0,keepdims=True)

        return x_grp

    def collapseBySubj_3d(self,x):
        #This function collapses a 2d matrix within subject (e.g., powMat)
        # assumes that rows are electrodes and columns are features

        uSubj_list = np.unique(self.subj_list)
        x_grp = np.zeros((x.shape[0],x.shape[1],len(uSubj_list)))

        for i in arange(0,len(uSubj_list)):
            s_idx = np.array(self.subj_list)==uSubj_list[i]
            x_grp[:,:,i] = np.nanmean(x[:,:,s_idx],axis = 2,keepdims=False)

        return x_grp
    def retIdx2masterRetIdx(self, ret_idx):
        """
        This function generates a boolean matched to uElbl_master based on the boolean provided (for filtered electodes)
        ret_idx .... bool of length uElbl_list (after filtering)

        Returns:
        ret_idx_master ... same electrodes represented in boolean of length uElbl_list_master
        """
        

        # get uElbls
        ret_uElbls = list(self.uElbl_list[ret_idx])

        # initialize master_re
        master_ret_idx = np.zeros(len(self.uElbl_list_master)).astype('bool')

        # loop through and fill in electrodes that were assigned to this cluster
        for i in ret_uElbls:

            master_ret_idx[self.uElbl_list_master.index(i)] = True

        return master_ret_idx

    def getAnatDf(self,ret_idx=None,cmap='rainbow',atlas='default'):
        # this function returns a dataframe for selected electrodes indicated by ret_idx (bool) that includes key anatomical data that can be used for anatomical plots
        # cmap assigns colors to each region based on the given colormap
        # atlas .. anatomical atlas to use for labeling and color.
        #.....'default', it uses neuromorphometrics labels along with probtrackX white matter labels and assigns colors based on colormap
        #.....'yeo', it uses yeo buckner intrinsic brain network parcellation normative functional connectivity data (defaults to '7 network version'). Assigns colors to each region per the atlas

        # returns dataframe and sorted roi list (from anterior to posterior)

        # parse inputs
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')
        
        if atlas=='yeo':
            atlas_dict = self.loadAtlas_yeo()

        # get list of anatomical labels
        anat_list = np.array(self.anat_list.copy())[ret_idx]
        anat_list_wm = np.array(self.anat_list_wm.copy())[ret_idx]

        # initialize dict list
        anat_dict_list = []
        rois = [] # will add this to the dataframe after parsing unlabelled electrodes
        mni_coords = np.zeros((len(anat_list),3))

        # loop through electrodes laterality
        for a in np.arange(0,len(anat_list)):
            # initialize dict
            anat_dict = {}

            # collect mni coords
            mni_coords[a,0] = np.array(self.mni_x)[ret_idx][a]
            mni_coords[a,1] = np.array(self.mni_y)[ret_idx][a]
            mni_coords[a,2] = np.array(self.mni_z)[ret_idx][a]


            # parse hemisphere
            if self.mni_x[a]<0:
                anat_dict['hemis'] = 'L'
            else:
                anat_dict['hemis'] = 'R'


            # parse WM
            if 'White Matter' in anat_list[a]:
                anat_dict['isWM'] = True
            else:
                anat_dict['isWM'] = False

            # assign anatomical label
            if atlas == 'yeo':
                rois.append(self.mni2roi(atlas_dict,mni_coords[a,0],mni_coords[a,1],mni_coords[a,2]))


            elif atlas == 'default':
                # Get ROI based on anat_list and anat_list_wm
                anat_lbl = self.parse_anat_lbl(anat_list[a],anat_list_wm[a])

                # error check (to make sure prefrontal is )
                if (self.anat2roi(anat_lbl)=='Prefrontal') & ((np.array(self.mni_y)[ret_idx][a])<=-10):
                    anat_lbl = self.parse_anat_lbl('White Matter',anat_list_wm[a])

                # convert to roi
                rois.append(self.anat2roi(anat_lbl))

            # append to dict list
            anat_dict_list.append(anat_dict)

        
        # replace 'unlabelled' rois
        rois = self.parse_unabelled(rois,mni_coords)

        # convert to data frame
        anatDf = pd.DataFrame(anat_dict_list,index = np.array(self.uElbl_list)[ret_idx])

        # add rois to anatDf
        anatDf.insert(len(anatDf.columns),'roi',rois)

        # get roi_list
        roi_list = np.unique(np.array(rois))

        # get mean mni y (ant-post) for each region
        mni_y = []      

        for r in roi_list:
            mni_y.append(np.nanmean(np.array(self.mni_y)[ret_idx][r==np.array(rois)]))

        # arange roi_list by ant-post axis
        sort_idx = np.argsort(mni_y)
        roi_list = list(np.array(roi_list)[sort_idx])


        # loop through sorted roi_list and assign colors using specified colormap
        from matplotlib import cm
        colmap = cm.get_cmap(cmap,100) 
        c_idx = np.linspace(0,1,len(roi_list)) # use this to index colors for each region

        # array holding colors for each electrode
        e_colors = np.zeros((len(rois),4))
        for r in roi_list:

            # find matching electrodes
            r_idx = (np.array(rois) == r)

            # [ ] PARSE ATLAS LABEL HERE
            #BACKHERE

            # update colors with a color from color list
            e_colors[r_idx,:] = np.tile(colmap(c_idx[roi_list.index(r)]),(np.sum(r_idx),1))


        # update anat df with a color
        anatDf.insert(len(anatDf.columns),'roi_color',list(e_colors))

        # update anatDf w MNI coords
        anatDf.insert(len(anatDf.columns),'mni_x',mni_coords[:,0])
        anatDf.insert(len(anatDf.columns),'mni_y',mni_coords[:,1])
        anatDf.insert(len(anatDf.columns),'mni_z',mni_coords[:,2])

        # update w raw anat list
        anatDf.insert(len(anatDf.columns),'anat',anat_list)
        anatDf.insert(len(anatDf.columns),'anat_wm',anat_list_wm)


        return anatDf, roi_list
    def getEligibleRegions(self,min_subj_thresh=5,min_elec_thresh=50):
        # appears to be DEPRECIATED
        # returns list of regions that meet minimum criteria
        pass
        # get anatDf and roi_list
        anatDf,roiList = self.getAnatDf(ret_idx=None)

        # container
        roi_list_eligible = []

        for r in roiList:
            # ret_idx
            ret_idx = anatDf.eval('roi==@r')

            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))
        
            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):
                continue
            roi_list_eligible.append(r)
        return roi_list_eligible




    ##### STATS 
    def stats_ttest1(self,x,collapseBySubj_flag):
        # funciton to compute a one sample t-test. returns t and p. Allows one to collapse data within subject to result in an across subj t-test. Variable must have same mapping between each observation and subject as in self.subj_list 

        if collapseBySubj_flag==True:
            x = self.collapseBySubj_1d(x)

        tstat,pval = stats.ttest_1samp(x,popmean=0,nan_policy='omit')
        return tstat,pval

    def stats_ttest2(self,x,y,collapseBySubj_flag):
        # same as above function but for a paired two-sample t-test

        if collapseBySubj_flag==True:
            x = self.collapseBySubj_1d(x)

        tstat,pval = stats.ttest_ind(x,y,nan_policy='omit')
        return tstat,pval


    ##### PLOT Corr results
    def plotCorr_results(self,use_clean = True, ax = None,collapseBySubj_flag = True,figsize=(5,5),fsize_tick=14,beh_feat = 'zrrt',neu_feat_list=['S0f','S0c','postCC','preResponse'], yL = None,delay_str='',ret_idx = None):
        # This function plots the results from the RT correlation analysis. It plots non-parametric z-stats that can be aggregated across subjects
        # inputs:
        # use_clean = True; this flag sets whether to plot residual HFA data that has been "cleaned" of stimulus and response locked components
        # delay_str ... indicates whether to use all trials (''), short delay trials only ('S'), or long delay trials only ('L')
        # ret_idx allows you to filter within a subset of electrodes (e.g. for a particular region or cluster)

        # make figure
        if ax==None:
            fig = plt.figure(figsize = figsize)
            ax = plt.subplot(111)

        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')

        # parse clean str
        if use_clean == True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'

        #parse delay lbl
        if delay_str == '':
            delay_lbl = 'all trials'
        elif delay_str == 'S':
            delay_lbl = 'short delay'
        elif delay_str == 'L':
            delay_lbl = 'long delay'

        xtick_lbl = []
        yvar_lbl = beh_feat
        xvar_lbl =neu_feat_list

        # loop through x var list and t-stats
        for i in range(0,len(xvar_lbl)):

            xtick_lbl.append(xvar_lbl[i].split('_')[-1])

            # get stats
            x = self.taskstats2d_df['rtCorr'+delay_str+'_'+beh_feat+'_'+xvar_lbl[i]+clean_str+'_zstatnp'].to_numpy()[ret_idx]

            # collapse by subj (optional)
            if collapseBySubj_flag == True:
                x = self.collapseBySubj_1d(x,subj_list_ret_idx=ret_idx)

            ax.bar(i+1,np.nanmean(x),color='.7',edgecolor='k')
            ax.errorbar(i+1,np.nanmean(x),stats.sem(x,nan_policy='omit'),ecolor='k')
        plt.xticks(ticks=(arange(0,len(xvar_lbl)))+1,
        labels = xtick_lbl,
        rotation=45,
        fontsize=fsize_tick)
        plt.ylabel('t-stat ' + yvar_lbl)

        if yL != None:
            ax.set_ylim(yL)


        plt.title('n electrodes = '+str(np.count_nonzero(ret_idx))+' n subj = '+str(np.count_nonzero(self.collapseBySubj_1d(ret_idx)))+'\n Collapse by subj = '+str(collapseBySubj_flag)+'\n'+delay_lbl+'; '+clean_lbl)
        plt.tight_layout()


    ##### PLOT REGRESS COUNTS
    def plotCorr_counts(self,use_clean = True,ax = None,
        p_thresh = 0.05,figsize=(5,5),fsize_tick=12,fsize_lbl=16,beh_feat = 'zrrt',neu_feat_list=['S0f','S0c','postCC','postCC_bur','preResponse_bur','preResponse'],yL=None,delay_str='',ret_idx=None,plot_z = False,plot_counts=True):
        # This function plots Counts results from the RT correlation analysis. p_thresh sets threshold for identifying "significant" electrodes
        # inputs:
        # use_clean = True; this flag sets whether to plot residual HFA data that has been "cleaned" of stimulus and response locked components
        # delay_str ... indicates whether to use all trials (''), short delay trials only ('S'), or long delay trials only ('L')
        # ret_idx allows you to filter within a subset of electrodes (e.g. for a particular region or cluster)



        # make figure
        if ax==None:
            fig = plt.figure(figsize = figsize)
            ax = plt.subplot(111)
        ax2 = ax.twinx()


        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')

        # parse clean str
        if use_clean == True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'

        #parse delay lbl
        if delay_str == '':
            delay_lbl = 'all trials'
        elif delay_str == 'S':
            delay_lbl = 'short delay'
        elif delay_str == 'L':
            delay_lbl = 'long delay'

        xtick_lbl = []
        yvar_lbl = beh_feat
        xvar_lbl =neu_feat_list

        # initialize containers to hold counts
        n_tot = np.zeros(len(xvar_lbl))
        n_exp = np.zeros(len(xvar_lbl))
        n_obs = np.zeros(len(xvar_lbl))
        n_obs_pos = np.zeros(len(xvar_lbl))
        n_obs_neg = np.zeros(len(xvar_lbl))


        # loop through x var list and t-stats
        for i in range(0,len(xvar_lbl)):

            # get stats
            x = self.taskstats2d_df['rtCorr'+delay_str+'_'+beh_feat+'_'+xvar_lbl[i]+clean_str+'_zstatnp'].to_numpy()[ret_idx]

            # get pvals
            pvals = self.taskstats2d_df['rtCorr'+delay_str+'_'+beh_feat+'_'+xvar_lbl[i]+clean_str+'_pvalnp'].to_numpy()[ret_idx]

            counts_pos = (x>0)&(pvals<=p_thresh)
            counts_neg = (x<0)&(pvals<=p_thresh)


            n_tot[i] = len(counts_pos)
            n_exp[i] = p_thresh*(n_tot[i])
            n_obs[i] = np.sum(counts_pos+counts_neg)
            n_obs_pos[i] = np.sum(counts_pos)
            n_obs_neg[i] = np.sum(counts_neg)

            # plot bar showing histogram of counts on ax 1
            if plot_counts == True:
                ax.bar(i+1,(np.count_nonzero(counts_pos)/len(counts_pos))*100,color='none',
                edgecolor='k',label = '(+) effects')
                ax.bar(i+1,-1*(np.count_nonzero(counts_neg)/len(counts_neg))*100, color='none',
                edgecolor='k',label = '(-) effects')


            # plot error of pos and neg z-stats
            if plot_z == True:

                if plot_counts==True:
                    # plot on ax 2
                    ax2.errorbar(i+1,np.nanmean(x[counts_pos]),\
                        1.96*np.nanstd(x[counts_pos]),fmt='none',ecolor='k',alpha=0.5)
                    ax2.errorbar(i+1,np.nanmean(x[counts_neg]),\
                        1.96*np.nanstd(x[counts_neg]),fmt='none',ecolor='k',alpha=0.5)
                else:
                    # plot on ax 1
                    ax.errorbar(i+1,np.nanmean(x[counts_pos]),\
                        1.96*np.nanstd(x[counts_pos]),fmt='none',ecolor='k',alpha=0.5)
                    ax.errorbar(i+1,np.nanmean(x[counts_neg]),\
                        1.96*np.nanstd(x[counts_neg]),fmt='none',ecolor='k',alpha=0.5)
                    ax2.axis('off')
            else:
                ax2.axis('off')

            #if i == 0:
            #    plt.legend(fontsize=fsize_tick)

        # clean up xtick
        xtick_lbls = []
        for xv in xvar_lbl:
            #xtick_lbls.append(xv)#.split('_')[-1]
            xtick_lbls.append(xvar_lbl.index(xv))
        plt.xticks(ticks=(np.arange(0,len(xvar_lbl)))+1,
        labels = xtick_lbls,
        rotation=90,
        fontsize=fsize_tick)

        # set ylabels
        if plot_counts==True:
            ax.set_ylabel('% electrodes \n significant',fontsize=fsize_lbl)
        else:
            ax.set_ylabel('z-stat \n (non-parametric)',fontsize=fsize_lbl)
        ax.set_xlim(0.5,len(xvar_lbl)+0.5)

        ax2.set_ylabel('z-stat \n (non-parametric)',fontsize=fsize_lbl)
        ax2.set_xlim(0.5,len(xvar_lbl)+0.5)

        # set ytick labels
        if yL!=None:
            ax.set_ylim(yL)
        ax.set_yticklabels(np.abs(ax.get_yticks()).astype('int'))
        #ax2.set_yticklabels(np.abs(ax2.get_yticks()))




        ax.set_title(' total n_elec = '+str(np.count_nonzero(ret_idx))+' n subj = '+str(np.count_nonzero(self.collapseBySubj_1d(ret_idx)))+'; p_thresh = '+str(np.round(p_thresh,2))+'\n'+delay_lbl+'; '+clean_lbl)
        plt.tight_layout()

        # return variables for post-hoc stats
        plot_dict = {}
        plot_dict['n_tot'] =n_tot
        plot_dict['n_exp'] =n_exp
        plot_dict['n_obs'] =n_obs
        plot_dict['n_obs_pos'] =n_obs_pos
        plot_dict['n_obs_neg'] =n_obs_neg
        plot_dict['neu_feat_list'] =xvar_lbl
        
        return plot_dict
    def plotDelayDiff_counts(self,use_clean = True,ax = None,
        p_thresh = 0.05,figsize=(5,5),fsize_tick=12,fsize_lbl=16,neu_feat_list=['S0f','S0c','postCC','postCC_bur','preResponse_bur','preResponse'],yL=None,ret_idx=None):
        # This function plots Counts results from the delay diff analysis p_thresh sets threshold for identifying "significant" electrodes
        # inputs:
        # use_clean = True; this flag sets whether to plot residual HFA data that has been "cleaned" of stimulus and response locked components
        # ret_idx allows you to filter within a subset of electrodes (e.g. for a particular region or cluster)




        # make figure
        if ax==None:
            fig = plt.figure(figsize = figsize)
            ax = plt.subplot(111)
        #ax2 = ax.twinx()

        #plot 


        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')

        # parse clean str
        if use_clean == True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'

        xtick_lbl = []
        xvar_lbl =neu_feat_list

        # initialize containers to hold counts
        n_tot = np.zeros(len(xvar_lbl))
        n_exp = np.zeros(len(xvar_lbl))
        n_obs = np.zeros(len(xvar_lbl))
        n_obs_pos = np.zeros(len(xvar_lbl))
        n_obs_neg = np.zeros(len(xvar_lbl))


        # loop through x var list and t-stats
        for i in range(0,len(xvar_lbl)):

            # skip if xvar lbl is pre-target baseline
            if xvar_lbl[i] == 'S0f':
                continue

            # get stats
            x = self.taskstats2d_df['delayDiff_'+xvar_lbl[i]+clean_str+'_zstatnp'].to_numpy()[ret_idx]

            # get pvals
            pvals = self.taskstats2d_df['delayDiff_'+xvar_lbl[i]+clean_str+'_pvalnp'].to_numpy()[ret_idx]


            counts_pos = (x>0)&(pvals<=p_thresh)
            counts_neg = (x<0)&(pvals<=p_thresh)


            n_tot[i] = len(counts_pos)
            n_exp[i] = p_thresh*(n_tot[i])
            n_obs[i] = np.sum(counts_pos+counts_neg)
            n_obs_pos[i] = np.sum(counts_pos)
            n_obs_neg[i] = np.sum(counts_neg)


            # plot bar showing histogram of counts on ax 1
            ax.bar(i+1,(np.count_nonzero(counts_pos)/len(counts_pos))*100,color='0.5',
            edgecolor='k',label = '(+) effects')
            ax.bar(i+1,-1*(np.count_nonzero(counts_neg)/len(counts_neg))*100, color='0.5',
            edgecolor='k',label = '(-) effects')


            # # plot error of pos and neg z-stats
            # ax2.errorbar(i+1,np.nanmean(x[counts_pos]),\
            #     1.96*np.nanstd(x[counts_pos]),fmt='none',ecolor='k',alpha=0.5)
            # ax2.errorbar(i+1,np.nanmean(x[counts_neg]),\
            #     1.96*np.nanstd(x[counts_neg]),fmt='none',ecolor='k',alpha=0.5)

            #if i == 0:
            #    plt.legend(fontsize=fsize_tick)

        # clean up xtick
        xtick_lbls = []
        for xv in xvar_lbl:
            #xtick_lbls.append(xv)#.split('_')[-1]
            xtick_lbls.append(xvar_lbl.index(xv))
        ax.set_xticks(np.arange(0,len(xvar_lbl))+1)
        ax.set_xticklabels(xtick_lbls)

        # set ylabels
        ax.set_ylabel('% electrodes \n significant',fontsize=fsize_lbl)
        #ax2.set_ylabel('z-stat (non-parametric)',fontsize=fsize_lbl)

        # set ytick labels
        if yL!=None:
            ax.set_ylim(yL)
        ax.set_yticklabels(np.abs(ax.get_yticks()).astype('int'))
        #ax2.set_yticklabels(np.abs(ax2.get_yticks()))


        ax.set_title(' total n_elec = '+str(np.count_nonzero(ret_idx))+' n subj = '+str(np.count_nonzero(self.collapseBySubj_1d(ret_idx)))+'; p_thresh = '+str(np.round(p_thresh,2))+clean_lbl)

        plt.tight_layout()

        ax.set_xlim(0.5,len(xvar_lbl)+0.5)

        # return variables for post-hoc stats
        plot_dict = {}
        plot_dict['n_tot'] =n_tot
        plot_dict['n_exp'] =n_exp
        plot_dict['n_obs'] =n_obs
        plot_dict['n_obs_pos'] =n_obs_pos
        plot_dict['n_obs_neg'] =n_obs_neg
        plot_dict['neu_feat_list'] =xvar_lbl
        
        return plot_dict


    def plotRegressCountsByRegion_SSE(self,ax = None,figsize=(7,5),beh_feat = 'zrrt',p_thresh=0.05,ret_idx = None,fsize_tick=14,use_colormap = False,cmap='rainbow'):

        # DEPRECIATED, use evalClus_anat instead
        return None

        # """bar plot of number of sig electrodes by region of interest. It is intended to be rotated 90 degrees"""
        # # make figure
        # if ax is None:
        #     f = plt.figure(figsize =figsize)
        #     ax = plt.subplot(111)

        # # parse ret_idx
        # if ret_idx is None:
        #     ret_idx = np.ones(len(self.uElbl_list)).astype('bool')

        # # get anatDf 
        # anatDf,roi_list = self.getAnatDf(ret_idx=ret_idx,cmap = cmap);

        # # get roi, roi_color, and hemisphere
        # roi = anatDf['roi'].to_numpy()
        # roi_color = anatDf['roi_color'].to_numpy()
        # hemis = anatDf['hemis'].to_numpy()

        # # get p val array and apply ret_idx  
        # pvals = self.taskstats2d_df['rtRegress_multivar_'+beh_feat+'_SSE'+'_pvalnp'].to_numpy()[ret_idx]

        # # initialize containers to hold counts
        # n_tot = np.zeros(len(roi_list))
        # n_exp = np.zeros(len(roi_list))
        # n_obs = np.zeros(len(roi_list))

        # # for laterality measurements
        # n_tot_right = np.zeros(len(roi_list))
        # n_tot_left = np.zeros(len(roi_list))
        
        # n_obs_right = np.zeros(len(roi_list))
        # n_exp_right = np.zeros(len(roi_list)) # based on sampling bias (left v right hemisphere)
        # n_obs_left = np.zeros(len(roi_list))



        # # loop through regions and count
        # for i in np.arange(0,len(roi_list)):

        #     # calculate overall counts
        #     this_roi_idx = (roi==roi_list[i]) #electrodes in this region

        #     n_tot[i] = np.sum(this_roi_idx).astype('int') # total obs
        #     n_exp[i] = p_thresh*(n_tot[i]) # expected number based on p-thresh
        #     n_obs[i] = np.sum(pvals[this_roi_idx]< p_thresh) # observed n sig. elecs 
        #     prct_sig = 100*(n_obs[i]/n_tot[i]) # percent sig (for plotting)


        #     # calculate hemisphere counts
        #     n_tot_right[i] = np.sum((this_roi_idx)&(hemis=='R')).astype('int') # total in left hemisphere for this region
        #     n_tot_left[i] = np.sum((this_roi_idx)&(hemis=='L')).astype('int') # total in right hemisphere for this region
        #     n_obs_right[i] = np.sum(pvals[(this_roi_idx)&(hemis=='R')]<p_thresh).astype('int') # actual number of sig electrodes observed in right
        #     n_obs_left[i] = np.sum(pvals[(this_roi_idx)&(hemis=='L')]<p_thresh).astype('int') # actual number of sig electrodes observed in right
        #     n_exp_right[i] = n_obs[i] * (n_tot_right[i]/n_tot[i]) # expectation is that sig. electrodes are distributed between left and right hemisphere per sampling bias
        #     LR_bias_obs = (n_obs_right[i]/n_obs[i]) # deviations from 0.5 indicate bias. > 0.5 indicates rightward bias, < 0.5 indicates leftward bias
        #     LR_bias_exp = (n_tot_right[i]/n_tot[i]) # deviations indicate sampling bias

        #     # plot bar 
        #     if use_colormap == True:
        #         c = roi_color[np.where(this_roi_idx)[0][0]] 
        #     else:
        #         c = '0.5'

        #     # plot percent significants
        #     ax.bar(i,prct_sig,color=c,edgecolor='k')

        #     # plot text (indicating electrode counts)
        #     ax.text(i-.15,2,str(int(n_obs[i]))+'/'+str(int(n_tot[i])),rotation=90,fontsize=fsize_tick)

        #     # plot LR bias
        #     y_start = prct_sig+5
        #     y_stop = y_start+10
        #     ax.plot((i,i),(y_start,y_stop),color = c)
        #     ax.plot(i,(y_start+(LR_bias_obs*(y_stop-y_start))),'o',markerfacecolor = c, markeredgecolor = 'k',markersize=5)
        #     ax.plot(i,(y_start+(LR_bias_exp*(y_stop-y_start))),marker='_',markeredgecolor = '0.5',linewidth=1)

        # # set xlim
        # ax.set_xlim(-0.5,len(roi_list)-0.5)

        # # plot h line
        # ax.hlines(p_thresh*100,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='--',colors = 'k',alpha = '0.5')

        # plt.xticks(ticks=(np.arange(0,len(roi_list))),
        # labels = roi_list,
        # rotation=90,
        # fontsize=fsize_tick) 


        # #rotate y tick labels
        # plt.gca().set_yticklabels(plt.gca().get_yticks().astype('int'),rotation=90)

        # # set y label
        # ax.set_ylabel ('% RT predictive')

        # #set title
        # plt.title('Number of electrodes = '+str(np.sum(ret_idx)))


        # plt.tight_layout()

        # # return variables for post-hoc stats
        # plot_dict = {}
        # plot_dict['n_tot'] =n_tot
        # plot_dict['n_exp'] =n_exp
        # plot_dict['n_obs'] =n_obs
        # plot_dict['roi_list'] =roi_list

        # plot_dict['n_tot_right'] =n_tot_right
        # plot_dict['n_tot_left'] =n_tot_left
        # plot_dict['n_obs_right'] =n_obs_right
        # plot_dict['n_exp_right'] =n_exp_right
        # plot_dict['n_obs_left'] =n_obs_left
    
        # return plot_dict

    def plotXSubjCorr(self,ax=None,beh_xSubj='rtDiff_mean',neu_feat='S0c',beh_feat = 'zrrt',use_clean = True, delay_str = '',ret_idx = None,figsize=(5,5),fsize_lbl=16, compare_delay = False):

        #Here we test relation between a specif neural feature (e.g., S0c zstat) and a behavioral measurement

        # It first obtain a neural feature (zstatistic) for a group of electrodes specify (e.g., from a cluster or region). Then it collapses that feature within subject such that there is one measure per subject. It also returns a subject list

        # Then it uses that subject list to obtain a measurement of behavioral performance. 

        # To do this efficiently, it runs self.getBehXSubj() that will loop through all subjects and assign a df to self (beh_df). Once that is created, it can query it as needed. If it hasnt been created, it will compute these data

        # then it plots a scatter between neural and behavioral variability using spearman correlation


        # Note: 'beh_feat' indicates the behavioral feature that is used to compute relation between neural activity and behavior within subject (e.g., zrrt). 'beh_xSubj' indicates the behavioral feature to compute across subject variability (e.g., 'rtDiff_mean')


        # 'compare_delay' is a bool that sets whether or not to take the difference between neural feature (long delay - short delay). if this is true, delay_str must be == ''

        # make figure
        if ax==None:
            fig = plt.figure(figsize = figsize)
            ax = plt.subplot(111)


        # error check
        if (compare_delay==True)&((delay_str=='')==False):
            raise NameError('compare_delay is true, delay_str must be "''"" ') 


        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')

        # parse clean str
        if use_clean == True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'

        #parse delay lbl
        if delay_str == '':
            delay_lbl = 'all trials'
        elif delay_str == 'S':
            delay_lbl = 'short delay'
        elif delay_str == 'L':
            delay_lbl = 'long delay'

        #### get neural feature
        if compare_delay == False:
            neu = self.taskstats2d_df['rtCorr'+delay_str+'_'+beh_feat+'_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]
        else:
            neuL = self.taskstats2d_df['rtCorrL'+'_'+beh_feat+'_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]
            neuS = self.taskstats2d_df['rtCorrS'+'_'+beh_feat+'_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]
            neu = neuL-neuS

        # collapse by subj
        neu_subj = self.collapseBySubj_1d(neu,subj_list_ret_idx=ret_idx)

        # get subj_list 
        subj_ret_idx = self.collapseBySubj_1d(ret_idx)

        ### get beahavioral feature
        # check if we have computed beh_df
        if hasattr(self,'behXSubj_df') == False:
            # compute acr
            self.getBehXSubj()
        # query our specific behavioral feature (for this list of subjects)
        beh_subj = self.behXSubj_df.iloc[np.nonzero(subj_ret_idx)[0]][beh_xSubj].to_numpy()

        # plot scatter
        x,y = self.plot_scatter(ax = ax, x=neu_subj,y=beh_subj,color = '0.5',use_spearman=True,s=200)


        #set labels
        ax = plt.gca()
        ax.set_xlabel(neu_feat,fontsize=fsize_lbl)
        ax.set_ylabel(beh_feat,fontsize=fsize_lbl)
        #ax.set_title('n subj = '+str(np.sum(subj_ret_idx)))

    def plotXSubjCorr_list(self,ax=None,beh_xSubj='rtDiff_mean',neu_feat_list=['S0f','S0c','postCC','postCC_bur','preResponse_bur','preResponse'],beh_feat = 'zrrt',use_clean = True, delay_str = '',ret_idx = None,figsize=(5,5),fsize_lbl=16,fsize_tick = 16, compare_delay = False):

        #Here we test relation between severa neural features (e.g., S0c zstat) and a behavioral measurement. Same desciption as plotXSubjCorr, but it loops through several neural features and summarizes the correlation via bar plot. 

        # It first obtain a neural feature (zstatistic) for a group of electrodes specify (e.g., from a cluster or region). Then it collapses that feature within subject such that there is one measure per subject. It also returns a subject list

        # Then it uses that subject list to obtain a measurement of behavioral performance. 

        # To do this efficiently, it runs self.getBehXSubj() that will loop through all subjects and assign a df to self (beh_df). Once that is created, it can query it as needed. If it hasnt been created, it will compute these data

        # then it plots a scatter between neural and behavioral variability using spearman correlation


        # Note: 'beh_feat' indicates the behavioral feature that is used to compute relation between neural activity and behavior within subject (e.g., zrrt). 'beh_xSubj' indicates the behavioral feature to compute across subject variability (e.g., 'rtDiff_mean')

        # make figure
        if ax==None:
            fig = plt.figure(figsize = figsize)
            ax = plt.subplot(111)
        # error check
        if (compare_delay==True)&((delay_str=='')==False):
            raise NameError('compare_delay is true, delay_str must be "''"" ') 

        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')

        # parse clean str
        if use_clean == True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'

        #parse delay lbl
        if delay_str == '':
            delay_lbl = 'all trials'
        elif delay_str == 'S':
            delay_lbl = 'short delay'
        elif delay_str == 'L':
            delay_lbl = 'long delay'

        #### get neural feature

        count=-1
        for neu_feat in neu_feat_list:

            count+=1


            #### get neural feature
            if compare_delay == False:
                neu = self.taskstats2d_df['rtCorr'+delay_str+'_'+beh_feat+'_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]
            else:
                neuL = self.taskstats2d_df['rtCorrL'+'_'+beh_feat+'_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]
                neuS = self.taskstats2d_df['rtCorrS'+'_'+beh_feat+'_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]
                neu = neuL-neuS

            # collapse by subj
            neu_subj = self.collapseBySubj_1d(neu,subj_list_ret_idx=ret_idx)

            # get subj_list 
            subj_ret_idx = self.collapseBySubj_1d(ret_idx)

            ### get beahavioral feature
            # check if we have computed beh_df
            if hasattr(self,'behXSubj_df') == False:
                # compute acr
                self.getBehXSubj()
            # query our specific behavioral feature (for this list of subjects)
            beh_subj = self.behXSubj_df.iloc[np.nonzero(subj_ret_idx)[0]][beh_xSubj].to_numpy()

            # compute spearman corr
            rval,pval = stats.spearmanr(neu_subj,beh_subj)

            # plot bar
            plt.bar(count,rval,color='0.5',edgecolor='k')

            #
            if pval <0.05:
                plt.text(count,rval,'p = '+str(np.round(pval,3)),fontsize=fsize_tick)

        ax.set_ylim(-1,1)
        ax.set_xlim(-0.5,len(neu_feat_list)+0.5)


        #set labels
        plt.xticks(ticks=(np.arange(0,len(neu_feat_list))),
        labels = neu_feat_list,rotation=90,fontsize=fsize_tick)
        plt.ylabel('% electrodes')
        ax.set_ylabel('Spearman $r$',fontsize=fsize_lbl)
        ax.set_title(beh_xSubj+' '+ax.get_title())
        #ax.set_title('n subj = '+str(np.sum(subj_ret_idx)))
    def plotXSubjDelayDiff(self,ax=None,beh_xSubj='rtDiff_mean',neu_feat='S0c',use_clean = True, ret_idx = None,figsize=(5,5),fsize_lbl=16,s=200):

        #Here we test relation between a delay-related difference in a specific neural feature (e.g., S0c ) and a behavioral measurement

        # similar to xSubjCorr methods above, except adjusted to use delayDiff rather than rtCorr data see documentation for those functions

        # make figure
        if ax==None:
            fig = plt.figure(figsize = figsize)
            ax = plt.subplot(111)

        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')

        # parse clean str
        if use_clean == True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'


        #### get neural feature
        neu = self.taskstats2d_df['delayDiff_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]

        # collapse by subj
        neu_subj = self.collapseBySubj_1d(neu,subj_list_ret_idx=ret_idx)

        # get subj_list 
        subj_ret_idx = self.collapseBySubj_1d(ret_idx)

        ### get beahavioral feature
        # check if we have computed beh_df
        if hasattr(self,'behXSubj_df') == False:
            # compute acr
            self.getBehXSubj()
        # query our specific behavioral feature (for this list of subjects)
        beh_subj = self.behXSubj_df.iloc[np.nonzero(subj_ret_idx)[0]][beh_xSubj].to_numpy()

        this_subj_list =self.behXSubj_df.iloc[np.nonzero(subj_ret_idx)[0]]['subj'].to_numpy()

        # plot scatter
        # apply across subj z-score
        x,y = self.plot_scatter(ax = ax, x=neu_subj,y=beh_subj,color = '0.5',use_spearman=True,s=s,text_lbls = this_subj_list)



        #set labels
        ax = plt.gca()
        ax.set_xlabel('$\Delta$ '+neu_feat,fontsize=fsize_lbl)
        ax.set_ylabel(beh_xSubj,fontsize=fsize_lbl)
        #ax.set_title('n subj = '+str(np.sum(subj_ret_idx)))
    ########  Elec Report - PLOT TASKSTATS TIME COURSE #######

    # plotTaskStats2d_timeCourse - over write
    def plotTaskStats2d_timeCourse(self,taskstats2d_S, ax = None, lbl=None,evType = 'FIX_START', yL = None, xL_ms = None,add_vline=True,fsize_lbl=16,fsize_tick=16,alpha = 0.6,color_short = None,color_long = None,add_legend = False,add_title=False,overlay_model_fit = True,use_clean_data=False):
        # This function overwrites the version from subjElectrode. It uses taskstats data from a Collection dataframe instead of self.taskstats rate plots the power time course for the signal that was used to compute taskstats. Separately plots long and short delay trials. 

        # requires input of taskstats2d_S ... a taskstats series for a single subj electrode object
        # overlay_model_fit = True. If true, will plot the best fitting model estimation of the activation function

        # set ax
        if ax == None:
            f = plt.figure()
            ax = plt.subplot(111)

        # parse inputs
        if lbl == None:
            lbl_short = 'short delay'
            lbl_long = 'long delay'
        else:
            lbl_short = lbl+' short delay'
            lbl_long = lbl+' long delay'

        if color_short == None:
            color_short  = 'C0'
        if color_long == None:
            color_long = 'C1'

        # get pow data
        if evType == 'FIX_START':
            fieldLbl = 'postFix'
            xLbl = 'target onset'
            fieldLbl2 = ''
        elif evType == 'CC':
            fieldLbl = 'postCC'
            xLbl = 'color change'
            fieldLbl2 = '_ccLocked'
        elif evType == 'RESPONSE':
            fieldLbl = 'postResponse'
            xLbl = 'response'
            fieldLbl2 = '_respLocked'

        # get x values
        pow_xval = taskstats2d_S['timeCourse_xval'+'_'+fieldLbl]
        # create subj object (for sample rate conversion)
        subj = taskstats2d_S.name.split('-')[0]
        elec1_lbl = taskstats2d_S.name.split('-')[1]
        elec2_lbl = taskstats2d_S.name.split('-')[2]
        E = Electrode(subj=taskstats2d_S.name.split('-')[0],sess_idx=0,elec1_lbl=elec1_lbl,elec2_lbl=elec2_lbl,paramsDict=None,do_init=False)

        # parse xlim 
        if xL_ms == None:
            if self.taskstats2d_apply_time_bins == False:
                xL_ms = (E.samples_to_ms(pow_xval[0]),E.samples_to_ms(pow_xval[-1]))
            else:
                xL_ms = (pow_xval[0],pow_xval[-1])

        # get pow data
        if evType == 'FIX_START':
            fieldLbl = 'postFix'
            xLbl = 'target onset'
        elif evType == 'CC':
            fieldLbl = 'postCC'
            xLbl = 'color change'
        elif evType == 'RESPONSE':
            fieldLbl = 'postResponse'
            xLbl = 'response'

        if use_clean_data == False:
            powMat_short_mean = taskstats2d_S[fieldLbl+'_timeCourseShort_mean']
            powMat_long_mean = taskstats2d_S[fieldLbl+'_timeCourseLong_mean']
        else:
            powMat_short_mean = taskstats2d_S['responseS_clean'+fieldLbl2]
            powMat_long_mean = taskstats2d_S['responseL_clean'+fieldLbl2]


        powMat_short_sem = taskstats2d_S[fieldLbl+'_timeCourseShort_sem']
        powMat_long_sem = taskstats2d_S[fieldLbl+'_timeCourseLong_sem']


        # plot short delay
        ax.plot(pow_xval,powMat_short_mean,label=lbl_short,alpha=alpha,color = color_short)
        ax.fill_between(pow_xval,powMat_short_mean+powMat_short_sem,powMat_short_mean-powMat_short_sem,alpha=alpha-.2,color = color_short)

        # # plot long delay
        ax.plot(pow_xval,powMat_long_mean,label=lbl_long,alpha=alpha,color = color_long)
        ax.fill_between(pow_xval,powMat_long_mean+powMat_long_sem,powMat_long_mean-powMat_long_sem,alpha=alpha-.2,color = color_long)





        # if x val are in samples, then covert tick labels
        if self.taskstats2d_apply_time_bins == False:
            ax.set_xlim((E.ms_to_samples(xL_ms[0]),E.ms_to_samples(xL_ms[1])))
            xt = np.array([E.ms_to_samples(xL_ms[0]),0,0.5*E.samplerate,E.ms_to_samples(xL_ms[1])])
            ax.set_xticks(xt)
            ax.set_xticklabels(np.round(1000*(xt/E.samplerate)).astype('int'),fontsize=fsize_tick)
        else:
            ax.set_xlim((xL_ms[0],xL_ms[1]))
            xt = np.array([xL_ms[0],0,pow_xval[np.argmin(np.abs(pow_xval-500))],xL_ms[-1]])
            ax.set_xticks(xt)
            ax.set_xticklabels(xt,fontsize=fsize_tick)

        ax.set_xlabel('Time from '+xLbl+' (ms)',fontsize=fsize_lbl)
        if self.taskstats2d_do_zscore == True:
            ax.set_ylabel('z-score '+self.taskstats2d_pow_frange_lbl,fontsize=fsize_lbl)
        else:
            ax.set_ylabel('Power (a.u.)',fontsize=fsize_lbl)

   
        # set ylim
        if yL != None:
            ax.set_ylim(yL)
        else:
            yL = ax.get_ylim()
   
        # set yticklabels
        ax.set_yticks(np.linspace(yL[0], yL[1],5))
        ax.set_yticklabels(np.round(np.linspace(yL[0], yL[1],5),2), fontsize=fsize_tick)

        # v line
        if add_vline==True:
            # get vL_ticks
            if self.taskstats2d_apply_time_bins == False:
                if evType=='FIX_START':
                    vL_ticks = [0,int(0.5*E.samplerate),int(1.5*E.samplerate)]
                else:
                    vL_ticks = [0]

            else:
                if evType=='FIX_START':
                    vL_ticks= [pow_xval[np.argmin(np.abs(pow_xval-0))],pow_xval[np.argmin(np.abs(pow_xval-500))],pow_xval[np.argmin(np.abs(pow_xval-1500))]]
                else:
                    vL_ticks= [pow_xval[np.argmin(np.abs(pow_xval-0))]]

            for v in vL_ticks:
                if (v > xL_ms[0]) & (v < xL_ms[1]):
                    ax.vlines(x=v,
                      ymin=ax.get_ylim()[0],
                      ymax=ax.get_ylim()[1],
                      linestyles='--',alpha=0.5)
        # legend
        if add_legend == True:
            ax.legend(fontsize=fsize_tick)

        #title
        if add_title == True:
            ax.set_title(E.anat_dict['anat_native'],fontsize=fsize_tick)
            #ax.set_title(self.anat_dict['uElbl']+'-'+self.anat_dict['anat_native'],fontsize=fsize_tick)

        if overlay_model_fit == True:
        # over lay model fit (for short and long delay trials)

            if evType == 'FIX_START':
                ax2 = ax.twinx()
                ax2.plot(pow_xval,stats.zscore(taskstats2d_S['mod_fullS']),linestyle='--',color = 'k',linewidth = 3,alpha = 0.25)
                ax3 = ax.twinx()
                ax3.plot(pow_xval,stats.zscore(taskstats2d_S['mod_fullL']),linestyle='--',color = 'k',linewidth = 3,alpha = 0.25) 
                ax2.axis('off')
                ax3.axis('off')
            elif evType == 'RESPONSE':
                ax2 = ax.twinx()
                ax2.plot(pow_xval,stats.zscore(taskstats2d_S['mod_fullS_respLocked']),linestyle='--',color = 'k',linewidth = 3,alpha = 0.25)
                ax3 = ax.twinx()
                ax3.plot(pow_xval,stats.zscore(taskstats2d_S['mod_fullL_respLocked']),linestyle='--',color = 'k',linewidth = 3,alpha = 0.25) 
                ax2.axis('off')
                ax3.axis('off')

 ################ CLUSTERING DATA #####################
    def clusterElectrodesByTaskStats(self,feat_option = 'selectivity',num_levels_to_cut = 100,binarize_stats=True,p_thresh = 0.05,atlas='default'):
        # This function perfoms heirarchical clustering of gaussian modeled neural response functions. Assumes that taskstats has already been run and is initialized in self

        ##### dictionary with various collection of features ####
        feat_list_dict = {}
        feat_list_dict['activation_fix'] = ['modParams_responseTrend_zval','modParams_postTargOn_amp','modParams_postTargOn_cen_ms','modParams_postTargOn_wid_ms','modParams_postCCS_amp','modParams_postCCS_cen_ms','modParams_postCCS_wid_ms','modParams_periResponseS_amp','modParams_periResponseS_cen_ms','modParams_periResponseS_wid_ms','modParams_postNoCCS_amp','modParams_postNoCCS_cen_ms','modParams_postNoCCS_wid_ms','modParams_postCCL_amp','modParams_postCCL_cen_ms','modParams_postCCL_wid_ms','modParams_periResponseL_amp','modParams_periResponseL_cen_ms', 'modParams_periResponseL_wid_ms']



        feat_list_dict['activation_resp'] = ['modParams_responseTrend_zval','modParams_postTargOn_amp','modParams_postTargOn_cen_ms','modParams_postTargOn_wid_ms','modParams_preResponseS_respLocked_amp','modParams_preResponseS_respLocked_cen_ms','modParams_preResponseS_respLocked_wid_ms','modParams_postResponseS_respLocked_amp','modParams_postResponseS_respLocked_cen_ms','modParams_postResponseS_respLocked_wid_ms','modParams_postNoCCS_amp','modParams_postNoCCS_cen_ms','modParams_postNoCCS_wid_ms','modParams_preResponseL_respLocked_amp','modParams_preResponseL_respLocked_cen_ms','modParams_preResponseL_respLocked_wid_ms','modParams_postResponseL_respLocked_amp','modParams_postResponseL_respLocked_cen_ms', 'modParams_postResponseL_respLocked_wid_ms']

        feat_list_dict['activation_ramp'] = ['modParams_postCCS_amp','modParams_postCCS_cen_ms','modParams_postCCS_wid_ms','modParams_periResponseS_amp','modParams_periResponseS_cen_ms','modParams_periResponseS_wid_ms','modParams_postCCL_amp','modParams_postCCL_cen_ms','modParams_postCCL_wid_ms','modParams_periResponseL_amp','modParams_periResponseL_cen_ms', 'modParams_periResponseL_wid_ms']


        feat_list_dict['selectivity'] = []
        beh_feat = self.taskstats2d_df['feat_list_beh'][0][0]
        for n in ['S0f','S0c','postCC','postCC_bur','preResponse_bur','preResponse']:
            feat_list_dict['selectivity'].extend(['rtCorr_'+beh_feat+'_'+n+'_clean'+'_zstatnp'])

            # add delay-related difference (not clean)
            if (n == 'S0f') == False:
                feat_list_dict['selectivity'].extend(['delayDiff_'+n+'_zstatnp'])
 
        feat_list_dict['selectivity_rtOnly'] = []
        beh_feat = self.taskstats2d_df['feat_list_beh'][0][0]
        for n in ['S0f','S0c','postCC','postCC_bur','preResponse_bur','preResponse']:
            feat_list_dict['selectivity_rtOnly'].extend(['rtCorr_'+beh_feat+'_'+n+'_clean'+'_zstatnp'])

        feat_list_dict['selectivity_delayOnly'] = []
        beh_feat = self.taskstats2d_df['feat_list_beh'][0][0]
        for n in ['S0f','S0c','postCC','postCC_bur','preResponse_bur','preResponse']:
            
            # add delay-related difference (not clean)
            if (n == 'S0f') == False:
                feat_list_dict['selectivity_delayOnly'].extend(['delayDiff_'+n+'_zstatnp'])

        feat_list_dict['selectivity_anat'] = []
        beh_feat = self.taskstats2d_df['feat_list_beh'][0][0]
        for n in ['S0f','S0c','postCC','postCC_bur','preResponse_bur','preResponse']:
            feat_list_dict['selectivity_anat'].extend(['rtCorr_'+beh_feat+'_'+n+'_clean'+'_zstatnp'])

            # add delay-related difference (not clean)
            if (n == 'S0f') == False:
                feat_list_dict['selectivity_anat'].extend(['delayDiff_'+n+'_zstatnp'])

        # add anatomical label as another field
        feat_list_dict['selectivity_anat'].extend(['roi'])


        # to include pt correlation
        #feat_list_dict['selectivity'].extend(['ptCorr_S0f_rval'])
        #feat_list_dict['selectivity'].extend(['ptCorr_postTarg_rval'])




        # not including:

        #feat_list_dict['selectivity'] = ['tlAnova_zstat','errSel_tstat','tooFastSel_tstat','rewSel_tstat']

       ############  1. Build a feature matrix based on taskstats #######

        # build feature matrix using feature lists above
        if feat_option == 'activation_selectivity':
            # combine lists of features
            feat_list = list(np.unique(feat_list_dict['activation_fix']+feat_list_dict['activation_resp']+feat_list_dict['selectivity']))
        elif feat_option == 'activation_full':
            # combine lists of features
            feat_list = list(np.unique(feat_list_dict['activation_fix']+feat_list_dict['activation_resp']))      
        elif feat_option == 'ramp_selectivity':
            # combine lists of features
            feat_list = list(np.unique(feat_list_dict['activation_ramp']+feat_list_dict['selectivity'])) 

        else:
            # use an individual list
            feat_list = feat_list_dict[feat_option]



        # initialize container for feature mat
        featMat = np.zeros((len(self.uElbl_list),len(feat_list)))
        featMat[:] = np.nan
        self.clus_featMat_noz = np.copy(featMat)

        # populate feat mat using taskstats2d_df (normalize as you go)
        for f in feat_list:

            if f == 'roi':
                # assign value based on anatDf
                anatDf,roiList = self.getAnatDf(atlas=atlas);

                # this converts anatomical labels to integer values that can be used in clustering below. Note: the numerical values will be sorted alphabetically and not based on the order in roiList (which is posterior to anterior)

                x,lookup_table = anatDf['roi'].factorize(sort=True) 
            else:
                # else pull from taskstats df
                x = np.copy(self.taskstats2d_df[f].to_numpy())
                x[np.isinf(x)] = np.nan

            # binarize amplitudes so we dont sort based on effect size. Use a z-score threshold of 1
            amp_thresh = 1
            if '_amp' in f:
                x[x>amp_thresh] = 1
                x[x<-amp_thresh] = -1

            if binarize_stats == True:
                # binarize zstats based on significance  
                if '_zstatnp' in f:
                    is_sig = self.taskstats2d_df[f.split('_zstatnp')[0]+'_pvalnp'].to_numpy()<p_thresh
                    x[is_sig==False] = 0
                    x[x>0] = 1
                    x[x<0] = -1
                if '_rval' in f:
                    is_sig = self.taskstats2d_df[f.split('_rval')[0]+'_pval'].to_numpy()<p_thresh
                    x[is_sig==False] = 0
                    x[x>0] = 1
                    x[x<0] = -1
                # binarize tstats based on significance  
                if '_tstat' in f:
                    is_sig = self.taskstats2d_df[f.split('_tstat')[0]+'_pval'].to_numpy()<p_thresh
                    x[is_sig==False] = 0
                    x[x>0] = 1
                    x[x<0] = -1

                # binarize zstat based on significance (this is a one way zstat, so all negative zstats will be set to 0 based on the pvalue)
                if ('_zstat' in f) & (('_zstatnp' in f)==False):
                    is_sig = self.taskstats2d_df[f.split('_zstat')[0]+'_pval'].to_numpy()<p_thresh
                    x[is_sig==False] = 0
                    x[x>0] = 1
                    x[x<0] = -1

            # replace nans with constant. This applies to center_ms and amp_ms. Ise zero here so that z-scores are interpretable
            x[np.isnan(x)] = 0

            #save a copy of feat mat without z-scoring
            self.clus_featMat_noz[:,feat_list.index(f)] = np.copy(x)


            # z-score before clustering
            featMat[:,feat_list.index(f)] = (x-np.nanmean(x))/np.nanstd(x)

            #featMat[:,feat_list.index(f)] = (self.taskstats2d_df[f].to_numpy() - np.nanmean(self.taskstats2d_df[f].to_numpy()))/np.nanstd(self.taskstats2d_df[f].to_numpy())

        ########### 2. Run heirachical clustering #####################

        from sklearn.cluster import AgglomerativeClustering

        # set distance threshold = 0 to compute the full tree
        clustering = AgglomerativeClustering(n_clusters= None, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=0)


        # fit it to data
        clustering.fit(featMat)

        ########### 3. create linkage matrix to be used with scipy functions#####################


        # Create linkage matrix and then plot the dendrogram
        # taken from scklearn user guide
        #https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
        # create the counts of samples under each node
        counts = np.zeros(clustering.children_.shape[0])
        n_samples = len(clustering.labels_)
        for i, merge in enumerate(clustering.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        clus_Z = np.column_stack([clustering.children_, clustering.distances_,
                                          counts]).astype(float)

        ########### 4. Cut the tree so you have a heirarchical index of cluster assignments #####################

        cut_tree = cluster.hierarchy.cut_tree(clus_Z,n_clusters=None)

        # flip cut_tree left/right so that column 0 refers to labels for highest level (electrodes split into two groups), and column -1 refers to lowest level (each electrode is its own cluster). Trucate the cut tree at num_levels_to_cut such that -1 column represents electrode groupings at num_levels_to_cut


        cut_tree = np.fliplr(cut_tree)[:,np.arange(0,num_levels_to_cut)]
        ########### 5. Calculate error measures as a funciton of number of levels #####################

        clus_num_obs = np.zeros(np.shape(cut_tree)[1])
        clus_num_obs[:] = np.nan

        clus_sil_score = np.zeros(np.shape(cut_tree)[1])
        clus_sil_score[:] = np.nan

        clus_cal_score= np.zeros(np.shape(cut_tree)[1])
        clus_cal_score[:] = np.nan

        clus_db_score= np.zeros(np.shape(cut_tree)[1])
        clus_db_score[:] = np.nan

        for i in np.arange(0,np.shape(cut_tree)[1]):
            clus_num_obs[i] = np.count_nonzero(cut_tree[:,i]==0)

            if (len(np.unique(cut_tree[:,i]))>1) & (i < np.shape(cut_tree)[0]):
                clus_sil_score[i]=silhouette_score(X=featMat, labels=cut_tree[:,i], metric='euclidean')

                clus_cal_score[i] = calinski_harabasz_score(X=featMat, labels=cut_tree[:,i])

                clus_db_score[i] = davies_bouldin_score(X=featMat, labels=cut_tree[:,i])

            print(i/np.shape(cut_tree)[1])

        # store in self
        self.clus_featMat = featMat
        self.clus_feat_list = feat_list
        self.clus_featLbls = feat_list
        self.clus_model = clustering
        self.clus_Z = clus_Z
        self.clus_cut_tree = cut_tree
        self.clus_sil_score = clus_sil_score
        self.clus_cal_score = clus_cal_score
        self.clus_db_score = clus_db_score
    def clus_getMasterRetIdx(self, cut_level, clus_id):
        # This function generates a boolean matched to uElbl_master indicated electrodes assigned to the particular clust_id at the cut level

        # get clus_idx
        clus_idx = self.clus_cut_tree[:,cut_level] == clus_id 

        # get uElbls
        clus_uElbls = list(self.uElbl_list[clus_idx])

        # initialize master_re
        master_ret_idx = np.zeros(len(self.uElbl_list_master)).astype('bool')

        # loop through and fill in electrodes that were assigned to this cluster
        for i in clus_uElbls:

            master_ret_idx[self.uElbl_list_master.index(i)] = True

        return master_ret_idx

    def clus_getMasterRetIdxMat(self, cut_level):
        # This function generates a boolean matched to uElbl_master for each cluster id for a given cluster level

        clus_id_list = np.unique(self.clus_cut_tree[:,cut_level])
        master_ret_idx_mat = np.zeros((len(self.uElbl_list_master),len(clus_id_list)))

        for c in range(0,len(clus_id_list)):
            master_ret_idx_mat[:,c] = self.clus_getMasterRetIdx(cut_level = cut_level, clus_id = clus_id_list[c])
        return master_ret_idx_mat



    def clus_getMasterRetIdx_from_list(self, cut_level, clus_id_list,exclude_flag=True):
        # This function generates a boolean matched to uElbl_master indicated electrodes assigned to one of several clus_id (clus_id_list) at the cut level. If exclude_flag is true, it will return electrodes that were not assigned to any of these group, otherwise, it will return electrodes that were assigned to one of these groups. 

        # init 2d array to collect the bools for each clus id
        master_ret_mat = np.zeros((len(self.uElbl_list_master),len(clus_id_list)))
        master_ret_mat[:] = np.nan


        # loop through clus_id_list and get master ret_idxt
        for i in np.arange(0,len(clus_id_list)):

            master_ret_mat[:,i] = self.clus_getMasterRetIdx(cut_level = cut_level, clus_id = clus_id_list[i])

        # parse exclude flag
        if exclude_flag == True:
            # find electrodes that were not assigned to these groups
            master_ret_idx = (np.sum(master_ret_mat,axis=1) == 0).astype('bool')        
            # generate string to label this set of electrodes
            ret_str = 'exclude-'+str(cut_level)+'-'+str(clus_id_list)

        else:
            # find electrodes that were assigned to one of these groups (that are mutually exclusive)
            master_ret_idx = (np.sum(master_ret_mat,axis=1) == 1).astype('bool')

            # generate string to label this set of electrodes
            ret_str = 'include-'+str(cut_level)+'-'+str(clus_id_list)

        return master_ret_idx, ret_str


    def clus_plotDendrogram(self,ax= None,cut_level = 10,figsize=(5,5),orientation='left'):
        # Here we plot the dendrogram for the computed cluster model
        # Plot the corresponding dendrogram
        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        cluster.hierarchy.dendrogram(self.clus_Z, truncate_mode='lastp',p =cut_level,show_leaf_counts =True, ax = ax,color_threshold =0,orientation=orientation)
        if orientation in ['left','right']:
            ax.set_xlabel('Distance (a.u.)')
        else:
            ax.set_ylabel('Distance (a.u.)')


    def clus_plotMetric(self,ax= None,metric_lbl = 'sil'):
        # Here we plot the dendrogram for the computed cluster model
        # Plot the corresponding dendrogram
        if ax is None:
            f = plt.figure()
            ax = plt.subplot(111)

        plt.plot(getattr(self,'clus_'+metric_lbl+'_score'),linewidth=3,color = '0.5')
        ax.set_xlabel('Number of levels')
        ax.set_ylabel('Cluster metric (a.u.)')
        ax.set_title(metric_lbl)

    def clus_getAdjMat(self,ret_idx=None,cut_level=37):

        # This function returns an adjacency matrix for a given set of electrodes as defined by ret_idx (bool). The adjancency matrix is binarized such that electrodes assigned to the same cluster are assigned a weight of 1, otherwise they have a weight of 0. cut_level indicates the cluster level to use when assigning cluster labels

        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')


        # calculate distance matrix (and threshold based on perfect similarity)
        D = pairwise_distances(self.clus_cut_tree[ret_idx,cut_level][:,np.newaxis])

        # f = plt.figure()
        # plt.imshow(D,cmap='plasma')  
        # plt.colorbar()


        # convert disimilartiy matrix to adjacency matrix (A) (distance of 0 becomes edge of 1)
        A = np.zeros(np.shape(D))
        A[D==0] = 1

        # f = plt.figure()
        # plt.imshow(A,cmap='plasma')  
        # plt.colorbar()

        return A

    def evalClusLevel_selectivityCounts(self,cut_level = 37, sel_feat = 'ptCorr_postTarg', ax = None, p_thresh = 0.05, beh_feat = 'zrrt', min_subj_thresh = 5,min_elec_thresh=50,fsize_tick=12,yL=None,figsize=(7,5)):
        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        xtick_lbls  = []
        n_obs = []
        n_exp = []
        n_tot = []
        for i in np.arange(0,cut_level+1):

            # ret_idx
            ret_idx = self.clus_cut_tree[:,cut_level]==i

            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))
        
            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):
                continue
            # update xtick_lbl (w clus id)
            xtick_lbls.append(str(cut_level)+'-'+str(i))

            # This is an extra step to exclude subjects who have low error rates (for 'tooFastSel','ptCorr' and 'errSel')
            if sel_feat in ['tooFastSel','ptCorr_S0f','ptCorr_postTarg','errSel']:
                # exclude data from subjects with too few errors
                min_tooFast_errors = 10

                # get behavioral data for all subjects
                # check if we have computed beh_df
                if hasattr(self,'behXSubj_df') == False:
                    #run behXSubj
                    self.getBehXSubj()

                self.behXSubj_df['n_tooFast_errors']

                subj_to_include = np.array(self.behXSubj_df.query('n_tooFast_errors<@min_tooFast_errors').index)  

                #subject retain list 
                subj_ret_idx = np.isin(np.array(self.subj_list),subj_to_include)

                # update ret_idx to only include electrodes from these subjects
                ret_idx = ret_idx & subj_ret_idx

            # if we no longer have any electrodes, store nans and skip
            # min n_elecs
            if np.sum(ret_idx)==0:
                # store nans, and dont plot anything
                n_tot.append(np.nan)
                n_obs.append(np.nan)
                n_exp.append(np.nan)

                continue


            if sel_feat == 'tooFastSel':
                # calculate z-statistic
                x = self.taskstats2d_df['tooFastSel_tstat'].to_numpy()[ret_idx]
                pvals = self.taskstats2d_df['tooFastSel_pval'].to_numpy()[ret_idx]

            elif sel_feat == 'ptCorr_S0f':
                # pre-stimulus correlation of prediction times (expecation representation/integration)

                # calculate z-statistic
                x = self.taskstats2d_df['ptCorr_S0f_rval'].to_numpy()[ret_idx]
                pvals = self.taskstats2d_df['ptCorr_S0f_pval'].to_numpy()[ret_idx]
            elif sel_feat == 'ptCorr_postTarg':
                # pre-stimulus correlation of prediction times (expecation representation/integration)

                # calculate z-statistic
                x = self.taskstats2d_df['ptCorr_postTarg_rval'].to_numpy()[ret_idx]
                pvals = self.taskstats2d_df['ptCorr_postTarg_pval'].to_numpy()[ret_idx]
            elif sel_feat == 'rewSel':
                # change in post responseactivity during postitive feedback (RT<300)

                # calculate z-statistic
                x = self.taskstats2d_df['rewSel_tstat'].to_numpy()[ret_idx]
                pvals = self.taskstats2d_df['rewSel_pval'].to_numpy()[ret_idx]
            elif sel_feat == 'errSel':
                # change in post response activity on error trials (prediction error)

                # calculate z-statistic
                x = self.taskstats2d_df['errSel_tstat'].to_numpy()[ret_idx]
                pvals = self.taskstats2d_df['errSel_pval'].to_numpy()[ret_idx]
            elif sel_feat == 'spatialSel':
                # one way anova of post-target activity related to locations
                # there is no direction of effect here because its a one way anova, so we count all effects as poistive
                x = np.ones(np.sum(ret_idx))
                pvals = self.taskstats2d_df['tlAnova_pval'].to_numpy()[ret_idx]                                

            counts_pos = (x>0)&(pvals<=p_thresh)
            counts_neg = (x<0)&(pvals<=p_thresh)

            # store data
            n_tot.append(np.sum(ret_idx))
            n_obs.append(np.sum(counts_pos+counts_neg))
            n_exp.append(np.sum(ret_idx)*p_thresh)


            ax.bar(len(xtick_lbls),(np.count_nonzero(counts_pos)/len(counts_pos))*100,color='tab:blue',
            edgecolor='k',label = '(+) effects')
            ax.bar(len(xtick_lbls),(np.count_nonzero(counts_neg)/len(counts_pos))*100,
            bottom = (np.count_nonzero(counts_pos)/len(counts_pos))*100, color='tab:orange',
            edgecolor='k',label = '(-) effects')

            if i == 0:
                plt.legend(fontsize=fsize_tick)

        plt.xticks(ticks=(np.arange(0,len(xtick_lbls)))+1,
        labels = xtick_lbls,rotation=90,fontsize=fsize_tick)
        plt.ylabel('% electrodes')

        if yL != None:
            ax.set_ylim(yL)
        plt.hlines(p_thresh*100,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='--',color = 'k',alpha = 0.5)

        plt.tight_layout()

        # store counts data in dict
        plot_dict = {}
        plot_dict['n_tot'] = n_tot
        plot_dict['n_obs'] = n_obs
        plot_dict['n_exp'] = n_exp
        plot_dict['freq'] = np.array(n_obs)/np.array(n_tot)
        plot_dict['lbls'] = xtick_lbls

        #replace nans w 0 (for plotting purposes)
        plot_dict['freq'][np.isnan(n_obs)] = 0

        return plot_dict

    def clus_getEligibleClusters(self,cut_level=21,min_subj_thresh=5,min_elec_thresh=50):
        # returns list of int that indicate the clusters that meet criteria for a given cluster level

        # container
        clus_list = []

        for i in np.arange(0,cut_level+1):
            # ret_idx
            ret_idx = self.clus_cut_tree[:,cut_level]==i

            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))
        
            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):
                continue
            clus_list.append(i)
        return clus_list

    def evalClusLevel_rtCorr(self,cut_level = 37, ax = None,neu_feat='S0c', use_clean = False, collapseBySubj_flag = False, beh_feat = 'zrrt', min_subj_thresh = 5,min_elec_thresh=50,fsize_tick=12,yL=None,figsize=(7,5),delay_str = ''):
        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        if use_clean==True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'

        if delay_str == '':
            delay_lbl = 'all trials'
        elif delay_str == 'S':
            delay_lbl = 'short trials'
        elif delay_str == 'L':
            delay_lbl = 'long trials'

        xtick_lbls = []



        for i in np.arange(0,cut_level+1):

            # ret_idx
            ret_idx = self.clus_cut_tree[:,cut_level]==i

            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))
        
            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):
                continue

            # update xtick_lbl (w clus id)
            xtick_lbls.append(str(cut_level)+'-'+str(i))

            # calculate z-statistic
            x = self.taskstats2d_df['rtCorr'+delay_str+'_'+beh_feat+'_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]

            # collapse by subj (optional)
            if collapseBySubj_flag == True:
                x = self.collapseBySubj_1d(x,subj_list_ret_idx=ret_idx)

            width=0.8
            ax.bar(len(xtick_lbls),np.nanmean(x),color='.7',edgecolor='k',width = width)
            ax.errorbar(len(xtick_lbls),np.nanmean(x),(1.96*stats.sem(x,nan_policy='omit')),ecolor='k',elinewidth=width)

        plt.xticks(ticks=(np.arange(0,len(xtick_lbls)))+1,
        labels = xtick_lbls,rotation=90,fontsize=fsize_tick)
        plt.ylabel('t-stat')

        if yL != None:
            ax.set_ylim(yL)

        plt.title('err = 95"%"" ci...n electrodes = '+str(len(self.uElbl_list))+' n subj = '+str(len(np.unique(self.subj_list)))+'\n Collapse by subj = '+str(collapseBySubj_flag)+'; '+delay_lbl+'; '+clean_lbl)
        plt.tight_layout()
    def evalClusLevel_rtRegressCounts_SSE(self,cut_level = 37, ax = None,beh_feat = 'zrrt', min_subj_thresh = 5,min_elec_thresh = 50,fsize_tick=12,yL=None,p_thresh=0.05,figsize=(7,5)):
        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        xtick_lbls = []


        # 
        n_obs = []
        n_exp = []
        n_tot = []

        for i in np.arange(0,cut_level+1):

            # ret_idx
            ret_idx = self.clus_cut_tree[:,cut_level]==i



            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))
        
            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):
                continue

            # update xtick_lbl (w clus id)
            xtick_lbls.append(str(cut_level)+'-'+str(i))

            # get p values (only non parametric available)
            pvals = self.taskstats2d_df['rtRegress_multivar_'+beh_feat+'_SSE'+'_pvalnp'].to_numpy()[ret_idx]

            counts_sig = pvals<=p_thresh


            # store data
            n_tot.append(np.sum(ret_idx))
            n_obs.append(np.sum(counts_sig))
            n_exp.append(np.sum(ret_idx)*p_thresh)

            ax.bar(len(xtick_lbls),(np.count_nonzero(counts_sig)/len(counts_sig))*100,color='0.5',
            edgecolor='k',label = 'sig. effects')
            ax.text(len(xtick_lbls)-.15,2,str(int(n_obs[-1]))+'/'+str(int(n_tot[-1])),rotation=90,fontsize=fsize_tick)

        plt.xticks(ticks=(np.arange(0,len(xtick_lbls)))+1,
        labels = xtick_lbls,rotation=90,fontsize=fsize_tick)
        plt.ylabel('% electrodes')
        plt.gca().set_xlim(0,len(xtick_lbls)+1)

        if yL != None:
            ax.set_ylim(yL)
        plt.hlines(p_thresh*100,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='--',color = 'k',alpha = 0.5)




        plt.title('n electrodes = '+str(len(self.uElbl_list))+' n subj = '+str(len(np.unique(self.subj_list))))
        plt.tight_layout()


        # store counts data in dict
        plot_dict = {}
        plot_dict['n_tot'] = n_tot
        plot_dict['n_obs'] = n_obs
        plot_dict['n_exp'] = n_exp
        plot_dict['freq'] = np.array(n_obs)/np.array(n_tot)
        plot_dict['lbls'] = xtick_lbls

        #
        return plot_dict
    def evalClusLevel_activationAmplitude(self,cut_level = 37, ax = None,  min_subj_thresh = 5,min_elec_thresh = 50,fsize_tick=12,yL=None,p_thresh=0.05,figsize=(7,5),neu_feat = 'postNoCCS',clus_range=None):
        # This function plots the fraction of electrodes that showed some deviation in average activation function during the task.
        # neu_feat = 'postNoCCS' .. during long delay trials from 500 ms - 1500ms, as a proxy for expectation related changes
        # neu_feat = 'preResponseS_respLocked' ... prior to response (stimulus or response locked)
        # neu_feat = 'postTargOn' ... following target


        # parse clus_range
        if clus_range is None:
            clus_range = [0,cut_level+1]

        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        xtick_lbls = []


        # 
        n_obs = []
        n_exp = []
        n_tot = []

        for i in np.arange(clus_range[0],clus_range[-1]):

            # ret_idx
            ret_idx = self.clus_cut_tree[:,cut_level]==i

            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))
        
            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):
                continue

            # update xtick_lbl (w clus id)
            xtick_lbls.append(str(cut_level)+'-'+str(i))

            # get num sig (only non parametric available)
            counts_sig = (np.abs(self.taskstats2d_df['modParams_'+neu_feat+'_amp'].to_numpy())[ret_idx])>0


            # store data
            n_tot.append(np.sum(ret_idx))
            n_obs.append(np.sum(counts_sig))
            n_exp.append(np.sum(ret_idx)*p_thresh)



            ax.bar(len(xtick_lbls),100*(n_obs[-1]/n_tot[-1]),color='0.5',
            edgecolor='k',label = 'sig. effects')
            ax.text(len(xtick_lbls)-.15,2,str(int(n_obs[-1]))+'/'+str(int(n_tot[-1])),rotation=90,fontsize=fsize_tick)

        plt.xticks(ticks=(np.arange(0,len(xtick_lbls)))+1,
        labels = xtick_lbls,rotation=90,fontsize=fsize_tick)
        plt.ylabel('% electrodes')

        if yL != None:
            ax.set_ylim(yL)
        plt.hlines(p_thresh*100,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='--',color = 'k',alpha = 0.5)


        plt.title('n electrodes = '+str(len(self.uElbl_list))+' n subj = '+str(len(np.unique(self.subj_list))))
        plt.tight_layout()


        # store counts data in dict
        plot_dict = {}
        plot_dict['n_tot'] = n_tot
        plot_dict['n_obs'] = n_obs
        plot_dict['n_exp'] = n_exp
        plot_dict['freq'] = np.array(n_obs)/np.array(n_tot)
        plot_dict['lbls'] = xtick_lbls

        #
        return plot_dict
    def evalClusLevel_activationTiming(self,cut_level = 37, ax = None,  min_subj_thresh = 5,min_elec_thresh = 50,fsize_tick=12,yL=None,p_thresh=0.05,figsize=(7,5),neu_feat = 'preResponseL_respLocked'):
        # This function plots the fraction of electrodes that showed some deviation in average activation function during the task.
        # neu_feat = 'postNoCCS' .. during long delay trials from 500 ms - 1500ms, as a proxy for expectation related changes
        # neu_feat = 'preResponseS_respLocked' ... prior to response (stimulus or response locked)
        # neu_feat = 'postTargOn' ... following target

        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        xtick_lbls = []


        # 
        mean_onset = []
        mean_width = []
        n_tot = []

        for i in np.arange(0,cut_level+1):

            # ret_idx
            ret_idx = self.clus_cut_tree[:,cut_level]==i

            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))
        
            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):
                continue

            # update xtick_lbl (w clus id)
            xtick_lbls.append(str(cut_level)+'-'+str(i))

            # get num sig (only non parametric available)
            onset_ms = (np.abs(self.taskstats2d_df['modParams_'+neu_feat+'_cen_ms'].to_numpy())[ret_idx])
            wid_ms = (np.abs(self.taskstats2d_df['modParams_'+neu_feat+'_wid_ms'].to_numpy())[ret_idx])

            # store data
            n_tot.append(np.sum(ret_idx))
            mean_onset.append(np.nanmean(onset_ms))
            mean_width.append(np.nanmean(wid_ms))




            ax.bar(len(xtick_lbls),np.nanmean(mean_onset),color='0.5',
            edgecolor='k',label = 'sig. effects')


        plt.xticks(ticks=(np.arange(0,len(xtick_lbls)))+1,
        labels = xtick_lbls,rotation=90,fontsize=fsize_tick)
        plt.ylabel('% electrodes')

        if yL != None:
            ax.set_ylim(yL)
        plt.hlines(p_thresh*100,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='--',color = 'k',alpha = 0.5)


        plt.title('n electrodes = '+str(len(self.uElbl_list))+' n subj = '+str(len(np.unique(self.subj_list))))
        plt.tight_layout()


        # store counts data in dict
        plot_dict = {}
        plot_dict['n_tot'] = n_tot
        plot_dict['onset_ms'] = np.array(mean_onset)
        plot_dict['wid_ms'] = np.array(mean_width)
        plot_dict['lbls'] = xtick_lbls

        #
        return plot_dict

    def evalClusLevel_rtCorrCounts(self,cut_level = 14, ax = None,neu_feat='S0c', use_clean = False, beh_feat = 'zrrt', min_subj_thresh = 5,min_elec_thresh = 50,fsize_tick=12,yL=None,p_thresh=0.05,figsize=(7,5),delay_str = ''):
        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        if use_clean==True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'

        if delay_str == '':
            delay_lbl = 'all trials'
        elif delay_str == 'S':
            delay_lbl = 'short trials'
        elif delay_str == 'L':
            delay_lbl = 'long trials'

        xtick_lbls = []
        n_tot = [] 
        n_obs = []
        n_pos = []
        n_neg = []
        n_exp = []

        for i in np.arange(0,cut_level+1):

            # ret_idx
            ret_idx = self.clus_cut_tree[:,cut_level]==i

            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))
        
            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):
                continue

            # update xtick_lbl (w clus id)
            xtick_lbls.append(str(cut_level)+'-'+str(i))

            # calculate z-statistic
            x = self.taskstats2d_df['rtCorr'+delay_str+'_'+beh_feat+'_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]
            pvals = self.taskstats2d_df['rtCorr'+delay_str+'_'+beh_feat+'_'+neu_feat+clean_str+'_pvalnp'].to_numpy()[ret_idx]

            counts_pos = (x>0)&(pvals<=p_thresh)
            counts_neg = (x<0)&(pvals<=p_thresh)
            
            # store data
            n_tot.append(np.sum(ret_idx))
            n_obs.append(np.sum(counts_pos+counts_neg))
            n_exp.append(np.sum(ret_idx)*p_thresh)
            n_pos.append(np.sum(counts_pos))
            n_neg.append(np.sum(counts_neg))

            ax.bar(len(xtick_lbls),(np.count_nonzero(counts_pos)/len(counts_pos))*100,color='tab:blue',
            edgecolor='k',label = '(+) effects')
            ax.bar(len(xtick_lbls),(np.count_nonzero(counts_neg)/len(counts_pos))*100,
            bottom = (np.count_nonzero(counts_pos)/len(counts_pos))*100, color='tab:orange',
            edgecolor='k',label = '(-) effects')

            if i == 0:
                plt.legend(fontsize=fsize_tick)

        plt.xticks(ticks=(np.arange(0,len(xtick_lbls)))+1,
        labels = xtick_lbls,rotation=90,fontsize=fsize_tick)
        plt.ylabel('% electrodes')

        if yL != None:
            ax.set_ylim(yL)
        plt.hlines(p_thresh*100,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='--',color = 'k',alpha = 0.5)


        plt.title('n electrodes = '+str(len(self.uElbl_list))+' n subj = '+str(len(np.unique(self.subj_list)))+'; '+delay_lbl+'; '+clean_lbl)
        plt.tight_layout()

        
        # store counts data in dict
        plot_dict = {}
        plot_dict['n_tot'] = n_tot
        plot_dict['n_obs'] = n_obs
        plot_dict['n_exp'] = n_exp
        plot_dict['n_pos'] = n_pos
        plot_dict['n_neg'] = n_neg
        plot_dict['freq'] = np.array(n_obs)/np.array(n_tot)
        plot_dict['freq_pos'] = np.array(n_pos)/np.array(n_tot)
        plot_dict['freq_neg'] = np.array(n_neg)/np.array(n_tot)
        plot_dict['freq_PosNegDiff'] = plot_dict['freq_pos'] - plot_dict['freq_neg']  # use this to illustrate bias towards positive or negative effects using a diverging colormap 
        plot_dict['lbls'] = xtick_lbls

        #
        return plot_dict
    def evalClusLevel_delayDiffCounts(self,cut_level = 37, ax = None,neu_feat='S0c', use_clean = False,  min_subj_thresh = 5,min_elec_thresh = 50,fsize_tick=12,yL=None,p_thresh=0.05,figsize=(7,5)):
        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        if use_clean==True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'


        xtick_lbls = []

        n_tot = []
        n_obs = []
        n_exp = []
        n_pos = []
        n_neg = [] 

        for i in np.arange(0,cut_level+1):

            # ret_idx
            ret_idx = self.clus_cut_tree[:,cut_level]==i

            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))
        
            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):
                continue

            # update xtick_lbl (w clus id)
            xtick_lbls.append(str(cut_level)+'-'+str(i))



            # calculate z-statistic
            x = self.taskstats2d_df['delayDiff_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]
            pvals = self.taskstats2d_df['delayDiff_'+neu_feat+clean_str+'_pvalnp'].to_numpy()[ret_idx]

            counts_pos = (x>0)&(pvals<=p_thresh)
            counts_neg = (x<0)&(pvals<=p_thresh)

            # store data
            n_tot.append(np.sum(ret_idx))
            n_obs.append(np.sum(counts_pos+counts_neg))
            n_exp.append(np.sum(ret_idx)*p_thresh)
            n_pos.append(np.sum(counts_pos))
            n_neg.append(np.sum(counts_neg))    

            ax.bar(len(xtick_lbls),(np.count_nonzero(counts_pos)/len(counts_pos))*100,color='tab:blue',
            edgecolor='k',label = '(+) effects')
            ax.bar(len(xtick_lbls),(np.count_nonzero(counts_neg)/len(counts_pos))*100,
            bottom = (np.count_nonzero(counts_pos)/len(counts_pos))*100, color='tab:orange',
            edgecolor='k',label = '(-) effects')

            if i == 0:
                plt.legend(fontsize=fsize_tick)

        plt.xticks(ticks=(np.arange(0,len(xtick_lbls)))+1,
        labels = xtick_lbls,rotation=90,fontsize=fsize_tick)
        plt.ylabel('% electrodes')

        if yL != None:
            ax.set_ylim(yL)
        plt.hlines(p_thresh*100,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='--',color = 'k',alpha = 0.5)


        plt.title('n electrodes = '+str(len(self.uElbl_list))+' n subj = '+str(len(np.unique(self.subj_list)))+'; '+'; '+clean_lbl)
        plt.tight_layout()


        # store counts data in dict
        plot_dict = {}
        plot_dict['n_tot'] = n_tot
        plot_dict['n_obs'] = n_obs
        plot_dict['n_exp'] = n_exp
        plot_dict['n_pos'] = n_pos
        plot_dict['n_neg'] = n_neg
        plot_dict['freq'] = np.array(n_obs)/np.array(n_tot)
        plot_dict['freq_pos'] = np.array(n_pos)/np.array(n_tot)
        plot_dict['freq_neg'] = np.array(n_neg)/np.array(n_tot)
        plot_dict['freq_PosNegDiff'] = plot_dict['freq_pos'] - plot_dict['freq_neg'] # use this to illustrate bias towards positive or negative effects using a diverging colormap
        plot_dict['lbls'] = xtick_lbls

        return plot_dict


    def evalClusLevel_xSubjCorr_rtCorr(self,cut_level = 37, ax = None,beh_xSubj = 'rtDiff_mean', neu_feat='S0c', use_clean = False, beh_feat = 'zrrt', min_subj_thresh = 5,min_elec_thresh = 50,fsize_tick=12,fsize_lbl=12,yL=None,p_thresh=0.05,figsize=(7,5),delay_str = '', compare_delay = False,subj_filt_query = ''):
        # for each cluster for a given level, we use a bar plot to illustrate the spearman correlation (across subjects between a neural feature relation to RT and a behavioral measure (e.g., rtDiff_mean))
        
        #Same desciption as plotXSubjCorr, but it loops through several clusters summarizes the correlation via bar plot. 

        # It first obtain a neural feature (zstatistic) for a group of electrodes specify (e.g., from a cluster or region). Then it collapses that feature within subject such that there is one measure per subject. It also returns a subject list

        # Then it uses that subject list to obtain a measurement of behavioral performance. 

        # To do this efficiently, it runs self.getBehXSubj() that will loop through all subjects and assign a df to self (beh_df). Once that is created, it can query it as needed. If it hasnt been created, it will compute these data

        # then it plots a scatter between neural and behavioral variability using spearman correlation


        # Note: 'beh_feat' indicates the behavioral feature that is used to compute relation between neural activity and behavior within subject (e.g., zrrt). 'beh_xSubj' indicates the behavioral feature to compute across subject variability (e.g., 'rtDiff_mean')

        # subj_filt_query applies additional subject-level filter for behavioral criteria (e.g., error_rate < 0.25)




        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        # error check
        if (compare_delay==True)&((delay_str=='')==False):
            raise NameError('compare_delay is true, delay_str must be "''"" ') 

        if use_clean==True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'

        if delay_str == '':
            delay_lbl = 'all trials'
        elif delay_str == 'S':
            delay_lbl = 'short trials'
        elif delay_str == 'L':
            delay_lbl = 'long trials'

        xtick_lbls = []
        rval_list = []
        pval_list = []
        stat_lbls = []

        for i in np.arange(0,cut_level+1):

            # ret_idx
            ret_idx = self.clus_cut_tree[:,cut_level]==i

            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))
        
            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):
                continue

            # update xtick_lbl (w clus id)
            xtick_lbls.append(str(cut_level)+'-'+str(i))

            # update stat_lbls (clus id + neu_feat + behxSubj)
            stat_lbls.append(str(cut_level)+'-'+str(i)+'-'+neu_feat+'-'+beh_xSubj+'-RTCORR')

            # check if we have computed beh_df
            if hasattr(self,'behXSubj_df') == False:
                # compute acr
                self.getBehXSubj()

            ### apply additional subject-level query (option)
            if (subj_filt_query == '')==False:
                subj_to_include = np.array(self.behXSubj_df.query(subj_filt_query).index)  

                #subject retain list 
                subj_ret_idx = np.isin(np.array(self.subj_list),subj_to_include)

                # update ret_idx to only include electrodes from these subjects
                ret_idx = ret_idx & subj_ret_idx

            #### get neural feature
            if compare_delay == False:
                neu = self.taskstats2d_df['rtCorr'+delay_str+'_'+beh_feat+'_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]
            else:
                neuL = self.taskstats2d_df['rtCorrL'+'_'+beh_feat+'_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]
                neuS = self.taskstats2d_df['rtCorrS'+'_'+beh_feat+'_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]
                neu = neuL-neuS

            # collapse by subj
            neu_subj = self.collapseBySubj_1d(neu,subj_list_ret_idx=ret_idx)

            # get subj_list 
            subj_ret_idx = self.collapseBySubj_1d(ret_idx)

            ### get beahavioral feature
            # query our specific behavioral feature (for this list of subjects)
            beh_subj = self.behXSubj_df.iloc[np.nonzero(subj_ret_idx)[0]][beh_xSubj].to_numpy()

            # compute spearman corr
            rval,pval = stats.spearmanr(neu_subj,beh_subj)

            # plot bar
            plt.bar(len(xtick_lbls),rval,color='0.5',edgecolor='k')

            # write pvalue
            if pval <0.05:
                plt.text(len(xtick_lbls),rval,'p = '+str(np.round(pval,3)),fontsize=fsize_tick)

            rval_list.append(rval)
            pval_list.append(pval)

        plt.xticks(ticks=(np.arange(0,len(xtick_lbls))+1),
        labels = xtick_lbls,rotation=90,fontsize=fsize_tick)
        plt.ylabel('Spearman $r$',fontsize=fsize_lbl)

        ax.set_ylim(-1,1)
        ax.set_xlim(-0.5,len(xtick_lbls)+0.5)
        plt.title(neu_feat+' vs '+beh_xSubj+' compare delay = '+str(compare_delay)+'; '+delay_lbl+'; '+clean_lbl)
        plt.tight_layout()

        # store corr data in dict
        plot_dict = {}
        plot_dict['rval'] = np.array(rval_list)
        plot_dict['pvals'] = np.array(pval_list)
        plot_dict['lbls'] = xtick_lbls
        plot_dict['stat_lbls'] = stat_lbls
        #
        return plot_dict


    def evalClusLevel_xSubjCorr_delayDiff(self,cut_level = 37, ax = None,beh_xSubj = 'rtDiff_mean', neu_feat='S0c', use_clean = False, min_subj_thresh = 5,min_elec_thresh = 50,fsize_tick=12,fsize_lbl=12,yL=None,p_thresh=0.05,figsize=(7,5),subj_filt_query = '',do_ttest=False,subj_grp_query= ''):
        # for each cluster for a given level, we use a bar plot to illustrate the spearman correlation (across subjects between dealy-related difference in neural feature and delay-related difference in behavior neural feature and a behavioral measure (e.g., rtDiff_mean))
        
        #Same desciption as plotXSubjCorr, but it loops through several clusters summarizes the correlation via bar plot. 

        # It first obtain a neural feature (zstatistic) for a group of electrodes specify (e.g., from a cluster or region). Then it collapses that feature within subject such that there is one measure per subject. It also returns a subject list

        # Then it uses that subject list to obtain a measurement of behavioral performance. 

        # To do this efficiently, it runs self.getBehXSubj() that will loop through all subjects and assign a df to self (beh_df). Once that is created, it can query it as needed. If it hasnt been created, it will compute these data

        # then it plots a scatter between neural and behavioral variability using spearman correlation


        # Note: 'beh_feat' indicates the behavioral feature that is used to compute relation between neural activity and behavior within subject (e.g., zrrt). 'beh_xSubj' indicates the behavioral feature to compute across subject variability (e.g., 'rtDiff_mean')

        # subj_filt_query applies an additional subject level filter based on behavioral criteria

        # do_ttest = True; bool. If set to True, it will compute t-stats instead of spearman correlations and plot t-stats in the bar plot.

        # subj_grp_query; if = '', it will compute a one-sample t-test aggregating delay-related differences across subjects. If a boolean is provided, it will compare groups do_ttest is set to True, must provide a boolean assigning electrodes to groups (subj_grp_idx=0 vs.subj_grp_idx = 1).

        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        if use_clean==True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'

        xtick_lbls = []
        rval_list = []
        pval_list = []
        tstat_list = []
        pval_t_list = []
        stat_lbls = []

        for i in np.arange(0,cut_level+1):

            # ret_idx
            ret_idx = self.clus_cut_tree[:,cut_level]==i

            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))
        
            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):
                continue

            # update xtick_lbl (w clus id)
            xtick_lbls.append(str(cut_level)+'-'+str(i))

            # update stat_lbls (clus id + neu_feat + behxSubj)
            stat_lbls.append(str(cut_level)+'-'+str(i)+'-'+neu_feat+'-'+beh_xSubj+'-DelayDiff')

            # check if we have computed beh_df
            if hasattr(self,'behXSubj_df') == False:
                # compute acr
                self.getBehXSubj()

            ### apply additional subject-level query (option)
            if (subj_filt_query == '')==False:
                subj_to_include = np.array(self.behXSubj_df.query(subj_filt_query).index)  

                #subject retain list 
                subj_ret_idx = np.isin(np.array(self.subj_list),subj_to_include)

                # update ret_idx to only include electrodes from these subjects
                ret_idx = ret_idx & subj_ret_idx


            #### get neural feature
            neu = self.taskstats2d_df['delayDiff_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[ret_idx]
            
            # collapse by subj
            neu_subj = self.collapseBySubj_1d(neu,subj_list_ret_idx=ret_idx)

            # get subj_list 
            subj_ret_idx = self.collapseBySubj_1d(ret_idx)

            ### get beahavioral feature
            # query our specific behavioral feature (for this list of subjects)
            beh_subj = self.behXSubj_df.iloc[np.nonzero(subj_ret_idx)[0]][beh_xSubj].to_numpy()

            # compute spearman corr
            rval,pval = stats.spearmanr(neu_subj,beh_subj)

            # update stat containers
            rval_list.append(rval)
            pval_list.append(pval)


            # parse t-test () 
            if do_ttest == True:
                # parse subj_grp_idx to see whether we are doing a one-sample t-test or comparing two groups
                if subj_grp_query == '':
                    # do one sample ttest
                    tstat,pval_t = stats.ttest_1samp(neu_subj,nan_policy='omit')
                else:
                    # do independent sample ttest between two groups of subjects

                    # we need to recompute neu_subj for each group of electrodes
                    subj_in_grpA= np.array(self.behXSubj_df.query(subj_grp_query).index)  

                    #subject retain list 
                    subj_grpA_idx = np.isin(np.array(self.subj_list),subj_in_grpA)

                    # define booleans for each subject group (across electrodes)
                    subj_grpA_ret_idx = ret_idx & subj_grpA_idx
                    subj_grpB_ret_idx = ret_idx & (subj_grpA_idx==False)

                    #### get neural feature
                    neu_A = self.taskstats2d_df['delayDiff_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[subj_grpA_ret_idx]
                    neu_B = self.taskstats2d_df['delayDiff_'+neu_feat+clean_str+'_zstatnp'].to_numpy()[subj_grpB_ret_idx]

                    # collapse within subject
                    neu_subj_A = self.collapseBySubj_1d(neu_A,subj_list_ret_idx=subj_grpA_ret_idx)
                    neu_subj_B = self.collapseBySubj_1d(neu_B,subj_list_ret_idx=subj_grpB_ret_idx)              

                    tstat,pval_t = stats.ttest_ind(neu_subj_A,neu_subj_B,equal_var = False,nan_policy='omit')


                # update containers 
                tstat_list.append(tstat)
                pval_t_list.append(pval_t)

                # plot bar
                plt.bar(len(xtick_lbls),tstat,color='0.5',edgecolor='k')

                # write pvalue
                if pval_t <0.05:
                    plt.text(len(xtick_lbls),tstat,'p = '+str(np.round(pval_t,3)),fontsize=fsize_tick)
                plt.ylabel('$t$ statistic',fontsize=fsize_lbl)
                ax.set_ylim(-3,3)


            else:
                # plot spearman correlations in bar plot
                plt.bar(len(xtick_lbls),rval,color='0.5',edgecolor='k')

                # write pvalue
                if pval <0.05:
                    plt.text(len(xtick_lbls),rval,'p = '+str(np.round(pval,3)),fontsize=fsize_tick)
                plt.ylabel('Spearman $r$',fontsize=fsize_lbl)
                ax.set_ylim(-1,1)





        plt.xticks(ticks=(np.arange(0,len(xtick_lbls))+1),
        labels = xtick_lbls,rotation=90,fontsize=fsize_tick)
        ax.set_xlim(-0.5,len(xtick_lbls)+0.5)
        plt.title(neu_feat+' vs '+beh_xSubj+clean_lbl)
        plt.tight_layout()

        # store corr data in dict
        plot_dict = {}
        plot_dict['rval'] = np.array(rval_list)
        plot_dict['pvals'] = np.array(pval_list)
        plot_dict['tstat'] = np.array(tstat_list)
        plot_dict['pvals_t'] = np.array(pval_t_list)
        plot_dict['lbls'] = xtick_lbls
        plot_dict['stat_lbls'] = stat_lbls
        #
        return plot_dict



    def evalClusLevel_anatLocalization(self,ax = None,figsize =(7,5),cut_level = 37, title = '',yL=None,fsize_tick = 14, fsize_lbl = 20, min_subj_thresh = 5,min_elec_thresh = 50,clus_range=None):
        # bar plot of chi-square statistics (indicating how regionaly localized a cluster is) 

        # parse clus_range
        if clus_range is None:
            clus_range = [0,cut_level+1]

        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)
    
        # get list of anatomical labels
        anat_list = np.array(self.anat_list.copy())
        anat_list_wm = np.array(self.anat_list_wm.copy())
        mni_coords = np.zeros((len(anat_list),3))
        rois = [] 
    
        # drop laterality
        for a in np.arange(0,len(anat_list)):

            # parse anat_lbl
            anat_lbl = self.parse_anat_lbl(anat_list[a],anat_list_wm[a])                
            # convert to roi
            rois.append(self.anat2roi(anat_lbl))
            
            # collect mni coords
            mni_coords[a,0] = self.mni_x[a]
            mni_coords[a,1] = self.mni_y[a]
            mni_coords[a,2] = self.mni_z[a]
       
     
        # replace 'unlabelled' rois
        rois = self.parse_unabelled(rois,mni_coords)

        # get roi_list
        roi_list = np.unique(np.array(rois))


        def getChi2_local(ret_idx,rois):

            # frequency of this cluster among all electrodes
            thisClus_freq = np.count_nonzero(ret_idx)/len(ret_idx)

            # get total number of electrodes
            tot = len(rois)


            # containers for counts
            counts_numInReg = []
            counts_thisClusInReg = []

            # loop through regions
            for r in roi_list:
                
                # count num of elecs in this region (total)
                numInReg = np.count_nonzero(np.array(rois)==r)

                # count number of this group in region 
                numThisClusInReg = np.count_nonzero(np.array(rois)[ret_idx]==r)
                
                # append counts lists
                counts_numInReg.append(numInReg)
                counts_thisClusInReg.append(numThisClusInReg)

            chisq,p = stats.chisquare(f_obs=counts_thisClusInReg, \
                                      f_exp=np.array(counts_numInReg)*thisClus_freq)

            return chisq

        xtick_lbls = []
        chi2stat_list = []

        # loop through clusters
        for i in np.arange(clus_range[0],clus_range[-1]):

            # ret_idx
            ret_idx = self.clus_cut_tree[:,cut_level]==i

            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))
        
            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):
                continue

            # update xtick_lbl (w clus id)
            xtick_lbls.append(str(cut_level)+'-'+str(i))

            # calculate measure of anatomical localization
            chi2stat_list.append(getChi2_local(ret_idx,rois))

            # plot bar
            plt.bar(len(xtick_lbls),chi2stat_list[-1],color = '0.5',edgecolor='k')


        # set ticks
        if (yL is None)==False:
            ax.set_ylim(yL)
        ax.set_xlim((0,len(xtick_lbls)+1))
        ax.set_xticks(np.arange(0,len(xtick_lbls))+1)
        ax.set_xticklabels(xtick_lbls,fontsize=fsize_tick,rotation=90);
        ax.set_ylabel('$\chi^{2}$ statistic',fontsize=fsize_lbl)
        ax.set_yticklabels(ax.get_yticks().astype('int'),fontsize=fsize_tick)

        plt.tight_layout()
        

        plot_dict = {}
        plot_dict['chi2stats'] = np.array(chi2stat_list)
        plot_dict['lbls'] = xtick_lbls
        return plot_dict


    def evalClusLevel_brainPlot(self,cut_level = 37,plot_each_cluster = False,min_subj_thresh = 5,min_elec_thresh = 50,cmap='prism',marker_size=10,snap_to_surface = True,plot_on_single_surface=True,view_in_browser=True,save_fullpath='',plot_connectome=False,adj_col =None,adj_linewidth=None,use_anat_colors = True,anat_cmap = 'rainbow', clus_range = None,atlas='default'):
        # this function plots a brain plot for a clus_level, where electrodes are colored by their cluster assignment (only considers clusters that meet minimum threshold criteria).
        # see evalClus_brainPlot for input definition
        #if plot_connectome=True, it plots a connectome connecting clusters 
        # adj_col is used to color connections (if None, it uses the color used for each cluster). NOTE: if we are plotting multiple clusters on a single plot (plot_each_cluster=False), it will plot all connections gray
        # adj_linewidth... linewidth of edge weights. If None it sets the weights dynamically based on number of observations in each cluster
        # use anat_colors=True uses anat region colors (using anat_cmapin anatDf). This only applies when plotting single clusters

        # parse clus_range
        if clus_range is None:
            clus_range = [0,cut_level+1]



        # container for adj_linewidth that was given as input 
        adj_linewidth_in = adj_linewidth


        # get anat df for all electrodes held in collection object (after filtering)
        anatDf,roiList = self.getAnatDf(cmap=anat_cmap,atlas=atlas);

        # initialize container for cluster colors
        clus_colors = np.zeros((len(anatDf),4))

        # list of colors for this cut level
        from matplotlib import cm
        colmap = cm.get_cmap(cmap,100)

        clus_list = []
        clus_color_list = []
        c_idx = np.linspace(0,1,int(np.diff(clus_range)+1)) # use this as an index to specify a color for each cluster

        # get list of clusters
        for i in np.arange(clus_range[0],clus_range[-1]):

            # ret_idx
            ret_idx = self.clus_cut_tree[:,cut_level]==i

            # calc num subj
            n_subj = np.count_nonzero(self.collapseBySubj_1d(ret_idx))

            if (n_subj<min_subj_thresh)|(np.count_nonzero(ret_idx)<min_elec_thresh):

                # color these electrodes grey
                clus_colors[ret_idx,:] = np.tile(colmap(c_idx[i]),(np.sum(ret_idx),1)) 
                # color these electrodes grey
                #clus_colors[ret_idx,:] = np.tile([0.5,0.5,0.5,1],(np.sum(ret_idx),1))
                continue

            # choose a color
            # update colors using the color map object 
            clus_colors[ret_idx,:] = np.tile(colmap(c_idx[i]),(np.sum(ret_idx),1))
            
            # update xtick_lbl (w clus id)
            clus_list.append(str(cut_level)+'-'+str(i))
            clus_color_list.append(colmap(c_idx[i]))

            # option to plot this cluster only
            if plot_each_cluster==True:

                # if plot connectome is true, get adjancy matrix
                if plot_connectome==True:
                    adj_mat = self.clus_getAdjMat(ret_idx=ret_idx,cut_level=cut_level)
                else:
                    adj_mat = None
                # get adj_color

                # set linewidth adaptively
                if adj_linewidth_in is None:
                    if np.sum(ret_idx)<=100:
                        adj_linewidth=0.25
                    elif np.sum(ret_idx)<=200:
                        adj_linewidth=.25
                    elif np.sum(ret_idx)>200:
                        # if > 200, decimate the adjacency matrix (only show 50% of the connections)
                        adj_linewidth=0.1
                        # if (adj_mat is None)==False:
                        #     adj_mat = adj_mat[np.random.randint(0,2,np.shape(adj_mat))]


                        
                # brain plot 
                if use_anat_colors == True:
                    #(using anat region colors with default colormap)
                    self.evalClus_brainPlot(ret_idx = ret_idx, marker_size=marker_size,snap_to_surface = snap_to_surface,plot_on_single_surface=plot_on_single_surface,c=anatDf['roi_color'].to_numpy(),view_in_browser=view_in_browser,save_fullpath=save_fullpath+'-'+clus_list[-1],plot_connectome=plot_connectome,adj_mat = adj_mat, adj_col = colmap(c_idx[i]), adj_linewidth=adj_linewidth)
                else:
                    #use cluster color
                    self.evalClus_brainPlot(ret_idx = ret_idx, marker_size=marker_size,snap_to_surface = snap_to_surface,plot_on_single_surface=plot_on_single_surface,c=np.tile(colmap(c_idx[i]),(np.sum(ret_idx),1)),view_in_browser=view_in_browser,save_fullpath=save_fullpath+'-'+clus_list[-1],plot_connectome=plot_connectome,adj_mat = adj_mat, adj_col = colmap(c_idx[i]), adj_linewidth=adj_linewidth)


        # update anatDf with cluster colors
        anatDf.insert(len(anatDf.columns),'clus_color',list(clus_colors))

        # brain plot (only if we are not plotting individual clusters)
        if plot_each_cluster == False:
            # if plot connectome is true, get adjancy matrix
            if plot_connectome==True:
                adj_mat = self.clus_getAdjMat(ret_idx=None,cut_level=cut_level)
            else:
                adj_mat = None

            adj_col = (0.5,0.5,0.5,0.5)

            # only select electrodes that were assigned a color
            ret_idx = clus_colors[:,3]>0
            self.evalClus_brainPlot(ret_idx = ret_idx,marker_size=marker_size,snap_to_surface = snap_to_surface,plot_on_single_surface=plot_on_single_surface,c=anatDf['clus_color'].to_numpy(),view_in_browser=view_in_browser,save_fullpath=save_fullpath+'-'+str(cut_level),adj_mat = adj_mat, adj_col = adj_col, adj_linewidth=0.1)

        return clus_list,clus_color_list


    def evalClus_rtCorrCounts(self,use_clean = True,ax = None,
        p_thresh = 0.05,figsize=(5,5),fsize_tick=12,beh_feat = 'zrrt',yL=None,delay_str='',ret_idx=None):

        self.plotCorr_counts(use_clean = use_clean,ax = ax,
        p_thresh = p_thresh,figsize=figsize,fsize_tick=fsize_tick,beh_feat = beh_feat,yL=yL,delay_str=delay_str,ret_idx=ret_idx)

    def evalClus_rtCorrCounts_anyTime(self,use_clean = True,ax = None,
        p_thresh = 0.05,figsize=(5,5),fsize_tick=12,fsize_lbl=16,beh_feat = 'zrrt',xvar_lbl=['S0f','S0c','postCC','postCC_bur','preResponse_bur','preResponse'],yL=None,delay_str='',ret_idx=None,plot_z = False,plot_counts=True):
        """ this function plots counts of electrodes that show sig. RT related correlations in any defined time interval; separately evaluates positive and negative effects """

        # make figure
        if ax==None:
            fig = plt.figure(figsize = figsize)
            ax = plt.subplot(111)
        ax2 = ax.twinx()

        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')


        # parse clean str
        if use_clean == True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'

        #parse delay lbl
        if delay_str == '':
            delay_lbl = 'all trials'
        elif delay_str == 'S':
            delay_lbl = 'short delay'
        elif delay_str == 'L':
            delay_lbl = 'long delay'

        # initialize containers to hold counts
        pos_idx = np.zeros(np.sum(ret_idx)).astype('bool')
        neg_idx = np.zeros(np.sum(ret_idx)).astype('bool')
        
        # loop through x var list and t-stats
        for i in range(0,len(xvar_lbl)):

            # get stats
            x = self.taskstats2d_df['rtCorr'+delay_str+'_'+beh_feat+'_'+xvar_lbl[i]+clean_str+'_zstatnp'].to_numpy()[ret_idx]

            # get pvals
            pvals = self.taskstats2d_df['rtCorr'+delay_str+'_'+beh_feat+'_'+xvar_lbl[i]+clean_str+'_pvalnp'].to_numpy()[ret_idx]

            pos_idx = pos_idx|((x>0)&(pvals<=p_thresh))
            neg_idx = neg_idx|((x<0)&(pvals<=p_thresh))

        # count it
        counts_dict = {}
        counts_dict['n_tot'] = np.sum(ret_idx)
        counts_dict['n_pos'] = np.sum(pos_idx)
        counts_dict['n_neg'] = np.sum(neg_idx)
        counts_dict['n_mixed'] = np.sum(pos_idx|neg_idx)

        # plot bar showing histogram of counts on ax 1
        if plot_counts == True:
            ax.bar(1,(np.count_nonzero(pos_idx)/len(pos_idx))*100,color='none',
            edgecolor='k',label = '(+) effects')
            ax.bar(1,-1*(np.count_nonzero(neg_idx)/len(neg_idx))*100, color='none',
            edgecolor='k',label = '(-) effects')

        # implement this later if you need
        if plot_z==True:
            pass 
        else:
            ax2.axis('off')

        # set ylabels
        if plot_counts==True:
            ax.set_ylabel('% electrodes \n significant',fontsize=fsize_lbl)
        ax.set_xlim(0.5,1.5)
        # set ytick labels
        if yL!=None:
            ax.set_ylim(yL)
        ax.set_yticklabels(np.abs(ax.get_yticks()).astype('int'))

        ax.set_title(' total n_elec = '+str(np.count_nonzero(ret_idx))+' n subj = '+str(np.count_nonzero(self.collapseBySubj_1d(ret_idx)))+'; p_thresh = '+str(np.round(p_thresh,2))+'\n'+delay_lbl+'; '+clean_lbl)
        plt.tight_layout()

        return counts_dict



    def evalClus_delayDiffCounts(self,use_clean = True,ax = None,
        p_thresh = 0.05,figsize=(5,5),fsize_tick=12,yL=None,ret_idx=None):

        self.plotDelayDiff_counts(use_clean = use_clean,ax = ax,
        p_thresh = p_thresh,figsize=figsize,fsize_tick=fsize_tick,yL=yL,ret_idx=ret_idx)

    def evalClus_delayDiffCounts_anyTime(self,use_clean = True,ax = None,
        p_thresh = 0.05,figsize=(5,5),fsize_tick=12,fsize_lbl=16,xvar_lbl=['S0f','S0c','postCC','postCC_bur','preResponse_bur','preResponse'],yL=None,ret_idx=None):
        """ this function plots counts of electrodes that show any delay-related increase vs. any delay-related decrease vs. mixed effects in any defined time interval """


        # make figure
        if ax==None:
            fig = plt.figure(figsize = figsize)
            ax = plt.subplot(111)

        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')


        # parse clean str
        if use_clean == True:
            clean_str = '_clean'
            clean_lbl = 'Residual z HFA'
        else:
            clean_str = ''
            clean_lbl = 'Original z HFA'

        # initialize containers to hold counts
        pos_idx = np.zeros(np.sum(ret_idx)).astype('bool')
        neg_idx = np.zeros(np.sum(ret_idx)).astype('bool')

        # loop through xvar lbls
        for i in range(0,len(xvar_lbl)):

            # skip if xvar lbl is pre-target baseline
            if xvar_lbl[i] == 'S0f':
                continue

            # get stats
            x = self.taskstats2d_df['delayDiff_'+xvar_lbl[i]+clean_str+'_zstatnp'].to_numpy()[ret_idx]

            # get pvals
            pvals = self.taskstats2d_df['delayDiff_'+xvar_lbl[i]+clean_str+'_pvalnp'].to_numpy()[ret_idx]

            pos_idx = pos_idx|((x>0)&(pvals<=p_thresh))
            neg_idx = neg_idx|((x<0)&(pvals<=p_thresh))

        # count it
        counts_dict = {}
        counts_dict['n_tot'] = np.sum(ret_idx)
        counts_dict['n_pos'] = np.sum(pos_idx)
        counts_dict['n_neg'] = np.sum(neg_idx)
        counts_dict['n_mixed'] = np.sum(pos_idx|neg_idx)


        # plot bar showing histogram of counts on ax 1
        ax.bar(1,(np.count_nonzero(pos_idx)/len(pos_idx))*100,color='0.5',
        edgecolor='k',label = '(+) effects')
        ax.bar(1,-1*(np.count_nonzero(neg_idx)/len(neg_idx))*100, color='0.5',
        edgecolor='k',label = '(-) effects')


        # set ylabels
        ax.set_ylabel('% electrodes \n significant',fontsize=fsize_lbl)


        # set ytick labels
        if yL!=None:
            ax.set_ylim(yL)
        ax.set_yticklabels(np.abs(ax.get_yticks()).astype('int'))


        ax.set_title(' total n_elec = '+str(np.count_nonzero(ret_idx))+' n subj = '+str(np.count_nonzero(self.collapseBySubj_1d(ret_idx)))+'; p_thresh = '+str(np.round(p_thresh,2))+clean_lbl)

        plt.tight_layout()

        ax.set_xlim(0.5,1.5)

        return counts_dict




    def evalClus_rtCorr(self,use_clean = True, ax = None,collapseBySubj_flag = True,figsize=(5,5),fsize_tick=14,beh_feat = 'zrrt',yL = None,delay_str='',ret_idx = None):

        self.plotCorr_results(use_clean = use_clean, ax = ax,collapseBySubj_flag = collapseBySubj_flag,figsize=figsize,fsize_tick=fsize_tick,beh_feat = beh_feat,yL = yL,delay_str=delay_str,ret_idx = ret_idx)

    def evalClus_fingerprint(self,ax = None,ret_idx = None, neu_feat_list = ['postTargOn','preResponseS_respLocked','postResponseS_respLocked'], anat_list = ['Occipital','Perirolandic-CST','Insula'],sel_feat_list = ['ptCorr_S0f','ptCorr_postTarg','errSel','rewSel','spatialSel'],p_thresh=0.05,use_roiList=True,count_by_hemis = True,count_anyChange=False, plot_option = 'sel',figsize=(5,5),yL=(0,40),atlas='default'):
        
        """ Plot bar plot that describes various features of the cluster in % of all electrodes held in collection. Also plots expected frequency for each feature. Hard codes features (for now)

        inputs:
        ax ... provides an axes to plot on
        ret_idx ... bool indicating membership in this cluster
        neu_feat_list .. list of neural features to get counts data on (amplitude only)
        anat_list ... list of regions to get counts data for
        sel_list ... list of selectivity tests to get counts data for
        p_thresh ... p threshold to use when identifying significant effects
        use_roiList ... if True, it will collect counts data on all regions of interest
        count_by_hemis ... if True, it will count left and right hemisphere separately 
        count_anyChange ... if True, it will count both increases and decreases (for activation function) and positive and negative effects (for selectivity)
        plot_option ... ('sel'). Only plots selectivity features. (actFunc and anatomy are not yet implemented)
        """
        #,'preResponseL_respLocked',,'postResponseL_respLocked','postNoCCS'

        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')

        # get anatDf (for all electrodes)
        anatDf,roiList = self.getAnatDf(ret_idx=None,cmap='rainbow',atlas=atlas);
        rois = anatDf['roi'].to_numpy()
        hemis = anatDf['hemis'].to_numpy()

        # get color
        from matplotlib import cm


        # activation function:
        # get features
        finger_actFunc_obsInClus = []
        finger_actFunc_obsTot = []
        finger_actFunc_lbls = []
        finger_actFunc_cols = []

        colmap = cm.get_cmap('tab20b',100) 
        c_idx = np.linspace(0,1,len(neu_feat_list)) # use this to index colors for each feature

        for n in neu_feat_list:

            if count_anyChange == True:
                # % activation increases
                finger_actFunc_lbls.append('actFunc-'+n)
                finger_actFunc_obsInClus.append(np.count_nonzero(np.abs(self.taskstats2d_df['modParams_'+n+'_amp'].to_numpy())[ret_idx]))
                finger_actFunc_obsTot.append(np.count_nonzero(np.abs(self.taskstats2d_df['modParams_'+n+'_amp'].to_numpy())))
                finger_actFunc_cols.append(colmap(c_idx[neu_feat_list.index(n)]))

            else:
                # % activation increases
                finger_actFunc_lbls.append('actFunc-'+n+'-inc')
                finger_actFunc_obsInClus.append(np.sum(self.taskstats2d_df['modParams_'+n+'_amp'].to_numpy()[ret_idx]>0))
                finger_actFunc_obsTot.append(np.sum(self.taskstats2d_df['modParams_'+n+'_amp'].to_numpy()>0))
                finger_actFunc_cols.append(colmap(c_idx[neu_feat_list.index(n)]))            
                # % activation decreases
                finger_actFunc_lbls.append('actFunc-'+n+'-dec')
                finger_actFunc_obsInClus.append(np.sum(self.taskstats2d_df['modParams_'+n+'_amp'].to_numpy()[ret_idx]<0))
                finger_actFunc_obsTot.append(np.sum(self.taskstats2d_df['modParams_'+n+'_amp'].to_numpy()<0))
                finger_actFunc_cols.append(colmap(c_idx[neu_feat_list.index(n)]))

        #for a in anat_list:
        # anatomy
        finger_anat_obsInClus = []
        finger_anat_obsTot = []
        finger_anat_lbls = []
        finger_anat_cols = []

        if use_roiList == True:
            anat_list = roiList

        for r in anat_list:
            if count_by_hemis == True:
                # left               
                finger_anat_lbls.append('anat-'+r+'-L')
                finger_anat_obsInClus.append(np.sum((rois[ret_idx]==r)&(hemis[ret_idx]=='L')))
                finger_anat_obsTot.append(np.sum((rois==r)&(hemis=='L')))
                finger_anat_cols.append(anatDf.query('roi==@r')['roi_color'][0])

                # right
                finger_anat_lbls.append('anat-'+r+'-R')
                finger_anat_obsInClus.append(np.sum((rois[ret_idx]==r)&(hemis[ret_idx]=='R')))
                finger_anat_obsTot.append(np.sum((rois==r)&(hemis=='R')))
                finger_anat_cols.append(anatDf.query('roi==@r')['roi_color'][0])
            else:
                finger_anat_lbls.append('anat-'+r)
                finger_anat_obsInClus.append(np.sum((rois[ret_idx]==r)))
                finger_anat_obsTot.append(np.sum((rois==r)))
                finger_anat_cols.append(anatDf.query('roi==@r')['roi_color'][0])

        # selectivity
        finger_sel_obsInClus = []
        finger_sel_obsTot = []
        finger_sel_lbls = []
        finger_sel_cols = []
        colmap = cm.get_cmap('tab10',100) 
        c_idx = np.linspace(0,1,len(neu_feat_list)) # use this to index colors for each feature


        for sel_feat in sel_feat_list:

            # parse statistic and p-value for various selectivity features

            if sel_feat == 'ptCorr_S0f':
                # pre-stimulus correlation of prediction times (expecation representation/integration)

                # calculate z-statistic
                x = self.taskstats2d_df['ptCorr_S0f_rval'].to_numpy()
                pvals = self.taskstats2d_df['ptCorr_S0f_pval'].to_numpy()
                thisCol = [0.8,0,0,1]# red


            elif sel_feat == 'ptCorr_postTarg':
                # pre-stimulus correlation of prediction times (expecation representation/integration)

                # calculate z-statistic
                x = self.taskstats2d_df['ptCorr_postTarg_rval'].to_numpy()
                pvals = self.taskstats2d_df['ptCorr_postTarg_pval'].to_numpy()
                thisCol = [0.8,0,0,0.8]# red
            elif sel_feat == 'rewSel':
                # change in post responseactivity during postitive feedback (RT<300)

                # calculate z-statistic
                x = self.taskstats2d_df['rewSel_tstat'].to_numpy()
                pvals = self.taskstats2d_df['rewSel_pval'].to_numpy()
                thisCol = [0.8,0,0.8,1]# purple
            elif sel_feat == 'errSel':
                # change in post response activity on error trials (prediction error)

                # calculate z-statistic
                x = self.taskstats2d_df['errSel_tstat'].to_numpy()
                pvals = self.taskstats2d_df['errSel_pval'].to_numpy()
                thisCol = [0.8,0,0.8,1]# purple
            elif sel_feat == 'spatialSel':
                # one way anova of post-target activity related to locations
                # there is no direction of effect here because its a one way anova, so we count all effects as poistive
                x = np.ones(len(ret_idx))
                pvals = self.taskstats2d_df['tlAnova_pval'].to_numpy()
                thisCol = [0,0,0.8,1]# blue                                
            if (count_anyChange == True)|(sel_feat=='spatialSel'):
                finger_sel_lbls.append(sel_feat)
                finger_sel_obsInClus.append(np.sum((pvals[ret_idx]<=p_thresh)))
                finger_sel_obsTot.append(np.sum((pvals<=p_thresh)))
                finger_sel_cols.append(thisCol)
                #finger_sel_cols.append(colmap(c_idx[sel_feat_list.index(sel_feat)]))

            else:
                finger_sel_lbls.append(sel_feat+'(+)')
                finger_sel_obsInClus.append(np.sum((x[ret_idx]>0)&(pvals[ret_idx]<=p_thresh)))
                finger_sel_obsTot.append(np.sum((x>0)&(pvals<=p_thresh)))
                finger_sel_cols.append(thisCol)
                #finger_sel_cols.append(colmap(c_idx[sel_feat_list.index(sel_feat)]))

            
                finger_sel_lbls.append(sel_feat+' (-)')
                finger_sel_obsInClus.append(np.sum((x[ret_idx]<0)&(pvals[ret_idx]<=p_thresh)))
                finger_sel_obsTot.append(np.sum((x<0)&(pvals<=p_thresh)))
                finger_sel_cols.append(list(np.array(thisCol)+np.array([0.1,0.1,0.1,0])))

                #finger_sel_cols.append(colmap(c_idx[sel_feat_list.index(sel_feat)]))

        # get counts into a dictionary
        counts_dict = {}
        counts_dict['n_tot'] = len(ret_idx)
        counts_dict['n_totInClus'] = np.sum(ret_idx)

        counts_dict['lbls_actFunc'] = finger_actFunc_lbls
        counts_dict['obsInClus_actFunc'] = finger_actFunc_obsInClus
        counts_dict['obsTot_actFunc'] = finger_actFunc_obsTot
        counts_dict['cols_actFunc'] = finger_actFunc_cols

        counts_dict['lbls_anat'] = finger_anat_lbls
        counts_dict['obsInClus_anat'] = finger_anat_obsInClus
        counts_dict['obsTot_anat'] = finger_anat_obsTot
        counts_dict['cols_anat'] = finger_anat_cols

        counts_dict['lbls_sel'] = finger_sel_lbls
        counts_dict['obsInClus_sel'] = finger_sel_obsInClus
        counts_dict['obsTot_sel'] = finger_sel_obsTot
        counts_dict['cols_sel'] = finger_sel_cols

        # global chi square test. Null hypothesis is that this cluster is a random subsample of all electrodes
        #obsInClus = np.array(counts_dict['obsInClus_actFunc'] + counts_dict['obsInClus_anat'] + counts_dict['obsInClus_sel'])
        #obsTot = np.array(counts_dict['obsTot_actFunc'] + counts_dict['obsTot_anat'] + counts_dict['obsTot_sel'])
        #expInClus = obsTot*np.array((counts_dict['n_totInClus']/counts_dict['n_tot']))


        # anat chi square test. Null hypothesis is that this cluster is a random subsample of all electrodes
        obsInClus = np.array(counts_dict['obsInClus_anat'])
        obsTot = np.array(counts_dict['obsTot_anat'])
        expInClus = obsTot*np.array((counts_dict['n_totInClus']/counts_dict['n_tot']))

        chisq, pval = stats.chisquare(f_obs=obsInClus,f_exp=expInClus) 
        counts_dict['chisq'] = chisq
        counts_dict['pval'] = pval

        # plot bar plot
        if plot_option == 'sel':
            # make figure
            if ax is None:
                f = plt.figure(figsize=figsize)
                ax = plt.subplot(111)

            #build full feat list
            freq_clus = np.array(counts_dict['obsInClus_sel'])/counts_dict['n_totInClus']
            freq_tot = np.array(counts_dict['obsTot_sel'])/counts_dict['n_tot']
            lbls = counts_dict['lbls_sel']
            cols = counts_dict['cols_sel']

            # plot bar plot
            ax.bar(np.arange(0,len(freq_clus)),freq_clus*100,color =cols, edgecolor='None')
            ax.bar(np.arange(0,len(freq_tot)),freq_tot*100,facecolor = 'None', edgecolor='k',alpha = 0.5)
            ax.set_xticks(np.arange(0,len(freq_clus)))
            ax.set_xticklabels(lbls,rotation=90)
            ax.set_ylabel('% electrodes')
            if (yL is None)==False:
                ax.set_ylim(yL)
            plt.tight_layout()
        elif plot_option == '':
            pass


        return counts_dict

    def evalClus_taskstatsTimeCourseByTime(self,ax = None,ret_idx = None,xL_ms = None,yL = None,time_bin_size_ms = 50,use_resp_locked=False,use_clean_data = False, plot_counts = False,z_thresh = 2,fsize_lbl = 14,print_status=True):
        """Plots counts data showing number electrodes with mean z-scored power values greater than z_thresh. Plots positive and negative counts separately. 
        inputs:
        'retIdx' applys a boolean mask on dataframe
        'time_bin_size_ms' length of time bins to use when binning data (because different sampling rates in different subjects)
        'use_resp_locked' bool indicating whether or not to use response locked data to do counts
        'use_clean_data' bool indicating whether or not to use cleaned data
        'plot_counts' bool indicating whether to show counts data based on 'z_thresh'
        'z_thresh' threshold to use for counts analysis 

        """
        # parse ax
        if ax is None:
            f = plt.figure()
            ax = plt.subplot(111)
        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')
        # parse response locked 
        if use_resp_locked == True:
            resp_str = '_respLocked'
            tc_xval_lbl = 'mod_xval_respLocked'
            xLbl = 'Time from response (ms)'
        else:
            resp_str = ''
            tc_xval_lbl = 'mod_xval'
            xLbl = 'Time from target onset (ms)'

        # parse use clean data
        if use_clean_data == True:
            clean_str = '_clean'
            z_str = ''
            tit = 'Clean data'
        else:
            clean_str = ''
            z_str = '_z'
            tit = 'Original data'

        # construct field labels for time course
        tcS_lbl = 'responseS'+clean_str+resp_str+z_str
        tcL_lbl = 'responseL'+clean_str+resp_str+z_str

        #subfunction to get xval_ms
        def get_xval_data(eIdx,tc_xval_lbl):
            """ gets xval_ms for a given eIdx (entry in self.taskstats2d_df)"""


            # get xval in samples
            xval_samp = self.taskstats2d_df.iloc[eIdx][tc_xval_lbl]

            # get samplerate of this electrode
            sr = self.taskstats2d_df.iloc[eIdx]['samplerate']

            # convert to samp to ms
            xval_ms = (xval_samp/sr)*1000

            return xval_samp,sr,xval_ms
        # subfunction to process an electrode
        def tc2timeBins(eIdx,tcS_lbl,tcL_lbl,tc_xval_lbl,time_bin_size_ms):
            """ takes eIdx (an index for an entry in self.taskstats2d_df) and returns binned time course data  """

            if self.isBadElectrode_list[eIdx] == True:
                raise NameError('Do not run this function for bad Electrodes')

            # get tc in samp
            tcS_samp = self.taskstats2d_df.iloc[eIdx][tcS_lbl]
            tcL_samp = self.taskstats2d_df.iloc[eIdx][tcL_lbl]

            # get xval_data from eIdx
            xval_samp,sr,xval_ms = get_xval_data(eIdx,tc_xval_lbl)

            # make time bins
            time_bin_starts = np.arange(xval_ms[0],xval_ms[-1],time_bin_size_ms)

            # init containter
            tcS_ms = np.zeros(len(time_bin_starts))
            tcL_ms = np.zeros(len(time_bin_starts))

            # loop through and return mean time course
            for t in np.arange(0,len(time_bin_starts)):

                # define sample start and end (relative to ev onset)
                samp_start = ((time_bin_starts[t]/1000)*sr).astype('int')
                samp_step = ((time_bin_size_ms/1000)*sr).astype('int')
                samp_end = samp_start+samp_step


                #define indices for this time window based on matching samp_start and samp_end with xval_samp
                samp_idx = np.arange(np.argmin(np.abs(xval_samp-samp_start)),np.argmin(np.abs(xval_samp-samp_end)))

                tcS_ms[t] = np.nanmean(tcS_samp[samp_idx])
                tcL_ms[t] = np.nanmean(tcL_samp[samp_idx])

            return tcS_ms,tcL_ms
        # get electrode list to loop throughs
        uElbl_list = self.taskstats2d_df.index.to_numpy()[ret_idx]

        # get xval_samp for first good electrode
        first_good_elec_idx = np.nonzero(np.array(self.isBadElectrode_list)==False)[0][0]
        xval_samp,sr,xval_ms = get_xval_data(first_good_elec_idx,tc_xval_lbl)
        time_bin_starts = np.arange(xval_ms[0],xval_ms[-1],time_bin_size_ms)

        # Initialize containers based on xval_ms
        tcS_ms = np.zeros((len(uElbl_list),len(time_bin_starts)))
        tcL_ms = np.zeros((len(uElbl_list),len(time_bin_starts)))
        tcS_ms[:] = np.nan
        tcL_ms[:] = np.nan

        # initialize counts arrays (short/long, pos/neg)
        countsS_pos_ms = np.zeros((len(uElbl_list),len(time_bin_starts)))
        countsS_neg_ms = np.zeros((len(uElbl_list),len(time_bin_starts)))
        countsL_pos_ms = np.zeros((len(uElbl_list),len(time_bin_starts)))
        countsL_neg_ms = np.zeros((len(uElbl_list),len(time_bin_starts)))

        # loop through electrodes
        for i in np.arange(0,len(uElbl_list)):

            # get index
            eIdx = np.nonzero(self.taskstats2d_df.index.to_numpy()==uElbl_list[i])[0][0]


            #skip if bad electrode
            if self.isBadElectrode_list[eIdx] == True:
                continue

            #get time course short, time course long, xval_samp, xval_ms
            tcS_ms[i,:],tcL_ms[i,:] = tc2timeBins(eIdx,tcS_lbl,tcL_lbl,tc_xval_lbl,time_bin_size_ms)


            if print_status == True:
                print(np.round(100*(i/len(uElbl_list))))

        # compute counts
        countsS_pos_ms = 100*(np.sum(tcS_ms>z_thresh,axis=0)/len(uElbl_list))
        countsS_neg_ms = 100*(np.sum(tcS_ms<-z_thresh,axis=0)/len(uElbl_list))
        countsL_pos_ms = 100*(np.sum(tcL_ms>z_thresh,axis=0)/len(uElbl_list))
        countsL_neg_ms = 100*(np.sum(tcL_ms<-z_thresh,axis=0)/len(uElbl_list))


        # plot
        if plot_counts == False:
            plt.plot(time_bin_starts,np.nanmean(tcS_ms,axis=0))
            plt.plot(time_bin_starts,np.nanmean(tcL_ms,axis=0))
            plt.set_ylabel('z HFA')
        else:
            plt.plot(time_bin_starts,countsS_pos_ms,label='increase/short delay',color='C0')
            plt.plot(time_bin_starts,countsS_neg_ms,label='decrease/short delay',color='C0',linestyle='--')            
            plt.plot(time_bin_starts,countsL_pos_ms,label='increase/short delay',color='C1')
            plt.plot(time_bin_starts,countsL_neg_ms,label='decrease/long delay',color='C1',linestyle='--') 
            ax.set_ylabel('% electrodes')
            ax.legend(fontsize=fsize_lbl)

        # parse xlabel
        ax.set_xlabel(xLbl,fontsize=fsize_lbl)
        ax.set_title(tit+' n = '+str(np.sum(ret_idx)),fontsize=fsize_lbl)

        # set ylim
        if (yL is None) == False:
            ax.set_ylim(yL[0],yL[1])

        # set x lim
        if (xL_ms is None) == False:
            ax.set_xlim(xL_ms[0],xL_ms[1])

        # plot vertical lines
        if use_resp_locked == True:
            ax.vlines(0,ax.get_ylim()[0],ax.get_ylim()[1],alpha = 0.5,linestyle='--')
        else:
            ax.vlines((0,500,1500),ax.get_ylim()[0],ax.get_ylim()[1],alpha = 0.5,linestyle='--')



        # init plot dict
        plot_dict = {}

        # fill plot_dict
        plot_dict['time_bin_starts'] = time_bin_starts
        plot_dict['tcS_ms'] = tcS_ms
        plot_dict['tcL_ms'] = tcL_ms
        plot_dict['countsS_pos_ms'] = countsS_pos_ms
        plot_dict['countsS_neg_ms'] = countsS_neg_ms
        plot_dict['countsL_pos_ms'] = countsL_pos_ms
        plot_dict['countsL_neg_ms'] = countsL_neg_ms

        return plot_dict


    def evalClus_taskstatsTimeCourse(self,ret_idx = None, subDir_lbl = '',figsize=(10,3),close_figs_flag=True,atlas='default'):
        # This function plots electrode-by-electrode report for all electrodes currently in the Collection. It uses cached data from task stats (so need to run that first).
        # Minimalist plot. Fixation locked and response locked with vertical lines indicating key time points. Axes labels are off (need to indicate these with another plot)
        # fixation locked, cc-locked and response locked

        # label sub directory
        if subDir_lbl=='':
            subDir_lbl = 'unlabelledGroup'
        subdir_name = ('ELECREP-timeCourse-'+subDir_lbl+'-'
                         +self.taskstats2d_pow_frange_lbl
                         +str(self.taskstats2d_do_zscore)
                         +str(self.taskstats2d_apply_time_bins)
                         +str(self.taskstats2d_time_bin_size_ms)
                         +str(self.taskstats2d_apply_gauss_smoothing)
                         +str(self.taskstats2d_gauss_sd_scaling)
                         +self.taskstats2d_pow_method)


        subdir_path = self.paramsFig_dir+subdir_name+'/'

        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')

        # make sub directory
        if os.path.exists(subdir_path)==False:
            os.mkdir(subdir_path)

        # get anatDf (for all electrodes)
        anatDf,roiList = self.getAnatDf(ret_idx=None,cmap='rainbow',atlas=atlas);

        # loop through electrodes in object
        for uElbl in self.taskstats2d_df.index[ret_idx]:

            # create figure
            f = plt.figure(figsize=figsize,facecolor='w',num=uElbl) 
            

            # create subplots and axes
            gs = f.add_gridspec(nrows=2,ncols=6)
            ax_fix = f.add_subplot(gs[:,0:4])
            ax_resp = f.add_subplot(gs[:,4:])

            
            # plot fix start
            self.plotTaskStats2d_timeCourse(taskstats2d_S=self.taskstats2d_df.loc[uElbl], ax = ax_fix, lbl=None,evType = 'FIX_START', yL = None, xL_ms = None,add_vline=True,fsize_lbl=16,fsize_tick=16,alpha = 0.6,color_short = None,color_long = None,add_legend = False,add_title=True,overlay_model_fit=True)

            # clean axis
            ax_fix.spines['right'].set_visible(False)
            ax_fix.spines['top'].set_visible(False)
            ax_fix.spines['bottom'].set_visible(False)
            ax_fix.set_xticks([])
            ax_fix.set_xlabel('')
            yt = ax_fix.get_yticks()
            ax_fix.set_yticks((yt[0],yt[-1]))
            ax_fix.set_yticklabels((str(int(yt[0])),str(int(yt[-1]))),fontsize=16)


            #plot response
            self.plotTaskStats2d_timeCourse(taskstats2d_S=self.taskstats2d_df.loc[uElbl], ax = ax_resp, lbl=None,evType = 'RESPONSE', yL = None, xL_ms=(-1000,1000),add_vline=True,fsize_lbl=14,fsize_tick=14,alpha = 0.6,color_short = None,color_long = None,add_legend = False,add_title=False,overlay_model_fit=True)
            ax_resp.set_yticks([])
            ax_resp.set_ylabel('')
            ax_resp.axis('off')

            # set title (region and corresponding text color )
            ax_fix.set_title(uElbl+'-'+anatDf.loc[uElbl]['roi'],fontsize=20,color =anatDf.loc[uElbl]['roi_color'])

            plt.tight_layout()

            # save figure
            f.savefig(subdir_path+'TIMECOURSE-'+uElbl)
            # option to close figures
            if close_figs_flag ==True:
                plt.close(f)



    def evalClus_brainPlot(self,ret_idx=None,c='r',marker_size=10, snap_to_surface = True,plot_on_single_surface=True,view_in_browser=True,save_fullpath='',plot_connectome=False,adj_mat = None,adj_col = (1,0,0,1),adj_linewidth=0.5):

        #This function plots electrodes on a brain surface.
        # ret_idx ... sets the subset of electrodes to plot (e.g., can be for a cluster, region or a subject)
        # c_list ... list of colors. if it is a single value, it will apply the same color to all markers, otherwise, it expects a list of length that matches ret_idx
        # marker_size ... size of markers
        # if plot on single_surface = True, then it plots all electrodes together, otherwise it plots separately on left and right hemispheres
        # snap_to_surface ... it snaps electrodes that are outside the brain surface to the nearest vertex
        # if view in brower = True it will open up a browser with the image to interact with
        # if save_fullpath is '', it wont save html page, if a full path is provied, it will save an .html page 
        # if plot connectome is true, it plots connections between electrodes as defined by adjacency matrix
        # adj_mat = None, Adjacency matrix to use if we are plotting the connectome
        # adj_col.... RGBA tuple used to color connections
        # adj_linewidth ... sets weights of edge connections


        from nilearn import plotting,surface,datasets

        # parse ret_idx
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')

        # error check
        if (plot_connectome==True)&(adj_mat is None):
            raise NameError('if plot_connectome = True, must provide an adjacency matrix')
        
        # get x y z (for this electrode group per ret_idx)
        xs= np.array(self.mni_x)[:,np.newaxis]
        ys = np.array(self.mni_y)[:,np.newaxis]
        zs = np.array(self.mni_z)[:,np.newaxis]
        
        xs = xs[ret_idx]
        ys = ys[ret_idx]
        zs = zs[ret_idx]

        coords = np.hstack((xs,ys,zs))

        # snap
        if snap_to_surface==True:
            # get fsaverage pial surface (to snap electrodes)
            fsaverage = datasets.fetch_surf_fsaverage()

            surfL_path = fsaverage['pial_left']
            surfR_path = fsaverage['pial_right']

            surfL = surface.load_surf_mesh(surfL_path)
            surfR = surface.load_surf_mesh(surfR_path)

            surfL_coords = surfL[0]
            surfR_coords = surfR[0]


            # compute pairwise distances between each electrode and each surface (left and right hemisphere)
            D_R = pairwise_distances(coords,surfR_coords)
            D_L = pairwise_distances(coords,surfL_coords)
            

            # loop through each electrode and identify outlier electodes by considering relative location to nearest surface coordinate
            for e in range(0,np.sum(ret_idx)):

                # parse whether left or right hemisphere
                if coords[e,0]>0:
                    # RIGHT HEMI
                    nearest_surf_coord = surfR_coords[np.argmin(D_R[e,:]),:]

                    # if electrode is more right or more superior than this coord, then snap. This should account for 3d curvaure
                    #Empirically ,inferior misalignments arent an issue
                    if (coords[e,0]>nearest_surf_coord[0])|(coords[e,2]>nearest_surf_coord[2]):
                        coords[e,:] = nearest_surf_coord

                else:
                    # LEFT HEMI
                    nearest_surf_coord = surfL_coords[np.argmin(D_L[e,:]),:]

                    # if electrode is more left or more superior than this coord, then snap. This should account for 3d curvaure. Empirically ,inferior misalignments arent an issue
                    if (coords[e,0]<nearest_surf_coord[0])|(coords[e,2]>nearest_surf_coord[2]):
                        coords[e,:] = nearest_surf_coord


            # plot surface (in python) - this is super slow and clunky
            # f = plotting.plot_surf(surfL,alpha = 0.5)

            # # plot electrode
            # from mpl_toolkits import mplot3d
            # plt.gca().scatter3D(coords[0],coords[1],coords[2])
        
        #parse colors
        if (len(c) == 1)&(str(c)==True):
            c_list = [c]*np.sum(ret_idx)
            # this errors out (need to fix this, require a tuple for now)
        elif (type(c)==tuple):
            # we have a ()
            c_list = np.tile(np.array(c),(np.sum(ret_idx),1))
        else:
            c_list = c[ret_idx]

        # if we are plotting on a single surface
        if plot_on_single_surface==True:
            if plot_connectome==True:# expects adjacency matrix
                # color the connections based on the color provided(adj_col)
                from matplotlib.colors import LinearSegmentedColormap 
                colmap2 = LinearSegmentedColormap.from_list('connect',[(1,1,1,1),adj_col])
                # only plot connections with edge weight > 0, linewidth at 0.5
                view = plotting.view_connectome(adjacency_matrix=adj_mat,node_coords = coords,node_size = marker_size,linewidth=adj_linewidth,edge_threshold=0,edge_cmap=colmap2)
            else:
                view = plotting.view_markers(coords,marker_size = marker_size,marker_color = np.array(c_list))

            if view_in_browser == True:
                view.open_in_browser()
            if (save_fullpath == '')==False:
                view.save_as_html(save_fullpath+'.html')
        else:
            # plot separately on left and right
            # identify electrodes on left hemisphere
            l_idx = np.array(coords[:,0]<0).astype('bool')
            r_idx = np.array(coords[:,0]>=0).astype('bool')

            #plot left on brain surface
            if plot_connectome==True:#[ ] not implemented because we want to include bilateral connections 
                raise NameError('if plot_connectome = True, must set plot_on_single_surface = True so that we show bilateral connections')

            else:
                viewL = plotting.view_markers(coords[l_idx,:],marker_size = marker_size,marker_color = np.array(c_list)[l_idx])
                viewR = plotting.view_markers(coords[r_idx,:],marker_size = marker_size,marker_color = np.array(c_list)[r_idx])

            if view_in_browser == True:
                viewL.open_in_browser()
                viewR.open_in_browser()

            # if save
            if (save_fullpath == '')==False:
                viewL.save_as_html(save_fullpath+'-L.html')
                viewR.save_as_html(save_fullpath+'-R.html')

    # plot anatomy
    def evalClus_anat(self,ax = None,fsize =(7,5),ret_idx = None,title = '',yL=None,plot_raw_counts=False,fsize_tick = 14, fsize_lbl = 16,use_colormap=True,cmap = 'rainbow', plot_by_hemis=False,atlas='default',add_text_labels=True,alpha_hline=0.5):
        """Function to plot counts or electrode frequencies by region"""    

        # parse inputs
        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')
        # parse title
        if title == '':
            title = '(n obs/n total) = '+str(np.count_nonzero(ret_idx))+'/'+str(len(ret_idx))

        # get anatDf (do not apply ret_idx so you can compare clus in region to total electrodes in region)
        anatDf,roi_list = self.getAnatDf(ret_idx=None,cmap = cmap,atlas=atlas);
        
        # get roi, roi_color, and hemisphere
        rois = anatDf['roi'].to_numpy()
        roi_color = anatDf['roi_color'].to_numpy()
        hemis = anatDf['hemis'].to_numpy()

        # get total number of electrodes
        tot = len(rois)

        # frequency of this cluster among all electrodes, and separately for right and left hemisphere
        thisClus_freq = np.count_nonzero(ret_idx)/len(ret_idx)
        thisClus_freq_L = np.count_nonzero(ret_idx&(hemis=='L'))/np.count_nonzero(hemis=='L')
        thisClus_freq_R = np.count_nonzero(ret_idx&(hemis=='R'))/np.count_nonzero(hemis=='R')

        # make fig
        if ax is None:
            f = plt.figure(figsize=fsize)
            ax = plt.subplot(111)


        # containers for counts
        counts_numInReg = []
        counts_thisClusInReg = []

        # laterality counts analysis
        counts_numInReg_L = []
        counts_numInReg_R = []
        counts_thisClusInReg_L = []
        counts_thisClusInReg_R = []
        #print(np.unique(anat_list[ret_idx]))
        binom_zvals= []


        # loop through regions
        count = -1
        for r in roi_list:
            count+=1
            
            # count num of elecs in this region (total)
            numInReg = np.count_nonzero(np.array(rois)==r)
            numInReg_L = np.sum((np.array(rois)==r)&(np.array(hemis)=='L'))
            numInReg_R = np.sum((np.array(rois)==r)&(np.array(hemis)=='R'))

            # count number of this group in region 
            numThisClusInReg = np.count_nonzero(np.array(rois)[ret_idx]==r)
            numThisClusInReg_L = np.count_nonzero((np.array(rois)[ret_idx]==r)&(np.array(hemis)[ret_idx]=='L'))
            numThisClusInReg_R = np.count_nonzero((np.array(rois)[ret_idx]==r)&(np.array(hemis)[ret_idx]=='R'))





            # parse colormap
            if use_colormap==True:
                c = roi_color[np.where(np.array(rois)==r)[0][0]]
            else:
                c = '0.5'

            # plot bar
            width = 0.8
            if plot_raw_counts == True:

                if plot_by_hemis == True:

                    # plot left as negative values
                    plt.bar(count,-numInReg_L,color='None',edgecolor='k',width = width)
                    plt.bar(count,-numThisClusInReg_L,color=c,edgecolor='k',width = width)
                    #ax.text(count-.15,-50,str(numThisClusInReg_L)+'/'+str(numInReg_L),rotation=90,fontsize=fsize_tick)

                    #plot right counts
                    plt.bar(count,numInReg_R,color='None',edgecolor='k',width = width)
                    plt.bar(count,numThisClusInReg_R,color=c,edgecolor='k',width = width)
                    #ax.text(count-.15,50,str(numThisClusInReg_L)+'/'+str(numInReg_L),rotation=90,fontsize=fsize_tick)

                else:
                    b = plt.bar(count,numInReg,color='None',edgecolor='k',width = width)
                    b = plt.bar(count,numThisClusInReg,color=c,edgecolor='k',width = width)
                    if add_text_labels==True:
                        t = ax.text(count+.15,5,str(numThisClusInReg)+'/'+str(numInReg),rotation=90,fontsize=fsize_tick)
            else:
                if plot_by_hemis == True:
                    clus_prct_L =100*(numThisClusInReg_L/numInReg)
                    clus_prct_R =100*(numThisClusInReg_R/numInReg)
                    # plot left hemis percentage as negative values
                    b_l = plt.bar(count,-clus_prct_L,color=c,edgecolor=None,width = width)
                    # plot right hemis percentage as pos values
                    b_r = plt.bar(count,clus_prct_R,color=c,edgecolor=None,width = width)
                    # draw a rectangle surrounding both bars
                    # calculate bottom left anchor point
                    xy = np.array(b_l.patches[0].get_xy())
                    xy[1] = xy[1]+b_l.patches[0].get_height()
                    
                    # draw a rectangle
                    r = matplotlib.patches.Rectangle(xy,b_l.patches[0].get_width(),np.abs(b_l.patches[0].get_height())+b_r.patches[0].get_height(),fill=False,facecolor = 'none',edgecolor='k',linewidth=1)
                    ax.add_patch(r)

                    # text indicating number electrode assigned to cluster in region (numerator)
                    ax.text(xy[0]+width/2,xy[1]+r.get_height()/2,'('+str(numThisClusInReg)+')',rotation=90,horizontalalignment='center',verticalalignment='center',fontsize=fsize_tick)
                else:
                    # get overall percentage of n this cluster electrodes in region/total electrodes in region
                    clus_prct =100*(numThisClusInReg/numInReg)

                    plt.bar(count,clus_prct,color=c,edgecolor='k',width = width)
                    if add_text_labels==True:
                        ax.text(count-.15,2,'('+str(numThisClusInReg)+')',rotation=90,fontsize=fsize_tick)

            # append counts lists
            counts_numInReg.append(numInReg)
            counts_thisClusInReg.append(numThisClusInReg)
            counts_numInReg_L.append(numInReg_L)
            counts_numInReg_R.append(numInReg_R)
            counts_thisClusInReg_L.append(numThisClusInReg_L)
            counts_thisClusInReg_R.append(numThisClusInReg_R)

            # do binomial test (obs vs. exp); 
            # k = num of cluster electrodes observed in this region
            # n = total number of electrodes in this region
            # p = expected frequency based on prevalence of this cluster across the brain

            # we use a one-tailed test (alternative = "less") because we are interested in obtaining z-values that indicate how the observed frequence deviates from expectation (and not the p-values). In this case, positive z-values indicate greater frequency than expected by chance and negative p-values indicate lower frequency than expected size  

            # collect z-statistics
            binom = stats.binomtest(k=numThisClusInReg,n=numInReg,p=thisClus_freq,alternative = 'less')
            # convert p to z (we dont have to do 1-p because it is a one tailed p)
            binom_zvals.append(stats.norm.ppf(binom.pvalue))       

   

        
        # set ticks
        if yL is None:
            ax.set_ylim((-np.max(np.abs(ax.get_ylim()))),(np.max(np.abs(ax.get_ylim()))))
        else:
            ax.set_ylim(yL)

        ax.set_xlim((-.5,len(roi_list)-0.5))
        ax.set_xticks(np.arange(0,len(roi_list)))
        ax.set_xticklabels(roi_list,fontsize=fsize_tick,rotation=90);
        #ax.set_xlabel('Region',fontsize=20)
        if plot_raw_counts==True:
            if plot_by_hemis==True:
                ax.set_ylabel('(L) Number of electrodes (R)',fontsize=fsize_lbl)
            else:
                ax.set_ylabel('Number of electrodes',fontsize=fsize_lbl)
        else:
            if plot_by_hemis==True:
                ax.set_ylabel('(L) % of electrodes (R)',fontsize=fsize_lbl)
            else:
                ax.set_ylabel('% of electrodes',fontsize=fsize_lbl)

            if plot_by_hemis == True:
                if alpha_hline is None:
                    ax.hlines(100*-thisClus_freq_L,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='dashed',alpha = 0.5)
                    ax.hlines(100*thisClus_freq_R,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='dashed',alpha = 0.5)
                else:
                    ax.hlines(100*-alpha_hline,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='dashed',alpha = 0.5)
                    ax.hlines(100*alpha_hline,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='dashed',alpha = 0.5)                    
            else:
                if alpha_hline is None:
                    ax.hlines(100*thisClus_freq,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='dashed',alpha = 0.5)
                else:
                    ax.hlines(100*alpha_hline,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='dashed',alpha = 0.5)

        # print chi2 stats

        ax.set_yticks((ax.get_ylim()[0],100*-thisClus_freq_L,0,100*thisClus_freq_R,ax.get_ylim()[1]))
        ax.set_yticklabels(np.abs(ax.get_yticks().astype('int')),fontsize=fsize_tick,rotation=90)
        
        if plot_by_hemis == True:
            # compute chi sq with each region as an idependent observation


            chisq,p = stats.chisquare(f_obs=counts_thisClusInReg_L+counts_thisClusInReg_R, \
                                  f_exp=list(np.array(counts_numInReg_L)*thisClus_freq_L)+list(np.array(counts_numInReg_R)*thisClus_freq_R))
        else:
            chisq,p = stats.chisquare(f_obs=counts_thisClusInReg, \
                                  f_exp=np.array(counts_numInReg)*thisClus_freq)


        #print title with chi sq stats
        ax.set_title(title+' chisq = '+str(np.round(chisq,3)) + ' p= '+str(np.round(p,4)),fontsize=fsize_lbl)

        plt.tight_layout()

        # return plot_dict
        plot_dict = {}
        plot_dict['counts_numInReg'] = counts_numInReg
        plot_dict['counts_thisClusInReg'] =counts_thisClusInReg
        plot_dict['prct_obs'] = list(np.array(counts_thisClusInReg)/np.array(counts_numInReg))
        plot_dict['prct_exp'] = thisClus_freq
        plot_dict['prct_deviation_from_expected'] = list(np.array(plot_dict['prct_obs'])-thisClus_freq)
        plot_dict['binom_zvals'] = binom_zvals
        plot_dict['counts_numInReg_L'] = counts_numInReg_L
        plot_dict['counts_numInReg_R'] = counts_numInReg_R
        plot_dict['counts_thisClusInReg_L'] = counts_thisClusInReg_L
        plot_dict['counts_thisClusInReg_R'] = counts_thisClusInReg_R
        plot_dict['roi_list'] = roi_list


        return plot_dict

        
    
    # plot RT by pow
    def evalClus_powByRT2d(self, ret_idx = None,lbl=None, ax = None,add_vline=True,fsize_lbl=16,fsize_tick=16,fsize_title=10,fsize_leg = 10,yL=None,xL_ms=None,figsize = (8,4),add_legend=True,add_title = True,collapseBySubj_flag=False,binByRT = False,color = None,alpha = 0.75,delays_list = [500,1500]):
        # This function plots mean power in various RT bins within each delay condition. It overwrites the function in Electrode class (uses cached data from the getPow_2d function, so need to run that first)
    
        # create axes  
        if ax == None:
            fig = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        # indicator for whether color is None (color gets overwritten later)
        color_init = color


        if ret_idx is None:
            ret_idx = np.ones(len(self.uElbl_list)).astype('bool')

        if lbl == None:
            if self.pow_evQuery==None:
                lbl = 'all electrodes'
            else:
                lbl = self.pow_evQuery

        # parse xlim 
        if xL_ms == None:
            if self.pow_apply_time_bins == False:
                xL_ms = (self.samples_to_ms(self.pow_xval[0]),self.samples_to_ms(self.pow_xval[-1]))
            else:
                xL_ms = (self.pow_xval[0],self.pow_xval[-1])

        
        # loop through delays
        for d in delays_list:

            # set the color for the delay condition
            if d == 500:
                if color_init is None:
                    color = 'C0'
            elif d == 1500:
                if color_init is None:
                    color = 'C1'

            # parse bin by RT
            if binByRT == True:
                # loop through bins
                for b in np.arange(0,self.num_bins):
                    binPowMat = getattr(self,'binPow_'+str(d))[b,:,:].T

                    # parse collapse by subj
                    if collapseBySubj_flag == True:
                        binPowMat = self.collapseBySubj_2d(binPowMat[ret_idx,:],subj_list_ret_idx = ret_idx)
                    else:
                        binPowMat = binPowMat[ret_idx,:]

                    binPowMat_mean = np.nanmean(binPowMat,axis=0)
                    binPowMat_sem = stats.sem(binPowMat,axis=0,nan_policy='omit')
                    # plot it
                    ax.plot(self.pow_xval,np.nanmean(binPowMat,axis=0), color = color,alpha = 0.1+(b/10))
                    ax.fill_between(self.pow_xval,binPowMat_mean+binPowMat_sem,binPowMat_mean-binPowMat_sem,alpha=0.1,color = '0.8')
            else:
                # collapse across rt bins for each delay condition
                binPowMat = np.nanmean(getattr(self,'binPow_'+str(d)),axis=0).T

                # parse collapse by subj
                if collapseBySubj_flag == True:
                    binPowMat = self.collapseBySubj_2d(binPowMat[ret_idx,:],subj_list_ret_idx= ret_idx)
                else:
                    binPowMat = binPowMat[ret_idx,:]

                binPowMat_mean = np.nanmean(binPowMat,axis=0)
                binPowMat_sem = stats.sem(binPowMat,axis=0,nan_policy='omit')
                # plot it
                ax.plot(self.pow_xval,np.nanmean(binPowMat,axis=0), color = color,alpha = alpha)
                ax.fill_between(self.pow_xval,binPowMat_mean+binPowMat_sem,binPowMat_mean-binPowMat_sem,alpha=alpha-0.25,color = color)

        # if x val are in samples, then covert tick labels
        if self.pow_apply_time_bins == False:
            xt = np.array([self.pow_xval[0],0,0.5*self.samplerate,self.pow_xval[-1]])
            ax.set_xticks(xt)
            ax.set_xticklabels(np.round(1000*(xt/self.samplerate)).astype('int'),fontsize=fsize_tick)
            ax.set_xlim((self.ms_to_samples(xL_ms[0]),self.ms_to_samples(xL_ms[1])))
        else:
            if self.pow_evType=='FIX_START':
                xt = np.array([self.pow_xval[0],0,500,1500])
            else:
                xt = np.array(np.linspace(xL_ms[0], xL_ms[1],3))

            ax.set_xticks(xt)
            ax.set_xticklabels(np.round(xt,2),fontsize=fsize_tick)
            ax.set_xlim((xL_ms[0],xL_ms[1]))

        
        #set x label
        ax.set_xlabel('Time from '+self.pow_evType+' (ms)',fontsize=fsize_lbl)
       

        #set y label    
        if self.pow_do_zscore == True:
            ax.set_ylabel('z-score '+self.pow_frange_lbl,fontsize=fsize_lbl)
        else:
            ax.set_ylabel('Power (a.u.)',fontsize=fsize_lbl)

        # set ylim
        if yL != None:
            ax.set_ylim(yL)
        else:
            yL = ax.get_ylim()


        # set yticklabels
        plt.yticks(np.linspace(yL[0], yL[1],5), np.round(np.linspace(yL[0], yL[1],5),2), fontsize=fsize_tick)

        # v line
        if add_vline==True:
            # get vL_ticks
            if self.pow_apply_time_bins == False:
                if evType=='FIX_START':
                    vL_ticks = [0,int(0.5*self.samplerate),int(1.5*self.samplerate)]
                else:
                    vL_ticks = [0]

            else:
                if self.pow_evType=='FIX_START':
                    vL_ticks= [0,500,1500]
                else:
                    vL_ticks= [0]

            for v in vL_ticks:
                ax.vlines(x=v,
                      ymin=ax.get_ylim()[0],
                      ymax=ax.get_ylim()[1],
                      linestyles='--',alpha=0.5,color='k')
        # legend
        if add_legend == True:
            lines = ax.get_lines()
            if (binByRT==True)&len(delays_list)==2:
                plt.legend((lines[0],lines[9],lines[10],lines[19]),
                               ('short delay/fast RT','short delay/slow RT','long delay/fast RT','long delay/slow RT'),fontsize=fsize_leg)
            else:
                plt.legend((lines[0],lines[-1]),('short delay','long delay'),fontsize=fsize_leg)

        #title
        if add_title == True:
            ax.set_title('n electrodes = '+str(np.count_nonzero(ret_idx))+' n subj = '+str(np.count_nonzero(self.collapseBySubj_1d(ret_idx)))+'\n Collapse by subj = '+str(collapseBySubj_flag)+'; '+lbl,fontsize=fsize_title)

        # fig layout
        plt.tight_layout()
    ############ wrapper around subjCollection objecr ####
    def getSubjCollectionDf(self,filter_bool,filter_bool_lbl, min_obs_thresh = 5,evQuery='error==0&fastResponse==0',overwriteFlag=False,popStats_iters = 1000,doStats_by_groups = False):
        #this function loops through subjects and collects behavioral and population-level neural statistics. Assumes that Collection has not been filtered


        # Load and return if we already have saved this file
        sc_df_fname = 'SC-DF-'+filter_bool_lbl+'-popStats_iters-'+str(popStats_iters)

        if (os.path.exists(self.params_dir+sc_df_fname)==True)&(overwriteFlag==False):
            
            #load file if it exists
            self.sc_df = (self.load_pickle(self.params_dir+
                                                 sc_df_fname))

            return self.sc_df

        # if we have not saved or are overwriting, continue with loop
        subj_list = list(np.unique(self.subj_list))

        # init containers
        sc_dict_list = []
        subj_included = []


        # get clus_ret_matrix
        if hasattr(self,'clus_cut_tree') == False:
            raise NameError('run self.clusterElectrodesByTaskStats() first before collecting subjCollection dataframe ')
        else:
            clus_ret_mat = self.clus_getMasterRetIdxMat(cut_level=19)

        for s in subj_list:

            # get subj collection
            sc = SubjCollection(subj=s,filter_bool = filter_bool,filter_bool_lbl = filter_bool_lbl)

            # skip if we dont have enough data
            if sc.n_obs < min_obs_thresh:
                print ('SKIPPING '+s)
                continue
            # init dict to hold data
            sc_dict = {}
            sc_dict['subj'] = s
            sc_dict['n_obs'] = sc.n_obs
            subj_included.append(s)


            # init subj
            S = Subject(s)

            # get RT difference
            # # std
            # rts short delay
            rts_s = S.getRTs(evQuery = evQuery+'&delay==500', rt_dist_type = 'standard')
            zrrt_s = S.getRTs(evQuery = evQuery+'&delay==500', rt_dist_type = 'zrrt')

            # rts long delay 
            rts_l = S.getRTs(evQuery = evQuery+'&delay==1500', rt_dist_type = 'standard')
            zrrt_l = S.getRTs(evQuery = evQuery+'&delay==1500', rt_dist_type = 'zrrt')

            sc_dict['rtDiff_mean'] = np.mean(rts_l)-np.mean(rts_s)
            sc_dict['zrrtDiff_mean'] = np.mean(zrrt_l)-np.mean(zrrt_s)
            sc_dict['rtDiff_std'] = np.std(rts_l)-np.std(rts_s)
            sc_dict['zrrtDiff_std'] = np.std(zrrt_l)-np.std(zrrt_s)

            # get error rate and fast response rate 
            choiceEv = S.ev_df.query('type=="RESPONSE"')
            sc_dict['error_rateL'] = np.sum(choiceEv.eval('RT<0&delay==1500'))/np.sum(choiceEv.eval('delay==1500'))
            sc_dict['error_rateS'] = np.sum(choiceEv.eval('RT<0&delay==500'))/np.sum(choiceEv.eval('delay==500'))
            sc_dict['error_diff'] = sc_dict['error_rateL'] - sc_dict['error_rateS']
            sc_dict['guess_rateL'] = np.sum(choiceEv.eval('RT<-500&delay==1500'))/np.sum(choiceEv.eval('delay==1500'))

            # fit LATER 2
            rts_A,rts_B,pred_idx_A,pred_idx_B = S.getRTs_for_LATER2()

            # FIT LATER 2
            sc_dict.update(S.fitLATER2_byCondition(rts_A,rts_B,pred_idx_A, pred_idx_B,model_type = 'std_bias'))


            # get pop response
            sc.getPopulationResponse(pow_frange_lbl='HFA',pow_method='wave',pow_evQuery=evQuery, do_zscore=True,        apply_gauss_smoothing=True, gauss_sd_scaling=0.075,
                num_iters=1, apply_time_bins=False, time_bin_size_ms=100,overwriteFlag=False,feat_list_beh = ['zrrt'],
                run_light=True,popStats_iters=popStats_iters)

            # update with popStats_dict
            sc_dict.update(sc.popStats_dict)


            # manually compute fraction of electrodes that showed HFA changes after 500 ms on long delay trials
            # this subj idx (for self that has taskstats)
            this_subj_idx = np.array(self.subj_list)==s
            sc_dict['popByDelay_postNoCC_diff'] = np.count_nonzero(self.taskstats2d_df['modParams_postNoCCS_amp'].to_numpy()[this_subj_idx])/np.sum(this_subj_idx)

            # get dimensionality of population data
            sc_dict['numDimensions'] = np.nonzero(np.cumsum(sc.pca_mod.explained_variance_ratio_)>.95)[0][0]

            if doStats_by_groups == True:

                # # get key memreg features
                # memReg_key_list = ['delayCondIsLong_tstat','errorMemFast_tstat','shortDelayMem_tstat','tau']
                # for k in memReg_key_list:
                #     sc_dict[k] = sc.memReg_dict[k]

                # measure how distributed the distance effects are (partial correlation)
                sc_dict.update(sc.pop_doStats_by_pca_cum(beh_var_lbl='zrrt',neu_var_lbl='SR_headingCorrect',covar_list = ['SR_dist'],stat_option='corrPartial'))

                # measure how distributed the heading direction effects are (partial correlation)
                sc_dict.update(sc.pop_doStats_by_pca_cum(beh_var_lbl='zrrt',neu_var_lbl='SR_speed',covar_list = ['SR_dist'],stat_option='corrPartial'))

                # measure how distributed the heading direction effects are (partial correlation)
                sc_dict.update(sc.pop_doStats_by_pca_cum(beh_var_lbl='zrrt',neu_var_lbl='SR_dist',covar_list = ['SR_headingCorrect','SR_speed'],stat_option='corrPartial'))


                ##### do pop stats by region 

                #(independent, forward, min elec thresh = 5)
                popStats_byRegion_dict = sc.pop_doStats_by_region(beh_var_lbl ='zrrt',min_elec_thresh = 5,master_roi_list = None,neu_var_list=['SR_dist','SR_speed','SR_headingCorrect'],stat_option='corrPartial',covar_list=[['SR_speed','SR_headingCorrect'],['SR_dist'],['SR_dist']],do_cumulative=False,do_reverse=False)
                # #update sc_dict
                sc_dict.update(popStats_byRegion_dict)


                #(cumulative, forward, min elec thresh = 1)
                popStats_byRegion_dict = sc.pop_doStats_by_region(beh_var_lbl ='zrrt',min_elec_thresh = 1,master_roi_list = None,neu_var_list=['SR_dist','SR_speed','SR_headingCorrect'],stat_option='corrPartial',covar_list=[['SR_speed','SR_headingCorrect'],['SR_dist'],['SR_dist']],do_cumulative=True,do_reverse=False)
                # #update sc_dict
                sc_dict.update(popStats_byRegion_dict)

                #(cumulative, reverse, min elec thresh = 1)
                popStats_byRegion_dict = sc.pop_doStats_by_region(beh_var_lbl ='zrrt',min_elec_thresh = 1,master_roi_list = None,neu_var_list=['SR_dist','SR_speed','SR_headingCorrect'],stat_option='corrPartial',covar_list=[['SR_speed','SR_headingCorrect'],['SR_dist'],['SR_dist']],do_cumulative=True,do_reverse=True)
                sc_dict.update(popStats_byRegion_dict)


                ##### do pop stats by cluster

                # # Get matrix of ret_idx for this subject (exclude bad electrodes)
                clus_ret_mat_thisSubj = clus_ret_mat[(sc.subj_bool)&(sc.isBadElectrode_bool==False),:]

                # (independent, forward, min elec thresh = 5)
                popStats_byClus_dict =sc.pop_doStats_by_clusLevel(clus_ret_mat=clus_ret_mat_thisSubj,beh_var_lbl ='zrrt',min_elec_thresh = 5,neu_var_list=['SR_dist','SR_speed','SR_headingCorrect'],stat_option='corrPartial',covar_list=[['SR_speed','SR_headingCorrect'],['SR_dist'],['SR_dist']],do_cumulative=False,do_reverse=False)
                # #update sc_dict with pop stats by cluster 
                sc_dict.update(popStats_byClus_dict)

                # (cumulative, forward, min elec thresh = 1)
                popStats_byClus_dict =sc.pop_doStats_by_clusLevel(clus_ret_mat=clus_ret_mat_thisSubj,beh_var_lbl ='zrrt',min_elec_thresh = 5,neu_var_list=['SR_dist','SR_speed','SR_headingCorrect'],stat_option='corrPartial',covar_list=[['SR_speed','SR_headingCorrect'],['SR_dist'],['SR_dist']],do_cumulative=True,do_reverse=False)
                # #update sc_dict with pop stats by cluster 
                sc_dict.update(popStats_byClus_dict)

                # (cumulative, reverse, min elec thresh = 1)
                popStats_byClus_dict =sc.pop_doStats_by_clusLevel(clus_ret_mat=clus_ret_mat_thisSubj,beh_var_lbl ='zrrt',min_elec_thresh = 5,neu_var_list=['SR_dist','SR_speed','SR_headingCorrect'],stat_option='corrPartial',covar_list=[['SR_speed','SR_headingCorrect'],['SR_dist'],['SR_dist']],do_cumulative=True,do_reverse=True)
                # #update sc_dict with pop stats by cluster 
                sc_dict.update(popStats_byClus_dict)

            # #append dict to list
            sc_dict_list.append(sc_dict)

            print(subj_list.index(s),'/',len(subj_list))

        # convert to data frame
        sc_df = pd.DataFrame(sc_dict_list,index = subj_included)

        # save pickle
        self.save_pickle(obj=sc_df,fpath=self.params_dir+sc_df_fname)

        # save in self
        self.sc_df = sc_df

        return sc_df
    def sc_plotPopStats_byDelay(self,ax=None,figsize=(7,5),fsize_tick=14,fsize_lbl=14,neu_feat_list = ['SR_dist','SR_rate','SR_speed','SR_var','SR_headingCorrect'],use_zstats=False,resid_str = ''):
        # plot regression results and confidence intervales
        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        # loop through neural features
        count = -1
        xtick_lbls = []
        for n in neu_feat_list:
            if 'pc' in n:
                # skip PCA coords (which do not have specific tstats)
                continue
            count+=1
            xtick_lbls.append(n)
            if use_zstats==False:
                lbl = 'popByDelay'+resid_str+'_'+n+'_diff'
                ylbl = 'diff score'
            elif use_zstats==True:
                lbl = 'popByDelay'+resid_str+'_'+n+'_zstatnp'
                ylbl = 'z stat (non param)'

            x = self.sc_df[lbl].to_numpy()

            # plot true tstat (with 95% ci)
            ax.bar(x=count,height=np.nanmean(x),yerr=stats.sem(x,nan_policy='omit')*1.96,color='0.5',edgecolor='k',ecolor='k')

        ax.set_xlim((-.5,count+.5))
        ax.set_xticks(np.arange(0,count+1))
        ax.set_xticklabels(xtick_lbls,fontsize=fsize_tick,rotation=45)
        ax.set_ylabel(ylbl,fontsize=fsize_lbl)
        ax.set_title('dynamics by delay '+' 95% CI error bars',fontsize=fsize_lbl)

    def sc_plotPopStats_byDelayCounts(self,ax=None,figsize=(7,5),fsize_tick=14,fsize_lbl=14,neu_feat_list = ['SR_dist','SR_rate','SR_speed','SR_var','SR_headingCorrect'],p_thresh=0.05,resid_str ='',use_zstats=True):
        # plot regression results and confidence intervales
        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        # containers
        n_tot = []
        n_obs = []
        n_exp = []
        n_obs_pos = []
        n_obs_neg = []


        # loop through neural features
        count = -1
        xtick_lbls = []
        for n in neu_feat_list:
            if 'pc' in n:
                # skip PCA coords (which do not have specific diffstat)
                continue
            count+=1
            xtick_lbls.append(n)
            if use_zstats==True:
                diffstat = self.sc_df['popByDelay'+resid_str+'_'+n+'_zstatnp'].to_numpy()
            else:
                diffstat = self.sc_df['popByDelay'+resid_str+'_'+n+'_diff'].to_numpy()
            pvals = self.sc_df['popByDelay'+resid_str+'_'+n+'_pvalnp'].to_numpy()

            counts_pos = (diffstat>0)&(pvals<p_thresh)
            counts_neg = (diffstat<0)&(pvals<p_thresh)

            # plot counts
            ax.bar(count,(np.count_nonzero(counts_pos)/len(counts_pos))*100,color='C0',
            edgecolor='k',label = 'positive effects')
            ax.bar(count,(np.count_nonzero(counts_neg)/len(counts_pos))*100,
            bottom = (np.count_nonzero(counts_pos)/len(counts_pos))*100, color='C1',
            edgecolor='k',label = 'negative effects')

            if count == 0:
                legend(fontsize=fsize_tick)

            # populate container
            n_tot.append(len(counts_pos))
            n_obs.append(np.sum(counts_pos)+np.sum(counts_neg))
            n_exp.append(p_thresh*len(counts_pos))
            n_obs_pos.append(np.sum(counts_pos))
            n_obs_neg.append(np.sum(counts_neg))


        ax.set_xlim((-.5,count+.5))
        ax.hlines(p_thresh*100,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='--',alpha =0.5)
        ax.set_xticks(np.arange(0,count+1))
        ax.set_xticklabels(xtick_lbls,fontsize=fsize_tick,rotation=45)
        ax.set_ylabel('% subjects',fontsize=fsize_lbl)
        ax.set_title('n subj = '+str(len(counts_pos))+' p threshold = '+str(np.round(p_thresh,2)),fontsize=fsize_lbl)


        #populate plot_dict
        plot_dict = {}
        plot_dict['n_tot'] = n_tot
        plot_dict['n_obs'] = n_obs
        plot_dict['n_exp'] = n_exp
        plot_dict['n_obs_pos'] = n_obs_pos
        plot_dict['n_obs_neg'] = n_obs_neg
        plot_dict['lbls'] = xtick_lbls

        #
        return plot_dict


    def sc_plotPopStats_reg(self,beh_var_lbl = 'zrrt',ax=None,fsize_tick=14,fsize_lbl=14,neu_feat_list = ['SR_dist','SR_rate','SR_speed','SR_var','SR_headingCorrect'],stat_option='tstat',d_str = '',use_zstats = True,figsize=(7,5)):
        # plot regression results and confidence intervales
        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        # loop through neural features
        count = -1
        xtick_lbls = []
        for n in neu_feat_list:
            if 'pc' in n:
                # skip PCA coords (which do not have specific tstats)
                continue
            count+=1
            xtick_lbls.append(n)
            if stat_option=='tstat':
                if use_zstats==True:
                    end_str = '_zstatnp'
                else:
                    end_str = '_tstat'
                lbl = 'popReg'+d_str+'_'+beh_var_lbl+'_'+n+end_str
            elif stat_option=='corr':
                if use_zstats==True:
                    end_str = '_zstatnp'
                else:
                    end_str = '_rval'
                lbl = 'popCorr'+d_str+'_'+beh_var_lbl+'_'+n+end_str


            elif stat_option=='corrPartial':
                if use_zstats==True:
                    end_str = '_zstatnp'
                else:
                    end_str = '_rval'
                lbl = 'popCorrPartial'+d_str+'_'+beh_var_lbl+'_'+n+end_str

            x = self.sc_df[lbl].to_numpy()

            # plot true tstat (with 95% ci)
            ax.bar(x=count,height=np.mean(x),yerr=stats.sem(x)*1.96,color='0.5',edgecolor='k',ecolor='k')

        ax.set_xlim((-.5,count+.5))
        ax.set_xticks(np.arange(0,count+1))
        ax.set_xticklabels(xtick_lbls,fontsize=fsize_tick,rotation=45)
        if use_zstats==True:
            ax.set_ylabel('z stat (non param)',fontsize=fsize_lbl)
        elif stat_option in ['corr','corrPartial']:
            ax.set_ylabel('Spearman r',fontsize=fsize_lbl)
        else:
            ax.set_ylabel('t stat',fontsize=fsize_lbl)

        ax.set_title(beh_var_lbl+' 95% CI error bars',fontsize=fsize_lbl)

    def sc_plotPopStats_regCounts(self,beh_var_lbl = 'zrrt',ax=None,fsize_tick=14,fsize_lbl=14,p_thresh=0.05,neu_feat_list = ['SR_dist','SR_rate','SR_speed','SR_var','SR_headingCorrect'],stat_option='tstat',d_str = '',figsize=(7,5)):
        # plot regression results and confidence intervales
        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.subplot(111)

        # containers
        n_tot = []
        n_obs = []
        n_exp = []
        n_obs_pos = []
        n_obs_neg = []


        # loop through neural features
        count = -1
        xtick_lbls = []
        for n in neu_feat_list:
            if 'pc' in n:
                # skip PCA coords (which do not have specific tstats)
                continue
            count+=1
            xtick_lbls.append(n)

            if stat_option=='tstat':
                lbl = 'popReg'+d_str+'_'+beh_var_lbl+'_'+n+'_tstat'
                split_lbl = '_tstat'
            elif stat_option=='corr':
                lbl = 'popCorr'+d_str+'_'+beh_var_lbl+'_'+n+'_rval'
                split_lbl = '_rval'
            elif stat_option=='corrPartial':
                lbl = 'popCorrPartial'+d_str+'_'+beh_var_lbl+'_'+n+'_rval'
                split_lbl = '_rval'


            #tstats = self.sc_df[lbl].to_numpy()
            tstats = self.sc_df[lbl.split(split_lbl)[0]+'_zstatnp'].to_numpy()
            pvals = self.sc_df[lbl.split(split_lbl)[0]+'_pvalnp'].to_numpy()

            counts_pos = (tstats>0)&(pvals<p_thresh)
            counts_neg = (tstats<0)&(pvals<p_thresh)


            ax.bar(count,(np.count_nonzero(counts_pos)/len(counts_pos))*100,color='C0',
            edgecolor='k',label = 'positive effects')
            ax.bar(count,(np.count_nonzero(counts_neg)/len(counts_pos))*100,
            bottom = (np.count_nonzero(counts_pos)/len(counts_pos))*100, color='C1',
            edgecolor='k',label = 'negative effects')

            if count == 0:
                legend(fontsize=fsize_tick)

            # populate container
            n_tot.append(len(counts_pos))
            n_obs.append(np.sum(counts_pos)+np.sum(counts_neg))
            n_exp.append(p_thresh*len(counts_pos))
            n_obs_pos.append(np.sum(counts_pos))
            n_obs_neg.append(np.sum(counts_neg))



            # # plot error bar (manually)
            # ax.plot((count,count),(cineg,cipos),color = '0.5')
            # ax.plot((count-.25,count+.25),(cineg,cineg),color = '0.5')
            # ax.plot((count-.25,count+.25),(cipos,cipos),color = '0.5')

            # # plot true tstat (with 95% ci)
            # ax.errorbar(x=count,y=np.mean(tstats),yerr=stats.sem(tstats)*1.96,fmt='.k',ecolor='0.5',markersize=10)

            # write pval?
            #ax.text(count+.1,tstat-.3,s='p = '+str(np.round(pvalnp,2)),fontsize=fsize_tick)
        ax.set_xlim((-.5,count+.5))
        ax.hlines(p_thresh*100,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='--',alpha =0.5)
        ax.set_xticks(np.arange(0,count+1))
        ax.set_xticklabels(xtick_lbls,fontsize=fsize_tick,rotation=45)
        ax.set_ylabel('% subjects',fontsize=fsize_lbl)
        ax.set_title(beh_var_lbl+' n subj = '+str(len(counts_pos))+' p threshold = '+str(np.round(p_thresh,2)),fontsize=fsize_lbl)

        #populate plot_dict
        plot_dict = {}
        plot_dict['n_tot'] = n_tot
        plot_dict['n_obs'] = n_obs
        plot_dict['n_exp'] = n_exp
        plot_dict['n_obs_pos'] = n_obs_pos
        plot_dict['n_obs_neg'] = n_obs_neg
        plot_dict['lbls'] = xtick_lbls

        #
        return plot_dict


    def sc_plotPopStats_regCounts_SSE(self,beh_var_lbl = 'zrrt',ax = None,fsize_tick=14,fsize_lbl=14,p_thresh=0.05,neu_feat_list = ['S0_pc','St_pc']):
        # plot popStats SSE results and associated confidence intervals (for predicting behav. variables based on PCA coords)
        if ax is None:
            f = plt.figure()
            ax = plt.subplot(111)

        # containers
        n_tot = []
        n_obs = []
        n_exp = []

        # loop through neural features
        count = -1
        for n in neu_feat_list:
            count+=1

            lbl = 'popReg_'+beh_var_lbl+'_'+n+'_SSE'

            pvals = self.sc_df[lbl+'_pvalnp']
            counts = (pvals<p_thresh)

            ax.bar(count,100*(np.sum(counts)/len(counts)),color='0.5')

            # populate container
            n_tot.append(len(counts))
            n_obs.append(np.sum(counts))
            n_exp.append(p_thresh*len(counts))

        ax.set_xlim((-.5,1.5))
        ax.set_xticks([0,1])
        ax.set_xticklabels(neu_feat_list,fontsize=fsize_tick)
        ax.set_ylabel('% subjects',fontsize=fsize_lbl)
        ax.hlines(p_thresh*100,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='--',alpha =0.5)
        ax.set_title(beh_var_lbl+' n subj = '+str(len(counts))+' p threshold = '+str(np.round(p_thresh,2)),fontsize=fsize_lbl)

        #populate plot_dict
        plot_dict = {}
        plot_dict['n_tot'] = n_tot
        plot_dict['n_obs'] = n_obs
        plot_dict['n_exp'] = n_exp
        plot_dict['lbls'] = neu_feat_list

        #
        return plot_dict
    def sc_plotPopStats_cumPCA(self,ax=None,num_dim = 100,beh_var_lbl = 'zrrt',neu_var_lbl = 'SR_headingDirection',stat_option='corrPartial',fsize_lbl = 14):

        if ax is None:
            f = plt.figure()
            ax = plt.subplot(111)

        # figure out dimensionality across subjects
        num_dim_vec = self.sc_df['cumPCA_num_dim'].to_numpy()
        if num_dim is None:
            num_dim = np.max(num_dim_vec)
        
        # create a matrix
        statS_mat = np.zeros((len(num_dim_vec),num_dim))
        statS_mat[:] = np.nan
        statL_mat = np.zeros((len(num_dim_vec),num_dim))
        statL_mat[:] = np.nan

        # populate it
        #loop through subjects
        for s in range(0,len(num_dim_vec)):
            statS = self.sc_df['cumPCAS_'+beh_var_lbl+'_'+neu_var_lbl].iloc[s]
            statL = self.sc_df['cumPCAL_'+beh_var_lbl+'_'+neu_var_lbl].iloc[s]

            statS_mat[s,:np.min((len(statS),num_dim))] = statS[:np.min((len(statS),num_dim))]
            statL_mat[s,:np.min((len(statL),num_dim))] = statL[:np.min((len(statS),num_dim))]

        # plot it
        # short delay 
        ax.fill_between(np.arange(1,num_dim+1),np.nanmean(statS_mat,axis=0)+stats.sem(statS_mat,axis=0,nan_policy='omit')*1.96, np.nanmean(statS_mat,axis=0)-stats.sem(statS_mat,axis=0,nan_policy='omit')*1.96,alpha=0.5)
        # long delay 
        ax.fill_between(np.arange(1,num_dim+1),np.nanmean(statL_mat,axis=0)+stats.sem(statL_mat,axis=0,nan_policy='omit')*1.96, np.nanmean(statL_mat,axis=0)-stats.sem(statL_mat,axis=0,nan_policy='omit')*1.96,alpha=0.5)

        ax.set_xlabel('Number of principal components',fontsize=fsize_lbl)
        if self.sc_df['cumPCA_stat_option'][0] in ['corr','corrPartial']:
            ax.set_ylabel('Spearman $r$',fontsize=fsize_lbl)
        else:
            ax.set_ylabel('t stats',fontsize=fsize_lbl)
        ax.set_title(self.sc_df['cumPCA_stat_option'][0]+'-'+beh_var_lbl+'-'+neu_var_lbl)






class SubjCollection(Collection):
    # This Class inherits from Collection
    # Its purposes is to perform population-level analyses for a given subject
    # Input should be a list of  uElbls ("subj - elec1 - elec2") and a collection_lbl (for saving purposes). if list of uElbls is empty, will process the label (e.g., allsubj-allelecs)
    # Main functions should include:
    # collecting data by looping through list of u-eLbls - should be able to collect anatomical data, 2d power, 3d power or phase, or statistical tests implemented by SubjectElectrode
    # for each object, saving/loading group level object
    # build customFeatureMatrix (combination of above data)
    # Filtering electrodes by some statistical test (write this functionality into SubjElectrode)
    # Dimensionality Reduction
    # Clustering based on feature matrix
    # Making summary plots (avg power, avg stats, or anatomical data)
    # collapsing within subjects for across subj stats etc

    # CONSTRUCTOR
    def __init__(self, subj, filter_bool = None,filter_bool_lbl = None,paramsDict = None):


        # initialize collection object
        Collection.__init__(self,paramsDict=paramsDict)


        # initialize Subject object so it updates key fields
        #Subject.__init__(self,subj=subj)


        # initialize basic attributes
        self.subj = subj
        self.subj_bool = np.copy(np.array(self.subj_list)==subj)

        # parse filter electrodes
        if filter_bool is None:
            self.filter_bool = np.copy(np.ones(len(self.subj_list)).astype('bool'))
        else:
            self.filter_bool = filter_bool

        if filter_bool_lbl is None:
            self.filter_bool_lbl = 'None'
        else:
            self.filter_bool_lbl = filter_bool_lbl


        self.isBadElectrode_bool = np.copy(np.array(self.isBadElectrode_list).astype('bool'))


        # filter electrodes
        self.filterElectrodes(filtE_bool=self.subj_bool)

        # create bool to filter this subjects data
        self.thisSubjFilt_bool = np.copy(self.filter_bool[(self.subj_bool)&(self.isBadElectrode_bool==False)])

        # can use this to decide whether we have enough subjects for a full analysis
        self.n_obs = np.sum(self.thisSubjFilt_bool)

    ####### POPULATION ANALYSES #######
    # get population response matrix
    def getPopulationResponse(self,pow_frange_lbl='HFA',pow_method='wave',
        pow_evQuery='error==0&fastResponse==0',
        do_zscore=True,
        apply_gauss_smoothing=True,
        gauss_sd_scaling=0.075,
        apply_time_bins=False,num_iters=1,
        time_bin_size_ms=100,
        overwriteFlag=False,feat_list_beh = None,run_light = True,popStats_iters = 1000):


        if feat_list_beh is None:
            feat_list_beh_str = ''
        else:
            feat_list_beh_str = str(feat_list_beh)


        # hold input params in self
        self.pop_pow_evQuery = pow_evQuery
        self.pop_pow_frange_lbl=pow_frange_lbl
        self.pop_pow_method=pow_method
        self.pop_pow_evQuery=pow_evQuery
        self.pop_do_zscore=do_zscore
        self.pop_apply_gauss_smoothing=apply_gauss_smoothing
        self.pop_gauss_sd_scaling=gauss_sd_scaling
        self.pop_apply_time_bins=apply_time_bins
        self.pop_num_iters=num_iters

        # check if we have saved a pop response
        self.popResponse_fname = (('popResponse-'
                                  +self.subj
                                  +'-'
                                   +pow_frange_lbl
                                   +pow_method
                                   +pow_evQuery
                                   +str(do_zscore)
                                   +str(apply_gauss_smoothing)
                                   +str(gauss_sd_scaling)
                                   +str(apply_time_bins)
                                   +str(time_bin_size_ms)+'num_iters'+str(num_iters)+feat_list_beh_str))


        # look for saved file
        if (os.path.exists(self.params_dir+self.popResponse_fname)==True)&(overwriteFlag==False):

            #load file if it exists
            self.popResponse_dict = (self.load_pickle(self.params_dir+
                                                 self.popResponse_fname))
        else:
            # this function loops through electrodes, loads taskstats2d, and collects a response segment (eg. cc_locked, post color change), and RT data

            popMatS = []
            popMatL = [] 


            # init matrix (trials x time x electrodes)
            count = 0
            for uElbl in self.uElbl_list:

                # init subj electrode
                if run_light == True:
                    do_init = False
                else:
                    do_init = True

                SE = SubjElectrode(subj=uElbl.split('-')[0],elec1_lbl=uElbl.split('-')[1], elec2_lbl=uElbl.split('-')[2],do_init=do_init) 

                if self.isBadElectrode_list[list(self.uElbl_list).index(str(uElbl))]== True:
                    continue
                #else
                count +=1


                # load taskstats (to initialize parameters)
                SE.doTaskStats_2d(pow_frange_lbl=pow_frange_lbl,pow_method=pow_method,pow_evQuery=pow_evQuery, do_zscore=do_zscore,    apply_gauss_smoothing=apply_gauss_smoothing,gauss_sd_scaling=gauss_sd_scaling,apply_time_bins=apply_time_bins,time_bin_size_ms=time_bin_size_ms,overwriteFlag=overwriteFlag,feat_list_beh=feat_list_beh,num_iters=num_iters)

                # re run model time course function to get trial by trial data
                if hasattr(SE,'responseModel_dict') == False:
                    SE.taskstats_modelTimeCourseAndGetResponseFeatures()

                # populate response matrix based on response model dict
                if count == 1:
                    popMatS = SE.responseModel_dict['responseS_trials_clean_ccLocked_postCC'][:,:,np.newaxis]
                    popMatL = SE.responseModel_dict['responseL_trials_clean_ccLocked_postCC'][:,:,np.newaxis]
                    rtS = SE.taskstats2d['rt'][SE.taskstats2d['shortTrials_bool']]
                    rtL = SE.taskstats2d['rt'][SE.taskstats2d['longTrials_bool']]

                    #
                    se_samplerate = SE.taskstats2d['samplerate']
                    se_pow_ev_filt = SE.taskstats2d['pow_ev_filt']

                else:
                    popMatS = np.concatenate((popMatS,SE.responseModel_dict['responseS_trials_clean_ccLocked_postCC'][:,:,np.newaxis]),axis=2)
                    popMatL = np.concatenate((popMatL,SE.responseModel_dict['responseL_trials_clean_ccLocked_postCC'][:,:,np.newaxis]),axis=2)

                    print(count,'/',len(self.uElbl_list))

            # init dict
            self.popResponse_dict = {}

            # update with pop mat
            self.popResponse_dict['popMatS'] = popMatS
            self.popResponse_dict['popMatL'] = popMatL

            # save rt data, short and long trial bools
            self.popResponse_dict['rtS'] = rtS
            self.popResponse_dict['rtL'] = rtL

            #collect key params from SE
            self.popResponse_dict['samplerate'] = se_samplerate
            self.popResponse_dict['pow_ev_filt'] = se_pow_ev_filt

            #save pickle
            self.save_pickle(obj=self.popResponse_dict,fpath=self.params_dir+self.popResponse_fname)


        # now we have loaded key data
        # generate popMat (for all trials). We will use this for PCA
        self.popResponse_dict['popMat'] = np.concatenate((self.popResponse_dict['popMatS'],self.popResponse_dict['popMatL']),axis=0)
        self.popResponse_dict['rt'] = np.concatenate((self.popResponse_dict['rtS'],self.popResponse_dict['rtL']),axis=0)

        # create bool to index short and long trials from popResponse_dict (popMat and rts)
        this_bool =  np.zeros(len(self.popResponse_dict['rt']))
        this_bool[0:len(self.popResponse_dict['rtS'])] = 1
        self.popResponse_dict['pop_shortTrials_bool'] = this_bool.astype('bool')
        self.popResponse_dict['pop_longTrials_bool'] = (this_bool.astype('bool')==False)

        # apply electrode filt bool (outside if/then statement)
        self.popResponse_dict['popMat'] = self.popResponse_dict['popMat'][:,:,self.thisSubjFilt_bool]
        self.popResponse_dict['popMatS'] = self.popResponse_dict['popMatS'][:,:,self.thisSubjFilt_bool]
        self.popResponse_dict['popMatL'] = self.popResponse_dict['popMatL'][:,:,self.thisSubjFilt_bool]

        #update popResponse_dict with trajectory params on filtered data
        self.popResponse_dict.update(self.pop_getTrajParams(popMat3d=self.popResponse_dict['popMat'],rts_ms=self.popResponse_dict['rt']))

        # do PCA on combined data (short and long trials)
        self.popResponse_dict['popMat2d_pc'],pca_mod = self.pop_fitPCA(self.popResponse_dict['popMat'])
        # save pca_mod in self
        self.pca_mod = pca_mod

        # save trial by trial PC data
        self.popResponse_dict['popMat3d_pc'] = self.pop_vec2mat(self.popResponse_dict['popMat2d_pc'],trial_len_samp = self.popResponse_dict['popMat'].shape[1])

        # get trial by trial PCA features (coordinates of starting point and threshold in PC space)
        self.popResponse_dict.update(self.pop_getPCAParams(popMat3d_pc=self.popResponse_dict['popMat3d_pc'],rts_ms=self.popResponse_dict['rt']))
        
        # fit mem reg based on taskstats evquery (and save in self)
        S = Subject(self.subj)
        self.memReg_dict = S.fitMemoryRegression(evQuery = self.pop_pow_evQuery,decay_model = 'best',print_results=False,slow_drift_sigma=5)

        #  do Stats here
        self.pop_doStats(n_iters = popStats_iters)


    def pop_getRoiList(self):
        # get list of anatomical labels
        anat_list = np.array(self.anat_list.copy())[self.isBadElectrode_list==False]
        anat_list_wm = np.array(self.anat_list_wm.copy())[self.isBadElectrode_list==False]

        mni_coords = np.zeros((len(anat_list),3))
        rois = [] 
        
        # drop laterality
        for a in np.arange(0,len(anat_list)):

            # parse anat_lbl
            anat_lbl = self.parse_anat_lbl(anat_list[a],anat_list_wm[a])                
            # convert to roi
            rois.append(self.anat2roi(anat_lbl))

            # collect mni coords
            mni_coords[a,0] = self.mni_x[a]
            mni_coords[a,1] = self.mni_y[a]
            mni_coords[a,2] = self.mni_z[a]
           
         
        # replace 'unlabelled' rois
        rois = self.parse_unabelled(rois,mni_coords)

        # get roi_list
        roi_list = np.unique(np.array(rois))


        # get mean mni x for each region 
        mni_y = []      
        for r in roi_list:
            mni_y.append(np.nanmean(np.array(self.mni_y[self.isBadElectrode_list==False])[r==np.array(rois)]))

        # arange roi_list by ant-post axis
        sort_idx = np.argsort(mni_y)
        roi_list = list(np.array(roi_list)[sort_idx])

        # make roi_idct 

        return rois,roi_list


    def pop_ev2pop(self, x_ev, ev_filt = None):
        # this function takes an array from events (trial order after events have been filtered) and rearranges it to match the order of trials in popMat
        #x_ev... a 1d array that consists of the oridignal ordier in reference to ev_filt
        #ev_filt...events that have been filtered based on self.pop_pow_evQuery. If None, will use self.popResponse_dict['pow_ev_filt'

        # get ev_filt
        if ev_filt is None:
            ev_filt = self.popResponse_dict['pow_ev_filt']


        # get short and long trial bool
        shortTrials_bool = ev_filt.eval('delay==500')
        longTrials_bool = ev_filt.eval('delay==1500')


        # init a container that should total teh number of short and long delay trials
        x_pop = np.zeros(np.sum(shortTrials_bool)+np.sum(longTrials_bool))
        x_pop[:] = np.nan

        # populate x_pop with short delay trials
        x_pop[np.arange(0,np.sum(shortTrials_bool))] = x_ev[shortTrials_bool]
        x_pop[np.arange(np.sum(shortTrials_bool),len(x_pop))] = x_ev[longTrials_bool]


        # returun
        return x_pop


    def pop_pop2ev(self, x_pop, ev_filt = None):
        # This function rearranges data from popResponse_dict and rearranges it into the original order to maintain autocorrelation structure. (it fills in nans if there are medium delay trials ie 1000)
        #x_pop... a 1d or 2d array that consists of short trials (delay == 500) followed by long trials (delay == 1500). If 2d, assumes that trials are the 0 dim. 
        #ev_filt...events that have been filtered based on self.pop_pow_evQuery. If None, will use self.popResponse_dict['pow_ev_filt']

        # get ev_filt
        if ev_filt is None:
            ev_filt = self.popResponse_dict['pow_ev_filt']


        # get short and long trial bool
        shortTrials_bool = ev_filt.eval('delay==500')
        longTrials_bool = ev_filt.eval('delay==1500')

        # parse dim
        if x_pop.ndim==1:

            # initialize vector
            x_ev = np.zeros(len(ev_filt))
            x_ev[:] = np.nan

            # populate it with short trials 
            x_ev[shortTrials_bool] = x_pop[np.arange(0,np.sum(shortTrials_bool))]
            x_ev[longTrials_bool] = x_pop[np.arange(np.sum(shortTrials_bool),len(x_pop))]

        elif x_pop.ndim==2:
            # initialize 2d array
            x_ev = np.zeros((len(ev_filt),np.shape(x_pop)[1]))
            x_ev[:] = np.nan

            # populate it with short trials 
            x_ev[shortTrials_bool,:] = x_pop[np.arange(0,np.sum(shortTrials_bool)),:]
            x_ev[longTrials_bool,:] = x_pop[np.arange(np.sum(shortTrials_bool),len(x_pop)),:]
        else:
            raise NameError('x_pop must be either 1 dim or 2 dimensional array')

        return x_ev

    def pop_mat2vec(self,popMat3d):
        #This function takes a 3d population response array (trials x time x electrodes) and returns a 2d array (time x electrodes) to be used for dimensionality reduction. Trials are concatenated to create a single time series 

        popMat2d = np.reshape(a = popMat3d,newshape=(np.shape(popMat3d)[0]*np.shape(popMat3d)[1],np.shape(popMat3d)[2]),order='C')

        return popMat2d 
    def pop_vec2mat(self,popMat2d,trial_len_samp):
        # this function takes a 2d array (time x electrodes/components) where trials have been concatenated together and splits it apart back into trial by trial data (trials x time x electrodes/components)

        # popMat2d ...2d array (time(all trials concatenated) x electrodes)
        # trial_len_samp... int. indicating number of samples for each trial (it uses this to identify trial start and stop).
        # compute num of trials
        n_trials = int(np.shape(popMat2d)[0]/trial_len_samp)

        # reshape (trials x time x components/electrodes)
        popMat3d = np.reshape(popMat2d,newshape = (n_trials,trial_len_samp,np.shape(popMat2d)[1]),order='C')

        return popMat3d

        # to check:
        # init popMat3d (trials x time x components/electrodes)
        #popMat3d = np.zeros((n_trials,trial_len_samp,np.shape(popMat2d)[1]))
        #popMat3d[:] = np.nan 

        # for i in np.arange(0,n_trials):
        #     samp_start = (((i+1)*trial_len_samp) - trial_len_samp) 
        #     samp_end = (i+1)*trial_len_samp


    def pop_fitPCA(self,popMat3d):
        # this function transforms a 3d population response array (trials x time x electrodes) to a 2d population response principal components (time (all trials concatenated) x principal components) into principal components. It performs dimensionality reduction across electrodes

        from sklearn.decomposition import PCA

        pca_mod = PCA()

        popMat2d_pc = pca_mod.fit_transform(self.pop_mat2vec(popMat3d))

        return popMat2d_pc, pca_mod

    #def pop_extract
    def pop_plotExplainedVar(self,pca_mod,ax = None,num_dim_to_eval=None):

        if ax is None:
            f = plt.figure()
            ax = plt.subplot(111)

        if num_dim_to_eval is None:
            num_dim_to_eval = np.shape(self.popResponse_dict['popMat'])[2]


        x = np.arange(0,num_dim_to_eval)+1
        y = np.cumsum(pca_mod.explained_variance_ratio_[0:num_dim_to_eval])
        ax.plot(x,y,linewidth=3,color = '0.5')
        ax.set_xlabel('Number of components')
        ax.set_ylabel('% variance explained')
    def pop_getTrajParams(self,popMat3d,rts_ms,targOn_offset_samp = None, fixed_dur_ms = 200):
        # this function takes popMat3d (3d array, trials x time x electrodes (ordered as short delay trials, then long delay trials) and extracts various trial-by-trial measures of neural dynamics between stimulus and response across the given electrodes. rt_ms is array of rts in ms (ordered as short delay trials, then long delay trials).  targOn_offset_samp is sample when stimulus is presented (assumes 0 by default). fixed_dur_ms is the fixed length of time to use for analyses below

        # computes the following measures of s-r dynamics (stimulus-response dynamics)
        #SR_distance: euclidean distance between stim and response 
        #SR_speed: mean (change in s-r distance per time step). Are you making big jumps or small jumps? How autocorrelated is brain state?
        #SR_headingCorrect: Index of whether we are moving towards response or away from response.  with t=0 being stimulus onset and t=1 being fixed_dur_ms ms, compute s-r distance (t=0) / s-r distance (t=1). Are we moving in the right direction? > 1 indicates we are moving towards response, < 1 indicates we're moving away from response
        #SR_var: variability in brain states between s-r. 
        #SR_var_ps: variability in brain states during a short, fixed segment after stimulus is presented (fixed_dur_ms)
        #SR_var_pr: variability in brain states during a short, fixed segment prior to resposnse (fixed_dur_ms)

        from sklearn.metrics import pairwise_distances

        if targOn_offset_samp is None:
            targOn_offset_samp = 0


        # get S0_mat array representing starting points in multi-dimensional space (each dimension is an electrode)
        # s0 mat is trials x electrodes. Each row is brain state at stimulus presentation
        stimOn_samp = (0+targOn_offset_samp)
        S0_mat = popMat3d[:,stimOn_samp,:]

        # rts in samp
        rts_samp = self.ms_to_samples(rts_ms)

        # fixed_dur in samp 
        fixed_dur_samp = int(self.ms_to_samples(fixed_dur_ms))

        # init St_mat (brain state at time of response)
        St_mat = np.zeros(S0_mat.shape)
        St_mat[:] = np.nan



        # init St_mat: response (trials x electrodes)
        St_mat = np.zeros(S0_mat.shape)
        St_mat[:] = np.nan

        # SR_dist distance from starting point to response point
        SR_dist = np.zeros(np.shape(popMat3d)[0])
        SR_dist[:] = np.nan

        # SR_speed - mean change in state per time step
        SR_speed = np.zeros(np.shape(popMat3d)[0])
        SR_speed[:] = np.nan

        # SR_var- brain state variability per trial
        SR_var = np.zeros(np.shape(popMat3d)[0])
        SR_var[:] = np.nan

        # SR_var_ps= (post stim) brain state variability per trial (from stim onset to fixed_dur ms)
        SR_var_ps = np.zeros(np.shape(popMat3d)[0])
        SR_var_ps[:] = np.nan

        # SR_var_pr - (pre response) brain state variability per trial (from fixed_dur ms to response)
        SR_var_pr= np.zeros(np.shape(popMat3d)[0])
        SR_var_pr[:] = np.nan

        # SR_headingCorrect- index of whether we are moving towards 
        SR_headingCorrect = np.zeros(np.shape(popMat3d)[0])
        SR_headingCorrect[:] = np.nan

        # loop through trials
        for t in np.arange(0,np.shape(popMat3d)[0]):

            # populate St_mat based on rt
            # (0+targOn_offset_samp) is the targ on sample
            rt_samp_thisTrial = stimOn_samp+ int(rts_samp[t])
            St_mat[t,:] = popMat3d[t,rt_samp_thisTrial,:]

            # get brain state matrix for this trial
            # popMat_thisTrial (n samples x n electrodes) ~(n_obs x n_features)
            popMat_thisTrial = popMat3d[t,stimOn_samp:rt_samp_thisTrial,:]

            # calculate pairwise distances for this trial trajectory
            # dist_thisTrial (i,j) corresponding to distance between brain state at time samples i and j). Starts at stim on and ends at response
            dist_thisTrial = pairwise_distances(popMat_thisTrial,metric='euclidean')

            # calc SR distance (top row dist_thisTrial gives pairwise distances between S0 brain state and all other time points. Use last index to get distance between brain state at stimulus and response)
            SR_dist[t] = dist_thisTrial[0,-1]

            # calc SR_speed (mean distance between t and t+1)
            t_idx = np.arange(0,np.shape(dist_thisTrial)[0]-1)
            SR_speed[t] = np.nanmean(dist_thisTrial[t_idx,t_idx+1])

            # calculate SR_variability (mean distance from response point, which we are using as our fixed point. Alternatively, use stim as reference point).
            # bottom row of dist_thisTrial gives pairwise distances between St brain state and all other time points
            SR_var[t] = np.nanmean(dist_thisTrial[-1,:])

            # calculate SR_variability (post stim)
            # top row of dist_thisTrial gives pairwise distances between S0 brain state and all other time points. Just take the mean distance for fixed_dur_samp
            SR_var_ps[t] = np.nanmean(dist_thisTrial[0,:fixed_dur_samp])

            # calculate SR_variability (post stim)
            # bottom row of dist_thisTrial gives pairwise distances between St brain state and all other time points. Just take the mean distance for fixed_dur_samp prior to response
            pre_resp_samp_start = (np.shape(dist_thisTrial)[0]-fixed_dur_samp)
            SR_var_pr[t] = np.nanmean(dist_thisTrial[-1,pre_resp_samp_start:])

            # calculate heading index (s-r distance at stim onset/s-r distance at fixed dur samp). > 1 indicates we are moving towards response, < 1 indicates we're moving away from response
            SR_headingCorrect[t] = dist_thisTrial[0,-1]/dist_thisTrial[fixed_dur_samp,-1]
  
        # SR_rate ...SR_distance/response time in seconds
        SR_rate = SR_dist/(rts_ms/1000)


        # return dict
        trajParam_dict = {}
        trajParam_dict['SR_dist'] = SR_dist
        trajParam_dict['SR_rate'] = SR_rate
        trajParam_dict['SR_speed'] = SR_speed
        trajParam_dict['SR_var'] = SR_var
        trajParam_dict['SR_var_ps'] = SR_var_ps
        trajParam_dict['SR_var_pr'] = SR_var_pr
        trajParam_dict['SR_headingCorrect'] = SR_headingCorrect

        return trajParam_dict

    def pop_getPCAParams(self,popMat3d_pc,rts_ms,targOn_offset_samp = None):
        # this function gets population-activity parameters (trial by trial measures of starting point and response threshold in PCA coordinates)
        if targOn_offset_samp is None:
            targOn_offset_samp = 0


        # get S0_mat array representing starting points in multi-dimensional space (each dimension is a principal component value) across all trials
        #(trials x component vaues)
        S0_pc = popMat3d_pc[:,0+targOn_offset_samp,:]

        # init St_mat
        St_pc = np.zeros(S0_pc.shape)
        St_pc[:] = np.nan

        # rts in samp
        rts_samp = self.ms_to_samples(rts_ms)

        # loop through trials
        for t in np.arange(0,np.shape(popMat3d_pc)[0]):

            # populate St_mat based on rt
            # (0+targOn_offset_samp) is the targ on sample
            rt_samp_thisTrial = (0+targOn_offset_samp)+ int(rts_samp[t])
            St_pc[t,:] = popMat3d_pc[t,rt_samp_thisTrial,:]


        # return dict
        pcaParam_dict = {}
        pcaParam_dict['S0_pc'] = S0_pc
        pcaParam_dict['St_pc'] = St_pc

        return pcaParam_dict
    def pop_doStats(self,n_iters = 1000,overwriteFlag=False):
        # this function runs all stats relating trajectory params and PCA coords to various behavioral variables


        # look for saved file
        self.popStats_fname = self.popResponse_fname+'-'+'popStats-'+self.filter_bool_lbl+'-'+str(n_iters)
        if (os.path.exists(self.params_dir+self.popStats_fname)==True)&(overwriteFlag==False):
            
            #load file if it exists
            self.popStats_dict = (self.load_pickle(self.params_dir+
                                                 self.popStats_fname))

            return self.popStats_dict


        # subfunctions:
        # get beh data (option to shuffle)
        def getBehData(apply_circ_shift=False):
            beh_feat_dict = {}

            # list of behavioral data to analyze
            beh_var_list = ['rt_ms','zrrt','l_idx']
            # additional:
            #,'zrrt_resid_fast','zrrt_resid_slow','zrrt_pred','errorMemFast','shortDelayMem'

            # collect behavioral data from taskstats
            if apply_circ_shift == True:
                # randomly generate an amount to circularly shift data (constant shift across all behavioral variables)
                shift_idx = np.random.randint(low=0, high=len(self.popResponse_dict['pow_ev_filt']['RT'].to_numpy()))

            for f in beh_var_list:
                # get beh features (we will ignore s_idx and l_idx for our regression analysis, but use them to estimate null distributions for delay-related differences)
                if f == 'zrrt':
                    x = stats.zscore(-1/self.popResponse_dict['pow_ev_filt']['RT'].to_numpy())
                elif f == 'rt_ms':
                    x = self.popResponse_dict['pow_ev_filt']['RT'].to_numpy()
                elif f == 'l_idx':
                    x = self.popResponse_dict['pow_ev_filt'].eval('delay==1500').to_numpy()
                else:
                    x = np.copy(self.memReg_dict[f])

                    # z-score (to help interpret parameters)
                    x = (x - np.nanmean(x))/np.nanstd(x)

                # apply circular shift (using common shift_idx computed above)
                if apply_circ_shift == True:
                    # circ shift
                    x = np.roll(x,shift_idx)

                # update beh_feat_dict
                beh_feat_dict[f] = x 

            # keep track of whether we shuffled or not
            beh_feat_dict['is_null'] = apply_circ_shift




            return beh_feat_dict

        # get true stats (option to return light for shuffle)
        def getStatsDict(beh_feat_dict):
            def updateDict_w_reg(stats_dict,reg,beh_var_lbl,neu_var_lbl,d_str):
                stats_dict['popReg'+d_str+'_'+beh_var_lbl+'_'+neu_var_lbl+'_B1'] = reg.params.loc[neu_var_lbl]
                stats_dict['popReg'+d_str+'_'+beh_var_lbl+'_'+neu_var_lbl+'_tstat'] = reg.tvalues.loc[neu_var_lbl]
                stats_dict['popReg'+d_str+'_'+beh_var_lbl+'_'+neu_var_lbl+'_pval'] = reg.pvalues.loc[neu_var_lbl]
                return stats_dict

            def updateDict_w_corr(stats_dict,rval,pval,corrstats,beh_var_lbl,neu_var_lbl,d_str):
                stats_dict['popCorr'+d_str+'_'+beh_var_lbl+'_'+neu_var_lbl+'_rval'] = rval
                stats_dict['popCorr'+d_str+'_'+beh_var_lbl+'_'+neu_var_lbl+'_pval'] = pval
                stats_dict['popCorrPartial'+d_str+'_'+beh_var_lbl+'_'+neu_var_lbl+'_rval'] = corrstats.iloc[0]['r']
                stats_dict['popCorrPartial'+d_str+'_'+beh_var_lbl+'_'+neu_var_lbl+'_pval'] = corrstats.iloc[0]['p-val']

                return stats_dict

            def runRegCorr(data_df,stats_dict,neu_var_lbl,beh_var_lbl,covar_list,d_str):

                # run regression
                reg = smf.ols(beh_var_lbl+ ' ~ '+ neu_var_lbl, data = data_df).fit()
                # update stats dict with regression
                stats_dict = updateDict_w_reg(stats_dict,reg,beh_var_lbl=beh_var_lbl,neu_var_lbl=neu_var_lbl,d_str=d_str)

                # run spearman correlation
                rval, pval = stats.spearmanr(data_df[neu_var_lbl],data_df[beh_var_lbl],nan_policy='omit')

                # partial correlation
                corrstats = pg.partial_corr(data = data_df, x=neu_var_lbl,y =beh_var_lbl,covar = covar_list,method='spearman')
               
                # update stats dict with corr
                stats_dict = updateDict_w_corr(stats_dict,rval,pval,corrstats,beh_var_lbl,neu_var_lbl,d_str=d_str)      

                return stats_dict

            # init dict to hold results
            stats_dict = {}

            # list of behavioral features
            stats_dict['beh_feat_list'] = list(beh_feat_dict.keys())  

            #remove  l_idx so we do not perform a regression on them
            stats_dict['beh_feat_list'].remove('l_idx')
            stats_dict['beh_feat_list'].remove('is_null')

            # create data_dict (for regression), and separate dictionaries for short delay trials and long delay trials
            data_dict ={}
            dataS_dict = {}
            dataL_dict = {}
            
            #update with behavioral data (potentially circularly shifted)
            data_dict.update(beh_feat_dict)

    
            # populate by delay condition. It is important to use the unshuffled indices of short and long delay trials here so that you assign shuffled RT data to short and long blocks
            s_idx = self.popResponse_dict['pow_ev_filt'].eval('delay==500').to_numpy()
            l_idx = self.popResponse_dict['pow_ev_filt'].eval('delay==1500').to_numpy()
            for f in stats_dict['beh_feat_list']:
                dataS_dict[f] = beh_feat_dict[f][s_idx]
                dataL_dict[f] = beh_feat_dict[f][l_idx]


            #list of neural features we care about for stats
            feat_to_copy = ['SR_dist', 'SR_rate','SR_speed','SR_var','SR_headingCorrect']#,'S0_pc','St_pc'


            # # we have to recompute S-R distance and S-R rate using the provided rt_ms that are potentially shuffled. This ensures that  the null stats we perform below should capture any intrinsic relationship one might observe between distance and time)

            #NOTE: we run self.pop_ev2pop because getTrajParams function expects that rts are ordered in relation to the popMat (short trials, followed by long trials, rather than the original trial order during the experiment)
            thisTrajParams_dict = self.pop_getTrajParams(popMat3d=self.popResponse_dict['popMat'],rts_ms=self.pop_ev2pop(beh_feat_dict['rt_ms']))

            # update neural feature list (and beh feature list)
            stats_dict['neu_feat_list'] = feat_to_copy
            
            #copy neural features into stats dict, seprately for short and long delay trials
            # NOTE: use thisTrajParams_dict instead of self.popResponse_dict to use recomputed values
            # NOTE: we are changing order of trials back from from short delay/long delay to original events order

            for f in feat_to_copy:
                data_dict[f] = self.pop_pop2ev(thisTrajParams_dict[f])
                dataS_dict[f] = self.pop_pop2ev(thisTrajParams_dict[f])[s_idx]
                dataL_dict[f] = self.pop_pop2ev(thisTrajParams_dict[f])[l_idx]
            

            #compare values of neural features between delay condition. If this is shuffled behavioral data, we are going to randomly generate long and short trial. we are going to use the shuffled l_idx in beh_feat_dict to select long and short delay trials so that we can estimate a null distribution of these differences
            
            # parse whether we are dealing with shuffled behavioral data and assign condition labels accordingly
            if beh_feat_dict['is_null'] == False:

                # cond labels for difference measures
                cond_a = l_idx
                cond_b = (l_idx==False)

                # classification labels (of long delay trials) for classification analysis. Should be in popResponse order
                y_true = self.pop_ev2pop(l_idx).astype('bool') 
                y_true[np.isnan(y_true)]=False

            elif beh_feat_dict['is_null'] == True:

                # get a copy of l_idx in standard order and pop order
                shuf_l_idx = np.copy(l_idx)

                # classification labels
                shuf_l_idx_pop = np.copy(self.pop_ev2pop(shuf_l_idx).astype('bool'))
                shuf_l_idx_pop[np.isnan(shuf_l_idx_pop)]=False

                # shuffle these in place
                np.random.shuffle(shuf_l_idx)
                np.random.shuffle(shuf_l_idx_pop)

                # assign shuffled condition labels                
                cond_a = shuf_l_idx
                cond_b = (shuf_l_idx==False)

                # assign classification labels
                y_true = shuf_l_idx_pop


            for f in feat_to_copy:

                # take difference in mean between neural parameter in long and short condition 
                stats_dict['popByDelay_'+f+'Mean_diff'] = np.nanmean(data_dict[f][cond_a]) - np.nanmean(data_dict[f][cond_b])

                # take difference in std neural parameter in long and short condition 
                stats_dict['popByDelay_'+f+'Std_diff'] = np.nanstd(data_dict[f][cond_a]) - np.nanstd(data_dict[f][cond_b])
       
                # # first, do an idndependent t-test comparing distributions of neural features assuming unequal variance
                # tstat, pval = stats.ttest_ind(data_dict[f][cond_a],data_dict[f][cond_b],equal_var=False,nan_policy='omit')
                # stats_dict['popByDelay_'+f+'_tstat'] = tstat
                # stats_dict['popByDelay_'+f+'_pval'] = pval


            # compare discriminability between S0 brain states in each delay condition:


            # use y_true calculated above as classification labels.  (use shuffled indices here so that we can approximate a null distribution of this discriminability measure for this subject). Also, it should have been converted to to popResponse_order
            #y_true
            #l_idx_pop = self.pop_ev2pop(beh_feat_dict['l_idx']).astype('bool')

            # if any nans, replace with 0
            #l_idx_pop[np.isnan(l_idx_pop)]=False

            # get brain state representation at stimulus onset for short and long delay trials trials. popMat is trials x time x electrodes. Assume that popMat begins with stimulus onset (can set this with targOn_samp if needed). Resulting S0 is n trials x n electrodes (~n_obs x n_features)
            targOn_samp = 0
            S0 = self.popResponse_dict['popMat'][:,targOn_samp,:]

            # get brain state representation at time of response (using unshuffled RTs). Only the class labels (short delay vs. long delay) get shuffled. So the null distribution will tell us how distinguishable response brain states should expected to be by chance. St is n trials x n electrodes
            # rts in samp (unshuffled rts in popResponse order)
            rts_samp = self.ms_to_samples(self.popResponse_dict['rt'])

            # init St (brain state at time of response)
            St = np.zeros(S0.shape)
            St[:] = np.nan

            # loop through trials
            for t in np.arange(0,len(rts_samp)):
                rt_samp_thisTrial = targOn_samp+ int(rts_samp[t])
                St[t,:] = self.popResponse_dict['popMat'][t,rt_samp_thisTrial,:]

            #implement classifier here to distinguish short delay and long delay trials (indexed by l_idx_pop) using brain state representation at stimulus onset (S0).

            #Build two classifiers using the following global params:
            n_folds = 10
            scoring = 'roc_auc'
            #scoring = 'accuracy'

            #(1) How distinguishable are S0 brain states between delay conditions?.....Logistic regression without regularization. Use stratified 10 fold cross validation. Scoring done with roc_auc.predict long/short trial indices (possibly shuffled). 
            # initialize classifier. Dont need to fit it because cross_val_score will fit
            clf = LogisticRegression(random_state=0,penalty='none')

            stats_dict['popByDelay_S0_diff'] = np.nanmean(cross_val_score(estimator = clf,X = S0, y = y_true, cv = n_folds, scoring = scoring))

            #(2) How distinguishable are St brain states between delay conditions?.....Logistic regression without regularization. Use stratified 10 fold cross validation. Scoring done with roc_auc.predict long/short trial indices (possibly shuffled). 
            # initialize classifier. Dont need to fit it because cross_val_score will fit
            clf = LogisticRegression(random_state=0,penalty='none')

            stats_dict['popByDelay_St_diff'] = np.nanmean(cross_val_score(estimator = clf,X = St, y = y_true, cv = n_folds, scoring = scoring))



            #(3) How distinguishable are S0 brain states between delay conditions?.....Logistic regression WITH regularization. l2 penalty, standard lbgs solver, sociring via roc_auc.  Tune C (model complexity) over 10 log-spaced values, and report cross-validated roc_auc via 10 fold cross validation
            # initialize classifier and fit
            clfReg = LogisticRegressionCV(random_state=0,Cs=10,penalty='l2',scoring=scoring,cv=n_folds).fit(X=S0,y = y_true)

            # calculate average performance across the folds. scores_[True] is a 2d array (n_folds x n_cs). 
            avg_scores = np.mean(clfReg.scores_[True],axis=0)

            # summary stat is the maximum score across the different values of the c parameter
            stats_dict['popByDelay_S0reg_diff'] = np.max(avg_scores)

            #(4) How distinguishable are St brain states between delay conditions?..........Logistic regression WITH regularization. l2 penalty, standard lbgs solver, sociring via roc_auc.  Tune C (model complexity) over 10 log-spaced values, and report cross-validated roc_auc via 10 fold cross validation
            clfReg = LogisticRegressionCV(random_state=0,Cs=10,penalty='l2',scoring=scoring,cv=n_folds).fit(X=St,y = y_true)

            # calculate average performance across the folds. scores_[True] is a 2d array (n_folds x n_cs). 
            avg_scores = np.mean(clfReg.scores_[True],axis=0)

            # summary stat is the maximum score across the different values of the c parameter
            stats_dict['popByDelay_Streg_diff'] = np.max(avg_scores)

            # compare variability within S0 brain states and St brain states between delay conditions:

            # calculate reference points for S0 and St brain states within each delay condition. Use y_true (long idx in popMat order)calculated above that are potentially shuffled
            # S0 and St are n trials x n electrodes. Resulting reference values are 1 x n_electrodes
            S0_refL = np.nanmean(S0[y_true,:],axis=0,keepdims=True)
            S0_refS = np.nanmean(S0[y_true==False,:],axis=0,keepdims=True)

            St_refL = np.nanmean(St[y_true,:],axis=0,keepdims=True)
            St_refS = np.nanmean(St[y_true==False,:],axis=0,keepdims=True)

            # calculate S0 and St variability within delay condition (mean pairwise distance between each tiral and reference point)
            S0_varL = np.nanmean(pairwise_distances(S0[y_true,:],S0_refL,metric='euclidean'))
            S0_varS = np.nanmean(pairwise_distances(S0[y_true==False,:],S0_refS,metric='euclidean'))

            St_varL = np.nanmean(pairwise_distances(St[y_true,:],St_refL,metric='euclidean'))
            St_varS = np.nanmean(pairwise_distances(St[y_true==False,:],St_refS,metric='euclidean'))

            # store the difference in variance within delay condition
            stats_dict['popByDelay_S0var_diff'] = S0_varL - S0_varS
            stats_dict['popByDelay_Stvar_diff'] = St_varL - St_varS

            # calculate dimensionality of S0 and St in each delay condition using PCA
            #set threshold
            var_thresh = 0.95
            
            #init models
            pca_mod_s0L = PCA()
            pca_mod_s0S = PCA()
            pca_mod_stL = PCA()
            pca_mod_stS = PCA()

            # fit models and get num of dimensions needed to explain 95% of variance

            S0_dimL = np.nonzero(np.cumsum(pca_mod_s0L.fit(S0[y_true,:]).explained_variance_ratio_)>var_thresh)[0][0]
            S0_dimS = np.nonzero(np.cumsum(pca_mod_s0S.fit(S0[y_true==False,:]).explained_variance_ratio_)>var_thresh)[0][0]

            St_dimL = np.nonzero(np.cumsum(pca_mod_stL.fit(St[y_true,:]).explained_variance_ratio_)>var_thresh)[0][0]
            St_dimS = np.nonzero(np.cumsum(pca_mod_stS.fit(St[y_true==False,:]).explained_variance_ratio_)>var_thresh)[0][0]


            # store the difference in variance across electrodes within delay condition
            stats_dict['popByDelay_S0dim_diff'] = S0_dimL - S0_dimS
            stats_dict['popByDelay_Stdim_diff'] = St_dimL - St_dimS


            # convert to data frame (for pingouin spearman corr)
            data_df = pd.DataFrame.from_dict(data_dict)
            dataS_df = pd.DataFrame.from_dict(dataS_dict)
            dataL_df = pd.DataFrame.from_dict(dataL_dict)


            # loop through behavioral variables (we have removed s_idx and l_idx from beh feat list)

            for beh_var_lbl in stats_dict['beh_feat_list']:

                # run stats (regression, corr, partial corr). Separately for short and long delay trials
                ##### SR_distance ...Co-variates include all speed and heading dir
                # short delay trials
          
                stats_dict = runRegCorr(dataS_df,stats_dict,neu_var_lbl='SR_dist',beh_var_lbl=beh_var_lbl,covar_list=['SR_speed','SR_headingCorrect'],d_str = 'S')

                # long delay trials
                stats_dict = runRegCorr(dataL_df,stats_dict,neu_var_lbl='SR_dist',beh_var_lbl=beh_var_lbl,covar_list=['SR_speed','SR_headingCorrect'],d_str = 'L')


                ##### SR_speed ...Co-variates includes dist

                # short delay trials
                stats_dict = runRegCorr(dataS_df,stats_dict,neu_var_lbl='SR_speed',beh_var_lbl=beh_var_lbl,covar_list=['SR_dist'],d_str = 'S')

                # long delay trials
                stats_dict = runRegCorr(dataL_df,stats_dict,neu_var_lbl='SR_speed',beh_var_lbl=beh_var_lbl,covar_list=['SR_dist'],d_str = 'L')

                ##### SR_headingCorrect ...Co-variates includes dist
                 # short delay trials
                stats_dict = runRegCorr(dataS_df,stats_dict,neu_var_lbl='SR_headingCorrect',beh_var_lbl=beh_var_lbl,covar_list=['SR_dist'],d_str = 'S')

                # long delay trials
                stats_dict = runRegCorr(dataL_df,stats_dict,neu_var_lbl='SR_headingCorrect',beh_var_lbl=beh_var_lbl,covar_list=['SR_dist'],d_str = 'L')


                # Other parameters (not calculating now):


                # is beh_var related to SR_rate? Co-variates includes dist
                # all trials
                #stats_dict = runRegCorr(data_df,stats_dict,neu_var_lbl='SR_rate',beh_var_lbl=beh_var_lbl,covar_list=['SR_dist'],d_str = '')

                # # short delay trials
                # stats_dict = runRegCorr(dataS_df,stats_dict,neu_var_lbl='SR_rate',beh_var_lbl=beh_var_lbl,covar_list=['SR_dist'],d_str = 'S')

                # # long delay trials
                # stats_dict = runRegCorr(dataL_df,stats_dict,neu_var_lbl='SR_rate',beh_var_lbl=beh_var_lbl,covar_list=['SR_dist'],d_str = 'L')


                # is beh_var related to SR_var? Co-variates includes dist
                # all trials
                #stats_dict = runRegCorr(data_df,stats_dict,neu_var_lbl='SR_var',beh_var_lbl=beh_var_lbl,covar_list=['SR_dist',],d_str = '')

                # # short delay trials
                # stats_dict = runRegCorr(dataS_df,stats_dict,neu_var_lbl='SR_var',beh_var_lbl=beh_var_lbl,covar_list=['SR_dist'],d_str = 'S')

                # # long delay trials
                # stats_dict = runRegCorr(dataL_df,stats_dict,neu_var_lbl='SR_var',beh_var_lbl=beh_var_lbl,covar_list=['SR_dist'],d_str = 'L')




                # # can we predict beh_var based on starting location (not including delayCondIsLong?
                # neu_var_lbl = 'S0_pc'
                # reg = sm.regression.linear_model.OLS(endog = data_dict[beh_var_lbl],exog = data_dict[neu_var_lbl],hasconst=False,missing='drop').fit()
                # stats_dict['popReg_'+beh_var_lbl+'_'+neu_var_lbl+'_SSE'] = reg.ssr

                # # can we predict beh_var based on response location?
                # neu_var_lbl = 'St_pc'
                # reg = sm.regression.linear_model.OLS(endog = data_dict[beh_var_lbl],exog = data_dict[neu_var_lbl],hasconst=False,missing='drop').fit()
                # stats_dict['popReg_'+beh_var_lbl+'_'+neu_var_lbl+'_SSE'] = reg.ssr

            return stats_dict

        # update p-vals
        def update_pvals(stats_dict_true,stats_null_df):
            # updates true stats dict with non-parametric p-values based on circular shift of that procedure. generate pval_np fields ("non parametric")

            def get_p_two_tailed(null_vals,true_val):

                # parse true_val
                # we are using median null value here in case the null distribution is not symmetric about zero
                if true_val>=np.median(null_vals): #0
                    # probability of observing a null value greater than true value
                    pval_np = np.count_nonzero(null_vals >= true_val)/len(null_vals)
                else:
                    # prob of observing a null value less than true value
                    pval_np = np.count_nonzero(null_vals <= true_val)/len(null_vals)
                # multiply by 2 to make it a two-tailed p-value (ceiling value = 1)
                pval_np = np.min((1,pval_np * 2))

                # get 95% confidence intervals
                ci_neg,ci_pos = np.percentile(null_vals,[2.5,97.5]) 

                #zscore of the effect
                zstat_np = (true_val - np.mean(null_vals))/np.std(null_vals)

                return pval_np,ci_pos,ci_neg,zstat_np

            # loop through fields in stats dict true and update with non param p values and confidence intervals
            key_list = list(stats_dict_true.keys())
            for k in key_list:

                if 'SSE' in k:
                    # do we have more predictive power than expected by chance? (one-tailed pvalue for sum squared error)
                    null_sse = stats_null_df[k].to_numpy()
                    true_sse = stats_dict_true[k]

                    # calc false pos rate (prob of observing a null sse that is smaller than or equal to true sse))
                    pval_np = np.count_nonzero(null_sse <= true_sse)/len(null_sse)
                    stats_dict_true[k+'_pvalnp'] = pval_np
                    ci_neg,ci_pos = np.percentile(null_sse,[2.5,97.5]) 
                    stats_dict_true[k+'_cipos'] = ci_pos
                    stats_dict_true[k+'_cineg'] = ci_neg
                elif 'tstat' in k:
                    null_vals= stats_null_df[k].to_numpy()
                    stats_dict_true[k.split('_tstat')[0]+'_pvalnp'],stats_dict_true[k.split('_tstat')[0]+'_cipos'],stats_dict_true[k.split('_tstat')[0]+'_cineg'],stats_dict_true[k.split('_tstat')[0]+'_zstatnp'] = get_p_two_tailed(null_vals= null_vals, true_val= stats_dict_true[k])

                    # store null dist
                    #stats_dict_true[k+'_nullDist'] = null_vals

                # 'rval' is for spearman corr
                elif 'rval' in k:

                    null_vals= stats_null_df[k].to_numpy()
                    stats_dict_true[k.split('_rval')[0]+'_pvalnp'],stats_dict_true[k.split('_rval')[0]+'_cipos'],stats_dict_true[k.split('_rval')[0]+'_cineg'],stats_dict_true[k.split('_rval')[0]+'_zstatnp']  = get_p_two_tailed(null_vals= null_vals, true_val= stats_dict_true[k])
                    # store null dist
                    #stats_dict_true[k+'_nullDist'] = null_vals
                # 'byDelay' differences terms
                elif 'diff' in k:
                    null_vals= stats_null_df[k].to_numpy()
                    stats_dict_true[k.split('_diff')[0]+'_pvalnp'],stats_dict_true[k.split('_diff')[0]+'_cipos'],stats_dict_true[k.split('_diff')[0]+'_cineg'],stats_dict_true[k.split('_diff')[0]+'_zstatnp']  = get_p_two_tailed(null_vals= null_vals, true_val= stats_dict_true[k])

                    stats_dict_true[k+'_nullDist'] = null_vals


            return stats_dict_true


        # do stuff:
        # get true data
        beh_dict_true = getBehData(apply_circ_shift=False)
        stats_dict_true = getStatsDict(beh_dict_true)

        # loop through n_iters and shuffle
        # run through circular shifts
        stats_dict_null_list = []
        for i in np.arange(n_iters):

            # get null stats using circularly shifted behavioral data
            beh_dict_null = getBehData(apply_circ_shift=True)
            stats_dict_null = getStatsDict(beh_dict_null)

            # update list 
            stats_dict_null_list.append(stats_dict_null)

            print(i,'/',n_iters)

        # get nullStats_df
        stats_null_df = pd.DataFrame(stats_dict_null_list)

        # update stats dict with non-parametric p-values
        stats_dict_true = update_pvals(stats_dict_true,stats_null_df)

        # update self with popStats_dict
        self.popStats_dict = stats_dict_true

        #save pickle
        self.save_pickle(obj=self.popStats_dict,fpath=self.params_dir+self.popStats_fname)

        # return popStats_dict
        return stats_dict_true


    def pop_doStats_by_region(self,beh_var_lbl ='zrrt',min_elec_thresh = 5,master_roi_list = None,neu_var_list=['SR_dist','SR_speed','SR_headingCorrect'],stat_option='corrPartial',covar_list=[['SR_speed','SR_headingCorrect'],['SR_dist'],['SR_dist']],do_cumulative=False,do_reverse=False):
        # this function calculates relation between behavioral variable (e.g zrrt) and dynamics (trajectory parameters) separately for each brain region  (electrodes grouped by similar activation funcitons)
        # master_roi_list is a list of regions that it loops through. None uses the list below. if do_cumulative == True, it will iteratively add data from each region to the population data. If do_reverse == True (it works backwards through the list when doing the iterative calculation)

        if master_roi_list is None:
            master_roi_list = ['Occipital','Parietal','Temporal','ILF-MLF WM','IFOF WM','MTL','Perirolandic-CST','SLF WM','Cingulate','Striatum','Arc/Unc Fasiculus','Insula','Thalamocortical WM','Frontal','Lateral Prefrontal','Medial Prefrontal']

        # get list of regions
        rois,roi_list = self.pop_getRoiList()

        # get rt_ms (to compute traj params), and short and long trial indices
        rt_ms = self.popResponse_dict['pow_ev_filt']['RT'].to_numpy()
        s_idx = self.popResponse_dict['pow_ev_filt'].eval('delay==500').to_numpy()
        l_idx = self.popResponse_dict['pow_ev_filt'].eval('delay==1500').to_numpy()


        # get beh_var (x)
        if beh_var_lbl == 'zrrt':
            x = stats.zscore(-1/self.popResponse_dict['pow_ev_filt']['RT'].to_numpy())
        else:
            x = np.copy(self.memReg_dict[f])
            # z-score (to help interpret parameters)
            x = (x - np.nanmean(x))/np.nanstd(x)

        # make a dict
        data_dictS = {}
        data_dictS[beh_var_lbl] = x[s_idx]

        data_dictL = {}
        data_dictL[beh_var_lbl] = x[l_idx]

        # containers
        stats_dict = {}

        # parse labels
        if do_cumulative==True:
            cum_str = 'Cum'
        else:
            cum_str = ''

        if do_reverse==True:
            rev_str = 'Rev'
        else:
            rev_str = ''

        s_lbl = 'byRegion'+cum_str+rev_str+'StatS-'+beh_var_lbl+'-'
        l_lbl = 'byRegion'+cum_str+rev_str+'StatL-'+beh_var_lbl+'-'

        for n in neu_var_list:
            # initialize containers for long and short trials
            stats_dict[s_lbl+n]=np.zeros(len(master_roi_list))
            stats_dict[l_lbl+n]=np.zeros(len(master_roi_list))

        stats_dict['byRegion'+cum_str+rev_str+'_n_in_reg']=np.zeros(len(master_roi_list))

        # parse do_reverse
        if do_reverse==True:
            master_roi_list = list(np.flip(master_roi_list))

        # loop through regions
        i=-1
        for r in master_roi_list:
            i+=1
            stats_dict['byRegion'+cum_str+rev_str+'_n_in_reg'][i] = np.sum(np.array(rois)==r)
            if stats_dict['byRegion'+cum_str+rev_str+'_n_in_reg'][i]<min_elec_thresh:
                for n in neu_var_list:
                    stats_dict[s_lbl+n][i]=np.nan
                    stats_dict[l_lbl+n][i]=np.nan
            else:

                # get ret_idx
                if do_cumulative==False:
                    # only include this region
                    ret_idx = np.array(rois) == r
                else:
                    # include electrodes in all regions from the start of master_roi_list to this region (inclusive)
                    ret_idx = np.in1d(np.array(rois),master_roi_list[:master_roi_list.index(r)+1])

                # get popMat
                popMat3d_ret = self.popResponse_dict['popMat'][:,:,ret_idx]

                # get traj params using only electrodes from this region
                #NOTE: we run self.pop_ev2pop because getTrajParams function expects that rts are ordered in relation to the popMat (short trials, followed by long trials, rather than the original trial order during the experiment)
                thisTrajParams_dict = self.pop_getTrajParams(popMat3d=popMat3d_ret,rts_ms=self.pop_ev2pop(rt_ms))

                # loop through neural features
                n_count = -1
                for n in neu_var_list:
                    n_count+=1
                    # update data dict
                    data_dictS[n] = self.pop_pop2ev(thisTrajParams_dict[n])[s_idx]
                    data_dictL[n] = self.pop_pop2ev(thisTrajParams_dict[n])[l_idx]

                    # update data dict with covar list items also
                    for c in covar_list[n_count]:
                        data_dictS[c] = self.pop_pop2ev(thisTrajParams_dict[c])[s_idx]
                        data_dictL[c] = self.pop_pop2ev(thisTrajParams_dict[c])[l_idx]


                    # compute statistic
                    if stat_option=='tstat':
                        #run regression for short and long blocks
                        regS = smf.ols(beh_var_lbl+ ' ~ ' +n, data = data_dictS).fit()
                        regL = smf.ols(beh_var_lbl+ ' ~ ' +n, data = data_dictL).fit()

                        stats_dict[s_lbl+n][i] = regS.tvalues.loc[n]
                        stats_dict[l_lbl+n][i] = regL.tvalues.loc[n]

                    elif stat_option=='corr':
                        stats_dict[s_lbl+n][i],pval = stats.spearmanr(data_dictS[beh_var_lbl],data_dictS[n])
                        stats_dict[l_lbl+n][i],pval = stats.spearmanr(data_dictL[beh_var_lbl],data_dictL[n])

                    elif stat_option=='corrPartial':

                        #make it a dataframe for pingouin
                        data_dfS = pd.DataFrame.from_dict(data_dictS)
                        data_dfL = pd.DataFrame.from_dict(data_dictL)

                        # compute partial corr
                        corrstatsS = pg.partial_corr(data = data_dfS, x=n,y =beh_var_lbl,covar = covar_list[n_count],method='spearman')
                        corrstatsL = pg.partial_corr(data = data_dfL, x=n,y =beh_var_lbl,covar = covar_list[n_count],method='spearman')

                        # get stats
                        stats_dict[s_lbl+n][i] = corrstatsS.iloc[0]['r']
                        stats_dict[l_lbl+n][i] = corrstatsL.iloc[0]['r']
        # statsByRegion_dict

        stats_dict['byRegion'+cum_str+rev_str+'_stat_option'] =stat_option
        stats_dict['byRegion'+cum_str+rev_str+'_lbls'] = master_roi_list

        return stats_dict 

    def pop_doStats_by_clusLevel(self,clus_ret_mat,beh_var_lbl ='zrrt',min_elec_thresh = 5,master_roi_list = None,neu_var_list=['SR_dist','SR_speed','SR_headingCorrect'],stat_option='corrPartial',covar_list=[['SR_speed','SR_headingCorrect'],['SR_dist'],['SR_dist']],do_cumulative=False,do_reverse=False):
        # this function calculates relation between behavioral variable (e.g zrrt) and dynamics (trajectory parameters) separately for each electrode cluster (electrodes grouped by similar activation funcitons)
        # clus_ret_mat is a 2d matrix (electrodes x clusters for a given cluster level. use self.clus_getMasterRetIdxMat(cut_lvel) to generate this for the entire colleciton). Then you have to filter this matrix for electrodes in this subject and exclude bad electrodes so you match self.popResponse_dict['popMat']. If do_reverse == True (it works backwards through clus_ret_mat. This is important for when we are doing cumulative computations )

        # get list of regions
        rois,roi_list = self.pop_getRoiList()

        # get rt_ms (to compute traj params), and short and long trial indices
        rt_ms = self.popResponse_dict['pow_ev_filt']['RT'].to_numpy()
        s_idx = self.popResponse_dict['pow_ev_filt'].eval('delay==500').to_numpy()
        l_idx = self.popResponse_dict['pow_ev_filt'].eval('delay==1500').to_numpy()

        # get beh_var (x)
        if beh_var_lbl == 'zrrt':
            x = stats.zscore(-1/self.popResponse_dict['pow_ev_filt']['RT'].to_numpy())
        else:
            x = np.copy(self.memReg_dict[f])
            # z-score (to help interpret parameters)
            x = (x - np.nanmean(x))/np.nanstd(x)
     
        # make a dict
        data_dictS = {}
        data_dictS[beh_var_lbl] = x[s_idx]

        data_dictL = {}
        data_dictL[beh_var_lbl] = x[l_idx]

        # containers
        stats_dict = {}

        # parse labels
        if do_cumulative==True:
            cum_str = 'Cum'
        else:
            cum_str = ''

        if do_reverse==True:
            rev_str = 'Rev'
        else:
            rev_str = ''

        s_lbl = 'byClus'+cum_str+rev_str+'StatS-'+beh_var_lbl+'-'
        l_lbl = 'byClus'+cum_str+rev_str+'StatL-'+beh_var_lbl+'-'
       
        for n in neu_var_list:
            # initialize containers for long and short trials
            stats_dict[s_lbl+n]=np.zeros(np.shape(clus_ret_mat)[1])
            stats_dict[l_lbl+n]=np.zeros(np.shape(clus_ret_mat)[1])

        stats_dict['byClus'+cum_str+rev_str+'_n_in_clus']=np.zeros(np.shape(clus_ret_mat)[1])
        stats_dict['byClus'+cum_str+rev_str+'_lbls'] = []

        # parse do_reverse
        if do_reverse==True:
            clus_ret_mat = np.fliplr(clus_ret_mat)

        # loop through regions
        for i in range(0,np.shape(clus_ret_mat)[1]):
            stats_dict['byClus'+cum_str+rev_str+'_n_in_clus'][i] = np.sum(clus_ret_mat[:,i])


            # assign cluster label (inference is based on do_reverse)
            if do_reverse==False:
                stats_dict['byClus'+cum_str+rev_str+'_lbls'].append(str(np.shape(clus_ret_mat)[1]-1)+'-'+str(i))
            elif do_reverse==True:
                stats_dict['byClus'+cum_str+rev_str+'_lbls'].append(str(np.shape(clus_ret_mat)[1]-1)+'-'+str(np.shape(clus_ret_mat)[1]-(i+1)))

            if stats_dict['byClus'+cum_str+rev_str+'_n_in_clus'][i]<min_elec_thresh:
                for n in neu_var_list:
                    stats_dict[s_lbl+n][i]=np.nan
                    stats_dict[l_lbl+n][i]=np.nan
            else:
                # get ret_idx
                if do_cumulative==False:
                    # only include this cluster
                    ret_idx = clus_ret_mat[:,i].astype('bool')
                else:
                    # include electrodes in all regions from the first index in clus_ret_mat to this index (inclusive)
                    ret_idx = np.sum(clus_ret_mat[:,:i+1],axis=1).astype('bool')


                # get popMat
                popMat3d_ret = self.popResponse_dict['popMat'][:,:,ret_idx]

                # get traj params using only electrodes from this region
                #NOTE: we run self.pop_ev2pop because getTrajParams function expects that rts are ordered in relation to the popMat (short trials, followed by long trials, rather than the original trial order during the experiment)
                thisTrajParams_dict = self.pop_getTrajParams(popMat3d=popMat3d_ret,rts_ms=self.pop_ev2pop(rt_ms))

                # loop through neural features
                n_count = -1
                for n in neu_var_list:
                    n_count+=1
                    # update data dict
                    data_dictS[n] = self.pop_pop2ev(thisTrajParams_dict[n])[s_idx]
                    data_dictL[n] = self.pop_pop2ev(thisTrajParams_dict[n])[l_idx]

                    # update data dict with covar list items also
                    for c in covar_list[n_count]:
                        data_dictS[c] = self.pop_pop2ev(thisTrajParams_dict[c])[s_idx]
                        data_dictL[c] = self.pop_pop2ev(thisTrajParams_dict[c])[l_idx]

                    # compute statistic
                    if stat_option=='tstat':
                        #run regression for short and long blocks
                        regS = smf.ols(beh_var_lbl+ ' ~ ' +n, data = data_dictS).fit()
                        regL = smf.ols(beh_var_lbl+ ' ~ ' +n, data = data_dictL).fit()

                        stats_dict[s_lbl+n][i] = regS.tvalues.loc[n]
                        stats_dict[l_lbl+n][i] = regL.tvalues.loc[n]

                    elif stat_option=='corr':
                        stats_dict[s_lbl+n][i],pval = stats.spearmanr(data_dictS[beh_var_lbl],data_dictS[n])
                        stats_dict[l_lbl+n][i],pval = stats.spearmanr(data_dictL[beh_var_lbl],data_dictL[n])

                    elif stat_option=='corrPartial':

                        #make it a dataframe for pingouin
                        data_dfS = pd.DataFrame.from_dict(data_dictS)
                        data_dfL = pd.DataFrame.from_dict(data_dictL)

                        # compute partial corr
                        corrstatsS = pg.partial_corr(data = data_dfS, x=n,y =beh_var_lbl,covar = covar_list[n_count],method='spearman')
                        corrstatsL = pg.partial_corr(data = data_dfL, x=n,y =beh_var_lbl,covar = covar_list[n_count],method='spearman')

                        # get stats
                        stats_dict[s_lbl+n][i] = corrstatsS.iloc[0]['r']
                        stats_dict[l_lbl+n][i] = corrstatsL.iloc[0]['r']
        # statsByRegion_dict
        stats_dict['byClus'+cum_str+rev_str+'_stat_option'] =stat_option

        return stats_dict 


    def pop_doStats_by_pca(self,beh_var_lbl ='zrrt',n_bins = 10):
        # this function calculates relation between behavioral variable (e.g zrrt) and dynamics (trajectory parameters) separately for bins of pca dimensions set by n_bins. 

        # get rt_ms (to compute traj params)
        rt_ms = self.popResponse_dict['pow_ev_filt']['RT'].to_numpy()


        # get beh_var (x)
        if beh_var_lbl == 'zrrt':
            x = stats.zscore(-1/self.popResponse_dict['pow_ev_filt']['RT'].to_numpy())
            # include delay cond is long as a control variable
            d_str = 'delayCondIsLong + '

        else:
            x = np.copy(self.memReg_dict[f])
            # z-score (to help interpret parameters)
            x = (x - np.nanmean(x))/np.nanstd(x)
            # do not include delay cond in regresssion
            d_str = ''
                          

        # containers
        tstats_srRate = []
        tstats_srDist = []
        bin_lbls = []
        

        # num of pca dimensions (ie num eectrodes)
        num_pca_dim = np.shape(self.popResponse_dict['popMat3d_pc'])[2]

        # bin length
        bin_size = int(num_pca_dim/n_bins) 

        # loop through regions
        for n in range(0,n_bins):
            # infer label
            bin_lbls.append('pcaBin-'+str(n))

            # get bin idx

            # get ret_idx
            ret_idx = np.arange((n*bin_size),((n*bin_size)+bin_size))

            # get popMat (by PCA dim) for this cluster
            popMat3d_ret = self.popResponse_dict['popMat3d_pc'][:,:,ret_idx]

            # get traj params using only electrodes from this region
            #NOTE: we run self.pop_ev2pop because getTrajParams function expects that rts are ordered in relation to the popMat (short trials, followed by long trials, rather than the original trial order during the experiment)
            thisTrajParams_dict = self.pop_getTrajParams(popMat3d=popMat3d_ret,rts_ms=self.pop_ev2pop(rt_ms))

            # reg_dict (with behavioral and neural variable)
            reg_dict = {}
            reg_dict[beh_var_lbl] = x
            reg_dict['delayCondIsLong'] = self.memReg_dict['delayCondIsLong']

            # get neural features
            for n in ['SR_rate','SR_dist']:
                # note we run pop2ev here to rearrange traj params in events order
                reg_dict[n] = self.pop_pop2ev(thisTrajParams_dict[n])

            # sr rate regression
            reg_rate = smf.ols(beh_var_lbl+ ' ~ ' + d_str + 'SR_rate', data = reg_dict).fit()

            # store t-stats (of beta coeff)
            tstats_srRate.append(reg_rate.tvalues.loc['SR_rate'])

            # sr distance regression
            reg_dist = smf.ols(beh_var_lbl+ ' ~ ' + d_str + 'SR_dist', data = reg_dict).fit()

            # store t-stats (of beta coeff)
            tstats_srDist.append(reg_dist.tvalues.loc['SR_dist'])
        

        # statsByClus_dict
        statsByPca_dict = {}
        statsByPca_dict['byPca_lbls'] = bin_lbls
        statsByPca_dict['tstats_byPca_srRate'] = tstats_srRate
        statsByPca_dict['tstats_byPca_srDist'] = tstats_srDist

        return statsByPca_dict  

    def pop_doStats_by_pca_cum(self,beh_var_lbl='zrrt',neu_var_lbl='SR_headingCorrect',covar_list = ['SR_dist'],stat_option='corrPartial',num_pca_dim = None,overwriteFlag=False):
        # this function calculates relation between a behavioral variable  (beh_var_lbl) and a specific neural dynamic (neu_var_lbl) for various levels of data. Starting from only one principal component, and then iteratively including data from additional components. Performs this separately for long and short trials. For each iteration, it computes a non-parametric z-statistic that indicates how much the computed statistic deviates from the null distribution. It uses the null distribution already calculated by self.pop_doStats using all the data (need to run this first) 


        # implement a save and load pickle here
        # look for saved file
        this_fname = self.popResponse_fname+'-'+'popStats-'+self.filter_bool_lbl+'-cumPCA-'+stat_option+beh_var_lbl+'-'+neu_var_lbl
        if (os.path.exists(self.params_dir+this_fname)==True)&(overwriteFlag==False):
            
            #load file if it exists
            statsByPca_dict = (self.load_pickle(self.params_dir+
                                                 this_fname))

            return statsByPca_dict


        # get rt_ms (to compute traj params)
        rt_ms = self.popResponse_dict['pow_ev_filt']['RT'].to_numpy()

        # error check
        if hasattr(self,'popStats_dict') == False:
            raise NameError('need to run self.pop_doStats() first')

        # get short and long trial idx
        s_idx = self.popResponse_dict['pow_ev_filt'].eval('delay==500').to_numpy()
        l_idx = self.popResponse_dict['pow_ev_filt'].eval('delay==1500').to_numpy()

        # get beh_var (x)
        if beh_var_lbl == 'zrrt':
            x = stats.zscore(-1/self.popResponse_dict['pow_ev_filt']['RT'].to_numpy())
        else:
            x = np.copy(self.memReg_dict[f])
            # z-score (to help interpret parameters)
            x = (x - np.nanmean(x))/np.nanstd(x)


        # build data dictionaries by delay condition
        data_dictS = {}
        data_dictS[beh_var_lbl] = x[s_idx] 

        data_dictL = {}
        data_dictL[beh_var_lbl] = x[l_idx]
       
        # containers
        statS = []
        statL =[]
        zstatS = []
        zstatL = []

        # num of pca dimensions (ie num eectrodes)
        # popMat3d_pc has shape trials x time x num components
        if num_pca_dim is None:
            num_pca_dim = np.shape(self.popResponse_dict['popMat3d_pc'])[2]

        # loop through components
        for n in range(0,num_pca_dim):

            # get ret_idx. at n = 0, we only grab the first component
            ret_idx = np.arange(0,n+1)

            # get popMat (by PCA dim) for this cluster
            popMat3d_ret = self.popResponse_dict['popMat3d_pc'][:,:,ret_idx]

            # get traj params using only electrodes from this region
            #NOTE: we run self.pop_ev2pop because getTrajParams function expects that rts are ordered in relation to the popMat (short trials, followed by long trials, rather than the original trial order during the experiment)
            thisTrajParams_dict = self.pop_getTrajParams(popMat3d=popMat3d_ret,rts_ms=self.pop_ev2pop(rt_ms))

            # update data dict with neural feature of interest and covar_list. Note we are changing trial order back to events form
            data_dictS[neu_var_lbl] = self.pop_pop2ev(thisTrajParams_dict[neu_var_lbl])[s_idx]
            data_dictL[neu_var_lbl] = self.pop_pop2ev(thisTrajParams_dict[neu_var_lbl])[l_idx]

            for c in covar_list:
                data_dictS[c] = self.pop_pop2ev(thisTrajParams_dict[c])[s_idx]
                data_dictL[c] = self.pop_pop2ev(thisTrajParams_dict[c])[l_idx]


            # compute statistic
            if stat_option=='tstat':
                #run regression for short and long blocks
                regS = smf.ols(beh_var_lbl+ ' ~ ' +neu_var_lbl, data = data_dictS).fit()
                regL = smf.ols(beh_var_lbl+ ' ~ ' +neu_var_lbl, data = data_dictL).fit()

                # get stat (x)
                xS = regS.tvalues.loc[neu_var_lbl]
                xL = regL.tvalues.loc[neu_var_lbl]

                # labels for null distributions
                lblS = 'popRegS_'+beh_var_lbl+'_'+neu_var_lbl+'_tstat_nullDist'
                lblL = 'popRegL_'+beh_var_lbl+'_'+neu_var_lbl+'_tstat_nullDist' 
            elif stat_option=='corr':
                xS,pval = stats.spearmanr(data_dictS[beh_var_lbl],data_dictS[neu_var_lbl])
                xL,pval = stats.spearmanr(data_dictL[beh_var_lbl],data_dictL[neu_var_lbl])

                # labels for null distributions
                lblS = 'popCorrS_'+beh_var_lbl+'_'+neu_var_lbl+'_rval_nullDist'
                lblL = 'popCorrL_'+beh_var_lbl+'_'+neu_var_lbl+'_rval_nullDist' 

            elif stat_option=='corrPartial':

                #make it a dataframe for pingouin
                data_dfS = pd.DataFrame.from_dict(data_dictS)
                data_dfL = pd.DataFrame.from_dict(data_dictL)

                # compute partial corr
                corrstatsS = pg.partial_corr(data = data_dfS, x=neu_var_lbl,y =beh_var_lbl,covar = covar_list,method='spearman')
                corrstatsL = pg.partial_corr(data = data_dfL, x=neu_var_lbl,y =beh_var_lbl,covar = covar_list,method='spearman')

                # labels for null distributions
                lblS = 'popCorrPartialS_'+beh_var_lbl+'_'+neu_var_lbl+'_rval_nullDist'
                lblL = 'popCorrPartialL_'+beh_var_lbl+'_'+neu_var_lbl+'_rval_nullDist' 

                # get stats
                xS = corrstatsS.iloc[0]['r']
                xL = corrstatsL.iloc[0]['r']

            # convert to a z-score
            # zS = (xS-np.mean(self.popStats_dict[lblS]))/np.std(self.popStats_dict[lblS])
            # zL = (xL-np.mean(self.popStats_dict[lblL]))/np.std(self.popStats_dict[lblL])
            
            # append to container
            statS.append(xS)
            statL.append(xL)
            # zstatS.append(zS)
            # zstatL.append(zL)

            #
            print(n,'/',num_pca_dim)

        # statsByClus_dict
        statsByPca_dict = {}
        statsByPca_dict['cumPCA_stat_option'] = stat_option
        statsByPca_dict['cumPCA_covar_list'] = covar_list
        statsByPca_dict['cumPCA_num_dim'] = num_pca_dim
        statsByPca_dict['cumPCAS_'+beh_var_lbl+'_'+neu_var_lbl] = np.array(statS)
        statsByPca_dict['cumPCAL_'+beh_var_lbl+'_'+neu_var_lbl] = np.array(statL)
        #statsByPca_dict['zstatS'] = zstatS
        #statsByPca_dict['zstatL'] = zstatL

        #save pickle
        self.save_pickle(obj=statsByPca_dict,fpath=self.params_dir+this_fname)

        return statsByPca_dict  

    ########################## PLOTTING ##########################
    def pop_plotPopStats_SSE(self,beh_var_lbl = 'zrrt',ax = None,fsize_tick=14,fsize_lbl=14,neu_feat_list = ['S0_pc','St_pc']):
        # plot popStats SSE results and associated confidence intervals (for predicting behav. variables based on PCA coords)
        if ax is None:
            f = plt.figure()
            ax = plt.subplot(111)

        # loop through neural features
        count = -1
        for n in neu_feat_list:
            count+=1

            lbl = 'popReg_'+beh_var_lbl+'_'+n+'_SSE'

            pvalnp = self.popStats_dict[lbl+'_pvalnp']
            cipos = self.popStats_dict[lbl+'_cipos']
            cineg = self.popStats_dict[lbl+'_cineg']
            sse = self.popStats_dict[lbl]

            ax.plot((count,count),(cineg,cipos),color = '0.5')
            ax.plot((count-.25,count+.25),(cineg,cineg),color = '0.5')
            ax.plot((count-.25,count+.25),(cipos,cipos),color = '0.5')
            ax.plot(count,sse,'*k',markersize=10)
            ax.text(count+.1,sse-3,s='p = '+str(np.round(pvalnp,3)),fontsize=fsize_tick)
        ax.set_xlim((-.5,1.5))
        ax.set_xticks([0,1])
        ax.set_xticklabels(neu_feat_list,fontsize=fsize_tick)
        ax.set_ylabel('Sum of squared error',fontsize=fsize_lbl)
        ax.set_title(beh_var_lbl,fontsize=fsize_lbl)

        # plot error bars indicating null SSE distribution

    def pop_plotPopStats_reg(self,beh_var_lbl = 'zrrt',ax=None,fsize_tick=14,fsize_lbl=14, neu_feat_list = ['SR_dist','SR_speed','SR_headingCorrect'],stat_option = 'tstat',d_str = ''):
        # plot regression results and confidence intervales
        if ax is None:
            f = plt.figure()
            ax = plt.subplot(111)

        # loop through neural features
        count = -1
        xtick_lbls = []
        for n in neu_feat_list:
            if 'pc' in n:
                # skip PCA coords (which do not have specific tstats)
                continue

            count+=1
            xtick_lbls.append(n)

            if stat_option == 'tstat':
                # use ols regression model
                lbl = 'popReg'+d_str+'_'+beh_var_lbl+'_'+n+'_tstat'
                split_lbl = '_tstat'
                ylbl = 't statistic'
            elif stat_option == 'corr':
                # use spearman correlation
                lbl = 'popCorr'+d_str+'_'+beh_var_lbl+'_'+n+'_rval'
                split_lbl = '_rval'
                ylbl = 'spearman r value'
            elif stat_option == 'corrPartial':
                # use spearman correlation
                lbl = 'popCorrPartial'+d_str+'_'+beh_var_lbl+'_'+n+'_rval'
                split_lbl = '_rval'
                ylbl = 'spearman r value'            

            # x is a true statistic indicating relation between neural dynamics and behavior (tstat or rvalue) 
            x = self.popStats_dict[lbl]

            pvalnp = self.popStats_dict[lbl.split(split_lbl)[0]+'_pvalnp']
            cipos = self.popStats_dict[lbl.split(split_lbl)[0]+'_cipos']
            cineg = self.popStats_dict[lbl.split(split_lbl)[0]+'_cineg']

            # plot error bar (manually)
            ax.plot((count,count),(cineg,cipos),color = '0.5')
            ax.plot((count-.25,count+.25),(cineg,cineg),color = '0.5')
            ax.plot((count-.25,count+.25),(cipos,cipos),color = '0.5')

            # plot true tstat
            ax.plot(count,x,'*k',markersize=10)

            # write pval?
            ax.text(count+.1,x-.05,s='p = '+str(np.round(pvalnp,3)),fontsize=fsize_tick)

        ax.set_xlim((-.5,count+.5))
        ax.set_xticks(np.arange(0,count+1))
        ax.set_xticklabels(xtick_lbls,fontsize=fsize_tick)
        ax.set_ylabel(ylbl,fontsize=fsize_lbl)
        ax.set_title(beh_var_lbl,fontsize=fsize_lbl)
    def pop_plotPopStats_byDelay(self,ax=None,fsize_tick=14,fsize_lbl=14, neu_feat_list = ['SR_dist','SR_speed','SR_headingCorrect'],print_zstat=False,use_resid = True):
        # plot regression results and confidence intervales
        if ax is None:
            f = plt.figure()
            ax = plt.subplot(111)
            ax2 = ax.twinx()

        # loop through neural features
        count = -1
        xtick_lbls = []
        for n in neu_feat_list:
            if 'pc' in n:
                # skip PCA coords (which do not have specific tstats)
                continue

            count+=1
            xtick_lbls.append(n)

            # use ols regression model
            if use_resid == True:
                lbl = 'popByDelayResid'+'_'+n+'_tstat'
            else:
                lbl = 'popByDelay'+'_'+n+'_tstat'

            split_lbl = '_tstat'
            ylbl = 't statistic'

            # x is a true statistic indicating relation between neural dynamics and behavior (tstat or rvalue) 
            x = self.popStats_dict[lbl]

            pvalnp = self.popStats_dict[lbl.split(split_lbl)[0]+'_pvalnp']
            zstatnp = self.popStats_dict[lbl.split(split_lbl)[0]+'_zstatnp']
            cipos = self.popStats_dict[lbl.split(split_lbl)[0]+'_cipos']
            cineg = self.popStats_dict[lbl.split(split_lbl)[0]+'_cineg']

            # plot error bar (manually)
            ax.plot((count,count),(cineg,cipos),color = '0.5')
            ax.plot((count-.25,count+.25),(cineg,cineg),color = '0.5')
            ax.plot((count-.25,count+.25),(cipos,cipos),color = '0.5')

            # plot true tstat
            ax.plot(count,x,'*k',markersize=10)

            # write non param pval or zstat
            if print_zstat == False:
                ax2.text(count+.1,.75,s='p = '+str(np.round(pvalnp,3)),fontsize=fsize_tick,rotation=45)
            else:
                ax2.text(count+.1,.75,s='z (np) = '+str(np.round(zstatnp,3)),fontsize=fsize_tick,rotation=45)

        ax.set_xlim((-.5,count+.5))
        ax.set_xticks(np.arange(0,count+1))
        ax.set_xticklabels(xtick_lbls,fontsize=fsize_tick)
        ax.set_ylabel(ylbl,fontsize=fsize_lbl)

    def pop_plotTrajParamsByCondition(self,param_lbl,cond_lbl = ['short delay','long delay'], boolA = None, boolB = None, ax = None):
        # this function plots a violin plot showing distributions of trajectory parameters (distance measures)
        # param_lbl is the label of the parameter to plot, will compare short delay and long delay trials
        #cond_lbl list of strings to label the conditions we are comparing
        #boolA ... bool indicating trials in condition A referenced to self.popResponse_dict['pow_ev_filt']. if None, will default to short delay trials
        #bool B... bool indicating trials in condition B. if None, will default to long delay trials

        # get short and long trial bool
        if boolA is None:
            ev_filt = self.popResponse_dict['pow_ev_filt']
            boolA = ev_filt.eval('delay==500')
        if boolB is None:
            ev_filt = self.popResponse_dict['pow_ev_filt']
            boolB = ev_filt.eval('delay==1500')

        if ax is None:
            f = plt.figure()
            ax = plt.subplot(111)

        # get x and y arrays
        x = self.pop_pop2ev(self.popResponse_dict[param_lbl])[boolA]
        y = self.pop_pop2ev(self.popResponse_dict[param_lbl])[boolB]

        ax.violinplot((x,y),showmeans=True)
        tstat,pval = stats.ttest_ind(x,y,equal_var=False)
        fstat,pval_anov = stats.f_oneway(x,y)
        ax.set_ylabel(param_lbl)
        ax.set_xticks([1,2])
        ax.set_xticklabels(cond_lbl)
        ax.set_title('t stat:'+str(np.round(tstat,2))+'; pval:'+str(np.round(pval,2))+'\n f stat:'+str(np.round(fstat,2))+'; pval:'+str(np.round(pval_anov,2)))




    def pop_plot3d(self,popMat2d_pc,trial_len_samp,rts_ms, targOn_offset_samp = None,ax= None,data_for_cmap = None,plot_option = 'SR',trials_option = 'all',num_trials_to_plot=20,center_on_origin = True,center_on_response=False):
        #this function plots trial by data in principal component space for visualization. It plots each trial from start to finish. It marks stim onset time and response time for each trajectory. Inputs:
        #popMat2d_pc .... 2d array (time(all trials concatenated) x electrodes)
        #trial_len_samp... int. indicating number of samples for each trial (it uses this to identify trial start and stop).
        #rt_ms .... 1d array with RTs in ms matched to popMat2d (from stimulus onset time)
        #targOn_offset_samp ... if None, assumes that trial starts with stim onset time. 
        #data_for_cmap.. Data to be mapped onto the color map. Uses an ordinal mapping
        #plot_option ... determines segment to plot. 'SR'...reaction from stim on to response; 'S'...stim on (starting point), 'R'...response (threshold)
        #num_trials_for_SR_plot..... number of random trials to plot for SR option

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        #import matplotlib.colors as col

        if (center_on_origin==True)&(center_on_response==True):
            raise NameError('center on origin and center on response cannot both be set to True')

        if ax is None:
            f = plt.figure()
            ax = Axes3D(f, rect=[0, 0, .95, 1], elev=48, azim=134)    
            #ax.grid([])
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_zlabel('PC 3')


        if targOn_offset_samp is None:
            targOn_offset_samp = 0

        # compute num of trials
        n_trials = int(np.shape(popMat2d_pc)[0]/trial_len_samp)

        # get rts in samp
        rts_samp = self.ms_to_samples(rts_ms)


        # generate colormap 
        if data_for_cmap is None:
            # if none is provided use rts
            data_for_cmap = rts_ms

        # get sorted indices for data_for_cmap. Will use these indices to assign a color whose intensity is based on the data_for_cmap value on an ordinal scale
        data_for_cmap_sort_idx = np.argsort(data_for_cmap)

        # get colormap (highest values = HOT, lowest values = COLD)
        cmap = cm.get_cmap('coolwarm', n_trials)(np.arange(0,n_trials))
        #cmap = col.LinearSegmentedColormap.from_list("c", [(0,0,1),(.5,.5,1),(0,0,0)])


        # loop through trials and plot
        if trials_option == 'all':
            # plot all trials

            if num_trials_to_plot is None:
                # plot all trials
                trials_ = np.arange(0,n_trials)
            else:
                #randomly choose 'num_trials_to_plot' trials
                trials_ = np.random.randint(0,n_trials,num_trials_to_plot)

        elif trials_option == 'fast':
            if num_trials_to_plot is None:
                trials_ = np.argsort(rts_ms)
            else:
                trials_ = np.argsort(rts_ms)[0:num_trials_to_plot]

        elif trials_option == 'slow':
            if num_trials_to_plot is None:
                trials_ = np.argsort(rts_ms)
            else:
                trials_ = np.argsort(rts_ms)[-num_trials_to_plot:]

        for i in trials_:
            samp_start = (((i+1)*trial_len_samp) - trial_len_samp) 
            samp_end = (i+1)*trial_len_samp

            # identify stim on sample and RT sample for this trial
            targOn_idx_thisTrial = samp_start+targOn_offset_samp
            rt_idx_thisTrial =  int(targOn_idx_thisTrial+rts_samp[i])

            # parse color index by finding where this trial falls in data_for_cmap
            col_idx = np.nonzero(data_for_cmap_sort_idx==i)[0][0]

            if plot_option=='SR':
                coords = (popMat2d_pc[targOn_idx_thisTrial:rt_idx_thisTrial,0],popMat2d_pc[targOn_idx_thisTrial:rt_idx_thisTrial,1],popMat2d_pc[targOn_idx_thisTrial:rt_idx_thisTrial,2])

                if center_on_origin == True:
                    correction_x =  coords[0][0]
                    correction_y =  coords[1][0]
                    correction_z =  coords[2][0]
                elif center_on_response == True:
                    correction_x =  coords[0][-1]
                    correction_y =  coords[1][-1]
                    correction_z =  coords[2][-1]
                else:
                    correction_x =  0
                    correction_y =  0
                    correction_z =  0                    

                # plot stim on - response (reaction)
                ax.scatter(coords[0]-correction_x,coords[1]-correction_y,coords[2]-correction_z,s = .1,alpha = 0.2, color = cmap[col_idx,:])
                # # mark starting point
                ax.scatter(coords[0][0]-correction_x,coords[1][0]-correction_y,coords[2][0]-correction_z,s = 10,alpha = 0.2, color = 'g')

                # # mark response threshold
                ax.scatter(coords[0][-1]-correction_x,coords[1][-1]-correction_y,coords[2][-1]-correction_z,s = 10,alpha = 0.2, color = '0.5', edgecolor = 'k')
                # ax.scatter(popMat2d_pc[rt_idx_thisTrial,0],popMat2d_pc[rt_idx_thisTrial,1],popMat2d_pc[rt_idx_thisTrial,2],s = 10,alpha = 0.2, color = 'r')

            elif plot_option=='S':
                # plot starting point
                ax.scatter(popMat2d_pc[targOn_idx_thisTrial,0],popMat2d_pc[targOn_idx_thisTrial,1],popMat2d_pc[targOn_idx_thisTrial,2],s = 40,alpha = 0.2, color = cmap[col_idx,:],edgecolor='k')
            elif plot_option=='R':
                # plot response trheshold
                ax.scatter(popMat2d_pc[rt_idx_thisTrial,0],popMat2d_pc[rt_idx_thisTrial,1],popMat2d_pc[rt_idx_thisTrial,2],s = 40,alpha = 0.2, color = cmap[col_idx,:],edgecolor='k')                       







            #print(i,samp_start,samp_end,len(np.arange(samp_start,samp_end)))
   
























