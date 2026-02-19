import matplotlib
import os
import matplotlib.pyplot as plt


# matplotlib backend
matplotlib.use('Qt5Agg')

# matplotlib style
plt.style.use('default')
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Project directory
projdir = '.'

# Data dir
datadir = os.path.join(projdir, 'data')

# Fatigure report dir
reportdir = os.path.join(projdir, 'report')

# Path to fatigue data file
datapath = os.path.join(datadir, 'Sess2_kinematics_data.npy')

# Path to clustering labels
clustlabelspath = os.path.join(datadir, 'Clust_multispeed_ptlabels.csv')

# Master datasheet
masterdatapath = os.path.join(datadir, 'MasterDataSheet.xlsx')

# Physiological data
physdatapath = os.path.join(datadir, 'Sess2_physio_data.npy')

# Speeds
speeds = [11, 12, 13]

wantedgasvars = ['VO2', 'RQ', 'Rf', 'VT']

gas_titles = {'VO2': 'VO2',
              'RQ': 'Respiratory quotient',
              'Rf': 'Respiratory frequency',
              'VT': 'Tidal volume'}

gas_ylabels = {'VO2': '%VO2peak',
               'RQ': 'VCO2/VO2',
               'Rf': 'breaths/min',
               'VT': 'l'}

# Seglabels
seglabels = ['start', 'mid', 'end']

# discvars
discvars = ['DF', 'SFl']

# contvars
contvars = ['RCOM', 'RTRUNK2PELVIS', 'RPELV_ANG', 'RHIP', 'RKNEE', 'RANK']

# wanted vars
wantedvars = discvars + contvars

# coordination couplings
couplings = [('RTRUNK2PELVIS_VEL', 'RHIP_VEL'),
             ('RHIP_VEL', 'RKNEE_VEL'),
             ('RKNEE_VEL', 'RANK_VEL')]

kinematics_titles = {'SFl': 'Stride frequency',
                      'DF': 'Duty factor',
                      'RCOM': 'vCOM',
                      'RTRUNK2PELVIS': 'Trunk-pelvis',
                      'RHIP': 'Hip',
                      'RPELV_ANG': 'Pelvis tilt',
                      'RKNEE': 'Knee',
                      'RANK': 'Ankle'}

kinematics_ylabels = {'SFl': '1/ST/leg length',
                      'DF': 'CT/ST',
                      'RCOM': 'Position (m/leg) \n< Down - Up >',
                      'RTRUNK2PELVIS': '${\Theta}$ (°) \n< Flex - Ext >',
                      'RPELV_ANG': '${\Theta}$ (°) \n< Ant - Post >',
                      'RHIP': '${\Theta}$ (°) \n< Ext - Flex >',
                      'RKNEE': '${\Theta}$ (°) \n< Ext - Flex >',
                      'RANK': '${\Theta}$ (°) \n< Plantar - Dorsi >',
                      }

coord_titles = {'RTRUNK2PELVIS_VEL__RHIP_VEL': 'Trunk-pelvis ${\omega}$ \u2014 Hip ${\omega}$',
                'RHIP_VEL__RKNEE_VEL':'Hip ${\omega}$ \u2014 Knee ${\omega}$',
                'RKNEE_VEL__RANK_VEL':'Knee ${\omega}$ \u2014 Ankle ${\omega}$'}

coord_labels = {'RTRUNK2PELVIS_VEL__RHIP_VEL': 'Ellipse area (°²/s²)',
                'RHIP_VEL__RKNEE_VEL': 'Ellipse area (°²/s²)',
                'RKNEE_VEL__RANK_VEL': 'Ellipse area (°²/s²)'}

omega_labels = {'RTRUNK2PELVIS_VEL': '${\omega$ (°/s) \n< Flex - Ext >',
                'RHIP_VEL': '${\omega$ (°/s) \n< Ext - Flex >',
                'RKNEE_VEL': '${\omega$ (°/s) \n< Ext - Flex >',
                'RANK_VEL': '${\omega$ (°/s) \n< Plantar - Dorsi >',
                }

# Demographics, anthropometrics and physiological variables and titles
demoanthrophysvars_titles = {'Age': 'Age',
                             'Height': 'Height',
                             'Mass': 'Mass',
                             'TrunkLgth': 'Trunk length',
                             'PelvWidth': 'Pelvis width',
                             'LegLgth_r': 'Leg length',
                             'ThiLgth_r': 'Thigh length',
                             'ShaLgth_r': 'Shank length',
                             'FootLgth_r': 'Foot length',
                             'LT': 'LT',
                             'VO2peakkg': 'VO2peak',
                             'RE': 'Running Economy',
                             'RELT': 'Running Economy LT',
                             'RunningDaysAWeek': 'Weekly runs',
                             'KmAWeek': 'Weekly volume',
                             'Time10Ks': '10k time',
                             'Sess2_times': 'Time to exhaustion'
                            }

# Names and units for figures
demoanthrophysvars_ylabels = {'Sex': 'Females (%)',
                          'Age': 'years',
                          'Height': 'm',
                          'Mass': 'kg',
                          'TrunkLgth': 'm',
                          'LegLgth_r': 'm',
                          'PelvWidth': 'm',
                          'ThiLgth_r': 'm',
                          'ShaLgth_r': 'm',
                          'FootLgth_r': 'm',
                          'LT': 'km/h',
                          'VO2peakkg': 'ml/min/kg',
                          'RunningDaysAWeek': 'count',
                          'KmAWeek': 'km',
                          'Time10Ks': 'mm:ss',
                          'Sess2_times': 'mm:ss',
                          'RE': 'kcal/min/kg',
                          }

# Segment colours
segcolours = ['C0', 'C8', 'C3']