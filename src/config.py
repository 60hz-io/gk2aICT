LOCAL_GK2A_DIR = '/mnt/sda1/research/data/tmp_gk2a/'

SUPERRES_MODEL_PATH = '/mnt/sda1/research/data/superres_model'

LATLON_DIR = '/mnt/sdb1/wscho/data_for_research/ICTgk2a/latlon'

S3_GK2A_DIR = 's3://60hz.data/kmipa/gk2a'

ASOS_DIR = '/mnt/sda1/research/data/asos/mos_automation'


VAR2FILE = { # area = fd, ea, or ko (fulldisk, eastasia, or korea)
    'red': 'gk2a_ami_le1b_vi006_{area}005ge_{yyyymmddHHMM}.nc', # vi006
    'green': 'gk2a_ami_le1b_vi005_{area}010ge_{yyyymmddHHMM}.nc', # vi005
    'blue': 'gk2a_ami_le1b_vi004_{area}010ge_{yyyymmddHHMM}.nc', # vi004
    'veg': 'gk2a_ami_le1b_vi008_{area}010ge_{yyyymmddHHMM}.nc', # vi008
    'co2': 'gk2a_ami_le1b_ir133_{area}020ge_{yyyymmddHHMM}.nc', # ir133
    'cld': 'gk2a_ami_le2_cld_{area}020ge_{yyyymmddHHMM}.nc', # cloud detection
    'cla': 'gk2a_ami_le2_cla_{area}020ge_{yyyymmddHHMM}.nc', # cloud analysis
    'swrad': 'gk2a_ami_le2_swrad_{area}020ge_{yyyymmddHHMM}.nc', # radiance
}