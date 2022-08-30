VAR2DSKEY = {
    'red': ['image_pixel_values'],
    'green': ['image_pixel_values'],
    'blue': ['image_pixel_values'],
    'veg': ['image_pixel_values'],
    'co2': ['image_pixel_values'],
    'cld': ['CLD'],
    'cla': ['CT', 'CLL', 'CA', 'CF'],
    'swrad': ['RSR', 'DSR', 'ASR'],
}

FD_VAR2FILE = {
    'red': 'gk2a_ami_le1b_vi006_fd005ge_{target_datetime}.nc', # vi006
    'green': 'gk2a_ami_le1b_vi005_fd010ge_{target_datetime}.nc', # vi005
    'blue': 'gk2a_ami_le1b_vi004_fd010ge_{target_datetime}.nc', # vi004
    'veg': 'gk2a_ami_le1b_vi008_fd010ge_{target_datetime}.nc', # vi008
    'co2': 'gk2a_ami_le1b_ir133_fd020ge_{target_datetime}.nc', # ir133
    'cld': 'gk2a_ami_le2_cld_fd020ge_{target_datetime}.nc', # cloud detection
    'cla': 'gk2a_ami_le2_cla_fd020ge_{target_datetime}.nc', # cloud analysis
    'swrad': 'gk2a_ami_le2_swrad_fd020ge_{target_datetime}.nc', # radiance
}

EA_VAR2FILE = {
    'red': 'gk2a_ami_le1b_vi006_fd005ge_{target_datetime}.nc', # vi006
    'green': 'gk2a_ami_le1b_vi005_fd010ge_{target_datetime}.nc', # vi005
    'blue': 'gk2a_ami_le1b_vi004_fd010ge_{target_datetime}.nc', # vi004
    'veg': 'gk2a_ami_le1b_vi008_fd010ge_{target_datetime}.nc', # vi008
    'co2': 'gk2a_ami_le1b_ir133_fd020ge_{target_datetime}.nc', # ir133
    'cld': 'gk2a_ami_le2_cld_fd020ge_{target_datetime}.nc', # cloud detection
    'cla': 'gk2a_ami_le2_cla_fd020ge_{target_datetime}.nc', # cloud analysis
    'swrad': 'gk2a_ami_le2_swrad_fd020ge_{target_datetime}.nc', # radiance
}

KO_VAR2FILE = {
    'red': 'gk2a_ami_le1b_vi006_fd005ge_{target_datetime}.nc', # vi006
    'green': 'gk2a_ami_le1b_vi005_fd010ge_{target_datetime}.nc', # vi005
    'blue': 'gk2a_ami_le1b_vi004_fd010ge_{target_datetime}.nc', # vi004
    'veg': 'gk2a_ami_le1b_vi008_fd010ge_{target_datetime}.nc', # vi008
    'co2': 'gk2a_ami_le1b_ir133_fd020ge_{target_datetime}.nc', # ir133
    'cld': 'gk2a_ami_le2_cld_fd020ge_{target_datetime}.nc', # cloud detection
    'cla': 'gk2a_ami_le2_cla_fd020ge_{target_datetime}.nc', # cloud analysis
    'swrad': 'gk2a_ami_le2_swrad_fd020ge_{target_datetime}.nc', # radiance
}