import numpy as np

def get_mean_chisq(D):
    """
    Args: DiskFitting object
    Returns: mean chi^2 (weighted avg of both wavelengths)
    """
    mean_chisq_1 = D.mean_chisq_list[0]
    mean_chisq_2 = D.mean_chisq_list[1]
    disk_area_1 = D.disk_area_list[0]
    disk_area_2 = D.disk_area_list[1]
    if np.isnan(mean_chisq_1): # img=nan or disk area=0
        disk_area_1 = 0
        mean_chisq_1 = 0
    if np.isnan(mean_chisq_2): # img=nan or disk area=0
        disk_area_2 = 0
        mean_chisq_2 = 0
    if disk_area_1==0 and disk_area_2==0:
        return np.nan
    mean_chisq = (mean_chisq_1*disk_area_1 + mean_chisq_2*disk_area_2)/(disk_area_1+disk_area_2)
    return mean_chisq
get_mean_chisq_mult = np.vectorize(get_mean_chisq)