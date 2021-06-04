import numpy as np

def load_formations():
    """
    Returns:
        forms (list, shape (25, 10, 3)): the 2d array of positional data of position names for 25 templete formations 
        form_labels (list): the list of the names of 25 templete formations
        forwards (list): the list of the names of offensive roles
        defenders (list): the lsit of the names of defensive roles
    """

    # define the formations
    f_343 = np.array([[2, 3, 'RCB'], [2, 5, 'CB'], [2, 7, 'LCB'], [5, 2, 'RM'], [5, 4, 'RCM'], [5, 6, 'LCM'], [5, 8, 'LM'], [7, 2, 'RW'], [7, 8, 'LW'], [8, 5, 'ST']])

    f_352 = np.array([[2, 3, 'RCB'], [2, 5, 'CB'], [2, 7, 'LCB'], [4, 4, 'RCDM'], [4, 6, 'LCDM'],  [6, 2, 'RM'], [6, 5, 'CAM'], [6, 8, 'LM'], [8, 4, 'RST'], [8, 6, 'LST']])

    f_41212 = np.array([[3, 2, 'RB'], [2, 4, 'RCB'], [2, 6, 'LCB'], [3, 8, 'LB'], [4, 5, 'CDM'],  [5, 3, 'RCM'], [5, 7, 'LCM'], [6, 5, 'CAM'], [8, 4, 'RST'], [8, 6, 'LST']])

    f_4231 = np.array([[3, 2, 'RB'], [2, 4, 'RCB'], [2, 6, 'LCB'], [3, 8, 'LB'], [4, 4, 'RCDM'],  [4, 6, 'LCDM'], [6, 2, 'RM'], [6, 8, 'LM'], [6, 5, 'CAM'], [8, 5, 'ST']])

    f_442 = np.array([[3, 2, 'RB'], [2, 4, 'RCB'], [2, 6, 'LCB'], [3, 8, 'LB'], [6, 2, 'RM'],  [5, 4, 'RCM'], [5, 6, 'LCM'], [6, 8, 'LM'], [8, 4, 'RST'], [8, 6, 'LST']])

    f_4123 = np.array([[3, 2, 'RB'], [2, 4, 'RCB'], [2, 6, 'LCB'], [3, 8, 'LB'], [4, 5, 'CDM'],  [5, 3, 'RCM'], [5, 7, 'LCM'], [7, 2, 'RW'], [7, 8, 'LW'], [8, 5, 'ST']])

    f_4213 = np.array([[3, 2, 'RB'], [2, 4, 'RCB'], [2, 6, 'LCB'], [3, 8, 'LB'], [4, 4, 'RCDM'],  [4, 6, 'LCDM'], [6, 5, 'CAM'], [7, 2, 'RW'], [7, 8, 'LW'], [8, 5, 'ST']])

    f_541 = np.array([[2, 3, 'RCB'], [2, 5, 'CB'], [2, 7, 'LCB'], [3, 1, 'RWB'], [3, 9, 'LWB'], [5, 4, 'RCM'], [5, 6, 'LCM'], [7, 2, 'RW'], [7, 8, 'LW'], [8, 5, 'ST']])

    f_532 = np.array([[2, 3, 'RCB'], [2, 5, 'CB'], [2, 7, 'LCB'], [3, 1, 'RWB'], [3, 9, 'LWB'], [5, 2, 'RCM'], [5, 5, 'CM'], [5, 8, 'LCM'], [8, 4, 'RST'], [8, 6, 'LST']])

    forms = [f_343, f_352,
             f_41212, f_4231, f_442, f_4123, f_4213,
             f_541, f_532
             ]

    form_names = ["343", "352",
               "41212", "4231", "442", "4213", "4123",
               "541", "532"
               ]

    forwards = ['ST', 'RST', 'LST', 
                'LW', 'RW']
        
    defenders = ['CB', 'LCB', 'RCB', 
                 'LB', 'RB',
                 'LWB', 'RWB']

    return forms, form_names, forwards, defenders

