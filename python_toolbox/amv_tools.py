from eccodes import *
import numpy as np
from datetime import datetime

def load_amv(amv_file):
    bufr_keys = ['#1#latitude', '#1#longitude', '#1#pressure',
                 '#1#windDirection', '#1#windSpeed',
                 '#1#heightAssignmentMethod',
                 '#1#windDirection->qualityControl',
                 '#1#windSpeed->qualityControl']
    bufr_values = {}
    datetime_keys = ['#1#year', '#1#month', '#1#day', '#1#hour', '#1#minute',
                     '#1#second']
    datetime_values = {}
    bufr_values['datetime'] = []
    bufr_count = 0
    for gid in bufr_messages(amv_file):
        codes_set(gid, 'unpack', 1)
        for key in datetime_keys:
            datetime_values[key] = codes_get_array(gid, key)
        for i in np.array(datetime_values.values()).T.tolist():
            bufr_values['datetime'].append(datetime(**dict(zip(
                                [k[3:] for k in datetime_values.keys()], i))))
        for key in bufr_keys:
            if key[3:] in bufr_values.keys():
                bufr_values[key[3:]].extend(codes_get_array(gid, key).tolist())
            else:
                bufr_values[key[3:]] = codes_get_array(gid, key).tolist()
        bufr_count+=1
    print 'Total BUFR messages read:',bufr_count
    for key in bufr_values.keys():
        bufr_values[key] = np.array(bufr_values[key])
    return bufr_values

def get_amv_vectors(amv_dict):
    x = amv_dict['longitude']
    y = amv_dict['latitude']
    ang = np.radians(amv_dict['windDirection'])
    speed = amv_dict['windSpeed']
    # minus because SEVIRI azi out by 180 degrees
    u = -np.sin(ang)*speed
    v = -np.cos(ang)*speed
    return x,y,u,v

"""Generator function allowing iteration over all messages in a BUFR
file. Exception-safe."""
def bufr_messages(filename):
    with open(filename,'r') as F:
        while True:
            with NextBufrMessage(F) as gid:
                yield gid

"""Context manager class for a BUFR message. Automatically calls
codes_release() on the message even if an exception is raised while
processing it. Raises a StopIteration exception if there are no more
messages in the file."""
class NextBufrMessage(object):
    def __init__(self,F):
        # Store the bufr file descriptor on entering the context
        self.F=F
    def __enter__(self):
        # Read, store and return the next message
        self.gid=codes_bufr_new_from_file(self.F)
        if self.gid is None:
            raise StopIteration
        return self.gid
    def __exit__(self,ex_type,ex_value,ex_trace):
        # Clean up the message when leaving the context or on exception
        codes_release(self.gid)
