import numpy as np
from numpy.random import randint

class p:   
    seed = 1

    
    # converting to string
    def _str(self):
        s = '___'.join("%s=%s" % (attr, val) for (attr, val) in vars(p).items()
                       if not attr.startswith('_') and not attr.startswith('out') and not attr.startswith('its') and not attr.startswith('lr_ticks'))
        return ''.join(["%s" % randint(0, 9) for num in range(0, 20)]) + '_' + s.replace('/', '')[:50]
 # filenames must fit in byte 
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}

    
