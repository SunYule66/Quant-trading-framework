from ..tdagent import TDAgent
import numpy as np

class UBAH(TDAgent):

    def __init__(self, b = None):
        super(UBAH, self).__init__()
        self.b = b

    def decide_by_history(self, x, last_b):
        '''return new portfolio vector
        :param x: input matrix
        :param last_b: last portfolio weight vector
        '''
        # Align with CRP-style uniform over current asset universe size
        if self.b is None:
            n_assets = len(self.get_last_rpv(x))
            self.b = np.ones(n_assets) / float(n_assets)
        else:
            self.b = last_b
        return self.b
