import torch

class KNDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, params, mypath='/gpfs/group/vav5084/default/ashley/knemulator/pt_files/'):
        'Initialization'
        self.params = params
        self.mypath = mypath
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, indx):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        param = self.params[indx]
        y = torch.load(self.mypath + 'kn' + str(param[0])+'-'+str(param[1])+'-'+str(param[2])+'-'+str(param[3])+'-'+str(param[4])+'-'+str(int(param[5]))+'-'+str(int(param[6]))+'.pt').float()
        mymean = torch.mean(y)
        y = y/torch.mean(y)
        y = torch.cat((y, torch.log10(mymean).unsqueeze(0)), 0)
        X = torch.from_numpy(param).float()

        return X, y
        
        