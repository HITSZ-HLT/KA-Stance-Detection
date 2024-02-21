from utils.data_configs.sem16_config import Sem16Config
from utils.data_configs.p_stance_config import PStanceConfig
from utils.data_configs.covid_19_config import Covid19Config
from utils.data_configs.vast_config import VASTConfig


from utils.datasets.sem16_dataset import Sem16Dataset
from utils.datasets.p_stance_dataset import PStanceDataset
from utils.datasets.covid_19_dataset import Covid19Dataset
from utils.datasets.vast_dataset import VASTDataset

from utils.kasd_datasets.sem16_kasd_dataset import Sem16KASDDataset
from utils.kasd_datasets.p_stance_kasd_dataset import PStanceKASDDataset
from utils.kasd_datasets.covid_19_kasd_dataset import Covid19KASDDataset
from utils.kasd_datasets.vast_kasd_dataset import VASTKASDDataset

# ----------------------------------------------------------------------------------------------------------------------------------------------
data_configs = {
    'sem16': Sem16Config,
    'p_stance': PStanceConfig,
    'covid_19': Covid19Config,
    'vast': VASTConfig,
}

datasets = {
    'sem16': Sem16Dataset,
    'p_stance': PStanceDataset,
    'covid_19': Covid19Dataset,
    'vast': VASTDataset,
}

kasd_datasets = {
    'sem16': Sem16KASDDataset,
    'p_stance': PStanceKASDDataset,
    'covid_19': Covid19KASDDataset,
    'vast': VASTKASDDataset,
}