_default_sequences = {
    "FP": "GCCGACCAGTTTCCATAG",
    "IP": "AGCACTCAGTATTTGTCCG",
    "RP": "GTCCTCAACAACCTCCTG",
    "toehold": "GTCCTC",
    "biotin": "/5Biosg/UUUUUUUTT"
}

from .encoder import Encoder
from .predictor import Predictor
from .encoder_trainer import EncoderTrainer
from .simulator import Simulator
from .siamese_network import SiameseNetwork
from .triplet_network import TripletNetwork