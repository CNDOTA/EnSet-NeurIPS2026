from .cq_learner import CQLearner
from .facmac_learner import FACMACLearner
from .maddpg_learner import MADDPGLearner
from .enset_maddpg_learner import EnSetMADDPGLearner
from .enset_facmac_learner import EnSetFACMACLearner


REGISTRY = {}
REGISTRY["cq_learner"] = CQLearner
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["enset_maddpg_learner"] = EnSetMADDPGLearner
REGISTRY["enset_facmac_learner"] = EnSetFACMACLearner
