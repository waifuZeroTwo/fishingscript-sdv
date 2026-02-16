import logging
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='StardewFisherEnv-v0',
    entry_point='stardew_fisher.envs:StardewFisherEnv',
)
