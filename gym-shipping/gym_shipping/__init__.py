import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Shipping-v0',
    entry_point='gym_shipping.envs:ShippingEnv'
)
