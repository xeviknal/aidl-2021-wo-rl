from trainers.ppo_trainer import PPOTrainer
from trainers.reinforce_baseline_trainer import ReinforceBaselineTrainer
from trainers.reinforce_trainer import ReinforceTrainer

from policies.actor_critic_policy import ActorCriticPolicy
from policies.actor_policy import ActorPolicy

strategy2Trainer = {
    'vpg': [ReinforceTrainer, ActorPolicy],
    'baseline': [ReinforceBaselineTrainer, ActorCriticPolicy],
    'ppo': [PPOTrainer, ActorCriticPolicy],
}


def build(strategy, env, hyper_params):
    trainer_class, policy_class = get_strategy_descriptor(strategy)
    return trainer_class(env, hyper_params, policy_class)


def get_strategy_descriptor(strategy):
    return strategy2Trainer.get(strategy, strategy2Trainer['vpg'])
