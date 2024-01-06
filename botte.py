# Reminder: lanunch the pokemon showdown server with the command: node pokemon-showdown start --no-security

"""
The goal of this file is to create and train an agent trying to replicate what AlphaGo Zero to play Pokemon Showdown. 
First it will learn to play using a random teams, then it will learn to play using a specific team.
Note that with random team it doesn't mean random battle, but ou battle with a known team, randomly generated before each battle.
The final goal is to use a single team for all battles, and train the agent to win with that team using self-play.

Takeaways from AlphaGo Zero:
- Use a single neural network to evaluate the position and to select moves (instead of using a value network and a policy network)
  f_theta(s) = (p, v) where p is a vector of move probabilities and v is the value of the position s (probability of winning from state s)
- Use Monte Carlo Tree Search to improve the move selection. In place of the raw probability of a move, use the MCTS for selecting the next move
- The network's parameter are optimized for the MCTS, not for the actual game. The MCTS is used to generate training data, and the network is 
  trained to match the MCTS output. The loss is then
    L = (z - v)^2 - pi^T log(p) + c ||theta||^2
    where z is the game's outcome, v is the value predicted by the network, pi is the MCTS output, and c is a regularization parameter.

Define state s of pokemon battle:
- move data:
    - move name
    - move type
    - move category
    - move power
    - move accuracy
    - move priority
    - move target
    - move pp
    - move max pp

- pokemon data:
    - species
    - level
    - current HP
    - status
    - stats
    - stats modifiers
    - moves
        - for each move (x4):
            - move data
    - item
    - ability
    - types
    - weight

- field data:
    - weather
    - terrain
    - hazards
    - screens
    - tailwind
    - room

- player data:
    - active pokemon
    - team
        - for each pokemon (x6):
            - pokemon data

...too complicated!
"""

import asyncio
import time
import numpy as np
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env
from tabulate import tabulate
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import (
    Player,
    Gen8EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
    cross_evaluate
)


class Botte(Player):

    def get_state_vector(self, battle):
        state_vector = []
        # Need a way to get every data from the battle object and put it in the state vector
        # The journey probably ends here...
        # Try to print everything in the battle object
        #print(dir(battle))

        # Get all attributes of the battle object
        for attr in dir(battle):
            # Ignore private and protected attributes
            if not attr.startswith('_'):
                value = getattr(battle, attr)
                # Ignore methods
                if not callable(value):
                    print(f"{attr}: {value}")  # Print attribute name and value
                    state_vector.append(value)
        return state_vector
        

    # Usage in your choose_move method
    def choose_move(self, battle):
        # Get the state vector
        state_vector = self.get_state_vector(battle)
        print(state_vector)

        ## max damage player
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)
        ## end max damage player

        ## MCTS player


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    opponent = RandomPlayer(battle_format="gen8randombattle")
    test_env = SimpleRLPlayer(
        battle_format="gen8randombattle", start_challenging=True, opponent=opponent
    )
    check_env(test_env)
    test_env.close()

    # Create one environment for training and one for evaluation
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    opponent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())