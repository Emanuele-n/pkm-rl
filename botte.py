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

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import Player, RandomPlayer, cross_evaluate


class Botte(Player):

    def get_state_vector(self, battle):
        state_vector = []
        # Need a way to get every data from the battle object and put it in the state vector
        # The journey probably ends here...
        return state_vector
        

    # Usage in your choose_move method
    def choose_move(self, battle):
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

    

async def main():

    start = time.time()

    # We define two player configurations, we want to train with self play
    player_1_configuration = AccountConfiguration("MCTS player", None)

    # Start with random player
    player_2_configuration = AccountConfiguration("Random player", None)

    # We create the corresponding players.
    mcts_player_1 = Botte(
        account_configuration=player_1_configuration,
        battle_format="gen7randombattle",
        server_configuration=LocalhostServerConfiguration,
        #team=team
    )

    random_player = RandomPlayer(
        account_configuration=player_2_configuration,
        battle_format="gen7randombattle",
        server_configuration=LocalhostServerConfiguration,
        #team=team_2
    )


    # Now, let's evaluate our player
    cross_evaluation = await cross_evaluate(
        [mcts_player_1, random_player], n_challenges=100
    )

    print(
        "MCTS player won %d / 100 battles [this took %f seconds]"
        % (
            cross_evaluation[mcts_player_1.username][random_player.username] * 100,
            time.time() - start,
        )
    )

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())