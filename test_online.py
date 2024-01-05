import asyncio
import time
from poke_env import ShowdownServerConfiguration, AccountConfiguration
from poke_env.player import Player


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)




async def main():
    # Replace with your bot's Showdown username and password
    account_config = AccountConfiguration("lanolano", "zmxncbv0192")

    max_damage_player = MaxDamagePlayer(
        account_configuration=account_config,
        battle_format="gen7randombattle",
        server_configuration=ShowdownServerConfiguration
    )

    # This line starts searching for battles
    await max_damage_player.ladder(1)  # Number of battles to search for


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
