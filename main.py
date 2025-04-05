from games.tic_tac_toe import TicTacToe
from games.connect_four import ConnectFour
from agents.minimax import Minimax
from agents.q_learning import QLearning
from agents.default_opponent import DefaultOpponent
import time

def play_game(game_type='tic_tac_toe', agent1_type='minimax', agent2_type='default', train_episodes=1000):
    """
    Play a game between two agents, displaying moves and timing.
    Ensures agents are properly initialized with the game instance.
    """
    if game_type == 'tic_tac_toe':
        game = TicTacToe()
    else:
        game = ConnectFour()

    # Initialize agents with the current game instance
    if agent1_type == 'minimax':
        agent1 = Minimax(game, use_alpha_beta=True, max_depth=5 if game_type=='tic_tac_toe' else 3)
    elif agent1_type == 'q_learning':
        agent1 = QLearning(game, player=1)
        agent1.train(train_episodes)
    else:
        agent1 = DefaultOpponent(game)

    if agent2_type == 'minimax':
        agent2 = Minimax(game, use_alpha_beta=True, max_depth=5 if game_type=='tic_tac_toe' else 3)
    elif agent2_type == 'q_learning':
        agent2 = QLearning(game, player=-1)
        agent2.train(train_episodes)
    else:
        agent2 = DefaultOpponent(game)

    # Play the game, ensuring agents use the current game state
    while not game.game_over:
        if game.current_player == 1:
            agent = agent1
        else:
            agent = agent2

        start_time = time.time()
        if isinstance(agent, Minimax):
            move = agent.get_best_move()
        elif isinstance(agent, QLearning):
            valid_moves = game.get_valid_moves()
            move = agent.choose_action(valid_moves)
        else:
            move = agent.get_move()
        end_time = time.time()
        move_time = end_time - start_time

        if isinstance(game, TicTacToe):
            row, col = move
            game.make_move(row, col)
        else:
            game.make_move(move)

        game.print_board()
        print(f"Move time: {move_time:.4f}s")

    # Announce the result
    if game.winner == 1:
        print("Player 1 wins!")
    elif game.winner == -1:
        print("Player 2 wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    play_game('tic_tac_toe', 'minimax', 'default', train_episodes=1000)
    play_game('connect_four', 'minimax', 'default', train_episodes=1000)