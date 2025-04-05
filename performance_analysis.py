import numpy as np
import time
from games.tic_tac_toe import TicTacToe
from games.connect_four import ConnectFour

from agents.minimax import Minimax
from agents.q_learning import QLearning
from agents.default_opponent import DefaultOpponent

def run_performance_analysis(game_type='tic_tac_toe', num_games=100, train_episodes=1000):
    """
    Run performance analysis for different agents.
    
    Args:
        game_type (str): Type of game ('tic_tac_toe' or 'connect_four')
        num_games (int): Number of games to simulate for evaluation
        train_episodes (int): Number of episodes to train Q-learning agent
    """
    if game_type == 'tic_tac_toe':
        game_class = TicTacToe
    else:
        game_class = ConnectFour

    # Train Q-learning agent
    game = game_class()
    q_agent = QLearning(game)
    q_agent.train(episodes=train_episodes)

    results = {
        'minimax_vs_default': {'wins': 0, 'draws': 0, 'losses': 0, 'times': []},
        'q_learning_vs_default': {'wins': 0, 'draws': 0, 'losses': 0, 'times': []},
        'minimax_vs_q_learning': {'wins': 0, 'draws': 0, 'losses': 0, 'times': []}
    }

    for matchup in results.keys():
        parts = matchup.split('_vs_')
        agent1_type = parts[0]
        agent2_type = parts[1]
        for _ in range(num_games):
            game = game_class()
            if agent1_type == 'minimax':
                agent1 = Minimax(game, max_depth=5 if game_type == 'tic_tac_toe' else 3, use_alpha_beta=True, eval_strategy='advanced')
            elif agent1_type == 'q_learning':
                agent1 = q_agent
            else:
                agent1 = DefaultOpponent(game)

            if agent2_type == 'minimax':
                agent2 = Minimax(game, max_depth=5 if game_type == 'tic_tac_toe' else 3, use_alpha_beta=True, eval_strategy='advanced')
            elif agent2_type == 'q_learning':
                agent2 = q_agent
            else:
                agent2 = DefaultOpponent(game)

            while not game.game_over:
                start_time = time.time()
                if game.current_player == 1:
                    if agent1_type == 'minimax':
                        move = agent1.get_best_move()
                    elif agent1_type == 'q_learning':
                        valid_moves = game.get_valid_moves()
                        move = agent1.choose_action(valid_moves)
                    else:
                        move = agent1.get_move()
                else:
                    if agent2_type == 'minimax':
                        move = agent2.get_best_move()
                    elif agent2_type == 'q_learning':
                        valid_moves = game.get_valid_moves()
                        move = agent2.choose_action(valid_moves)
                    else:
                        move = agent2.get_move()

                if isinstance(game, TicTacToe):
                    row, col = move
                    game.make_move(row, col)
                else:
                    game.make_move(move)

                end_time = time.time()
                results[matchup]['times'].append(end_time - start_time)

            if game.winner == 1:
                results[matchup]['wins'] += 1
            elif game.winner == -1:
                results[matchup]['losses'] += 1
            else:
                results[matchup]['draws'] += 1

    # Print results
    for matchup, data in results.items():
        print(f"{matchup}:")
        print(f"  Wins: {data['wins']}")
        print(f"  Draws: {data['draws']}")
        print(f"  Losses: {data['losses']}")
        print(f"  Average time per move: {np.mean(data['times']):.4f} seconds")

if __name__ == "__main__":
    run_performance_analysis('connect_four', num_games=100, train_episodes=10000)