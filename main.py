import time
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Import game and agent modules
from games.tic_tac_toe import TicTacToe
from games.connect_four import ConnectFour
from agents.minimax import Minimax
from agents.q_learning import QLearning
from agents.default_opponent import DefaultOpponent

# Set up logging
logging.basicConfig(
    filename=f'experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------
# Interactive Game Play Function
# ------------------------------
def play_game(game_type='tic_tac_toe', agent1_type='minimax', agent2_type='default', train_episodes=1000, minimax_params=None):
    """Play a game between two agents with detailed output."""
    game = TicTacToe() if game_type == 'tic_tac_toe' else ConnectFour()
    
    # Handle the case where minimax_params is None
    if minimax_params is None:
        minimax_params = {"use_alpha_beta": True}
    
    agent1, agent2 = initialize_agents(game, agent1_type, agent2_type, train_episodes, minimax_params)
    
    logging.info(f"Starting {game_type} game: {agent1_type} vs {agent2_type}")
    print(f"\n=== {game_type.upper()} Game: {agent1_type} vs {agent2_type} ===")
    game.print_board()
    
    while not game.game_over:
        current_agent = agent1 if game.current_player == 1 else agent2
        start_time = time.time()
        move = get_agent_move(current_agent, game)
        move_time = time.time() - start_time
        
        if move is None:
            logging.error("Received None move, ending game")
            break
            
        try:
            make_move(game, move, game_type)
            print(f"\nPlayer {game.current_player} ({type(current_agent).__name__}) move: {move}")
            game.print_board()
            print(f"Move time: {move_time:.4f}s")
            logging.info(f"Player {game.current_player} move: {move}, Time: {move_time:.4f}s")
        except Exception as e:
            logging.error(f"Error making move: {e}")
            break
    
    announce_result(game)
    return game.winner

# ------------------------------
# Helper Functions
# ------------------------------
def initialize_agents(game, agent1_type, agent2_type, train_episodes, minimax_params):
    """Initialize agents based on type."""
    # Handle the case where minimax_params is None
    if minimax_params is None:
        minimax_params = {"use_alpha_beta": True}
        
    agents = {}
    for agent_type, player in [(agent1_type, 1), (agent2_type, -1)]:
        if agent_type == 'minimax':
            agents[player] = Minimax(game, max_depth=5 if isinstance(game, TicTacToe) else 3,
                                   use_alpha_beta=minimax_params.get("use_alpha_beta", True), eval_strategy='advanced')
        elif agent_type == 'q_learning':
            agent = QLearning(game, player=player)
            agent.train(train_episodes)
            agents[player] = agent
        else:
            agents[player] = DefaultOpponent(game)
    return agents[1], agents[-1]

def get_agent_move(agent, game):
    """Get move from agent based on its type."""
    try:
        if isinstance(agent, Minimax):
            move = agent.get_best_move()
        elif isinstance(agent, QLearning):
            move = agent.choose_action(game.get_valid_moves())
        else:  # DefaultOpponent
            move = agent.get_move()
        
        if move is None:
            logging.error(f"Agent {type(agent).__name__} returned None move. Valid moves: {game.get_valid_moves()}")
            valid_moves = game.get_valid_moves()
            move = valid_moves[0] if valid_moves else None  # Fallback to first valid move
        return move
    except Exception as e:
        logging.error(f"Error getting agent move: {e}")
        valid_moves = game.get_valid_moves()
        return valid_moves[0] if valid_moves else None  # Fallback to first valid move

def make_move(game, move, game_type):
    """Make a move based on game type with error handling."""
    if move is None:
        logging.error(f"Attempted to make None move in {game_type}")
        raise ValueError("Invalid move: None received")
    
    if game_type == 'tic_tac_toe':
        try:
            row, col = move
            game.make_move(row, col)
        except (TypeError, ValueError) as e:
            logging.error(f"Invalid Tic Tac Toe move format: {move}. Error: {str(e)}")
            raise
    else:
        game.make_move(move)

def announce_result(game):
    """Announce game result."""
    result = "Player 1 wins!" if game.winner == 1 else "Player 2 wins!" if game.winner == -1 else "It's a draw!"
    print(f"\nResult: {result}")
    logging.info(f"Game result: {result}")

# ------------------------------
# Experimental Evaluation
# ------------------------------
def run_experiment(game_type, matchup_name, agent1_type, agent2_type, num_games=100, train_episodes=1000):
    """Run experiment for a matchup with detailed statistics."""
    game_class = TicTacToe if game_type == 'tic_tac_toe' else ConnectFour
    results = {'wins': 0, 'draws': 0, 'losses': 0, 'move_times': [], 'game_times': []}
    
    # Pre-train Q-Learning agents
    q_agents = {}
    for atype, player in [(agent1_type, 1), (agent2_type, -1)]:
        if atype == 'q_learning':
            q_game = game_class()
            q_agent = QLearning(q_game, player=player)
            q_agent.train(train_episodes)
            q_agents[player] = q_agent
    
    logging.info(f"Starting {game_type} experiment: {matchup_name} ({num_games} games)")
    print(f"\nRunning {game_type} - {matchup_name} ({num_games} games)...")
    
    for game_num in range(num_games):
        game = game_class()
        minimax_params = {"use_alpha_beta": True} if 'AB' in matchup_name.split()[0] else {"use_alpha_beta": False}
        agent1, agent2 = initialize_agents(game, agent1_type, agent2_type, train_episodes, minimax_params)
        
        start_game_time = time.time()
        game_moves_made = 0
        
        while not game.game_over:
            current_agent = agent1 if game.current_player == 1 else agent2
            start_time = time.time()
            move = get_agent_move(current_agent, game)
            move_time = time.time() - start_time
            
            if move is None:
                logging.error(f"Game {game_num}: Received None move, ending game")
                break
                
            try:
                make_move(game, move, game_type)
                results['move_times'].append(move_time)
                game_moves_made += 1
            except Exception as e:
                logging.error(f"Game {game_num}: Error making move: {e}")
                break
        
        # Only record game time if moves were actually made
        if game_moves_made > 0:
            game_time = time.time() - start_game_time
            results['game_times'].append(game_time)
        
        if game.winner == 1:
            results['wins'] += 1
        elif game.winner == -1:
            results['losses'] += 1
        else:
            results['draws'] += 1
        
        if game_num % 10 == 0:
            logging.info(f"{game_type} - {matchup_name} Game {game_num}: W:{results['wins']}, D:{results['draws']}, L:{results['losses']}")
    
    # Safely calculate averages
    results['avg_move_time'] = np.mean(results['move_times']) if results['move_times'] else 0.0
    results['avg_game_time'] = np.mean(results['game_times']) if results['game_times'] else 0.0
    results['win_rate'] = results['wins'] / num_games if num_games > 0 else 0.0
    
    print(f"Results: Wins: {results['wins']}, Draws: {results['draws']}, Losses: {results['losses']}, "
          f"Avg Move Time: {results['avg_move_time']:.4f}s, Avg Game Time: {results['avg_game_time']:.4f}s")
    logging.info(f"{game_type} - {matchup_name} Results: {results}")
    return results

def run_all_experiments(num_games=100, train_episodes_ttt=1000, train_episodes_cf=1000):
    """Run all specified matchups for both games."""
    matchups = [
        ('Minimax_AB vs Default', 'minimax', 'default'),
        ('Default vs Minimax_AB', 'default', 'minimax'),
        ('Minimax_noAB vs Default', 'minimax', 'default'),
        ('Default vs Minimax_noAB', 'default', 'minimax'),
        ('Q1 vs Default', 'q_learning', 'default'),
        ('Default vs Q2', 'default', 'q_learning'),
        ('Minimax_AB vs Q2', 'minimax', 'q_learning'),
        ('Q1 vs Minimax_AB', 'q_learning', 'minimax'),
        ('Minimax_noAB vs Q2', 'minimax', 'q_learning'),
        ('Q1 vs Minimax_noAB', 'q_learning', 'minimax'),
        ('Minimax_AB vs Minimax_noAB', 'minimax', 'minimax'),
        ('Minimax_noAB vs Minimax_AB', 'minimax', 'minimax'),
    ]
    
    all_results = {'tic_tac_toe': {}, 'connect_four': {}}
    for game_type in ['tic_tac_toe', 'connect_four']:
        train_episodes = train_episodes_ttt if game_type == 'tic_tac_toe' else train_episodes_cf
        for matchup_name, agent1_type, agent2_type in matchups:
            result = run_experiment(game_type, matchup_name, agent1_type, agent2_type, num_games, train_episodes)
            all_results[game_type][matchup_name] = result
    return all_results

# ------------------------------
# Plotting Functions
# ------------------------------
def plot_experiment_results(all_results):
    """Create detailed plots for experiment results."""
    for game_type, matchups_data in all_results.items():
        matchups = list(matchups_data.keys())
        wins = [data['wins'] for data in matchups_data.values()]
        draws = [data['draws'] for data in matchups_data.values()]
        losses = [data['losses'] for data in matchups_data.values()]
        avg_move_times = [data['avg_move_time'] for data in matchups_data.values()]
        avg_game_times = [data['avg_game_time'] for data in matchups_data.values()]
        win_rates = [data['win_rate'] for data in matchups_data.values()]
        
        # Win/Draw/Loss Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(matchups))
        width = 0.25
        ax.bar(x - width, wins, width, label='Wins', color='green')
        ax.bar(x, draws, width, label='Draws', color='gray')
        ax.bar(x + width, losses, width, label='Losses', color='red')
        ax.set_ylabel('Number of Games')
        ax.set_title(f'{game_type.replace("_", " ").title()} Results')
        ax.set_xticks(x)
        ax.set_xticklabels(matchups, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{game_type}_results.png')
        plt.close()
        
        # Win Rate Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(matchups, win_rates, color='blue')
        ax.set_ylabel('Win Rate')
        ax.set_title(f'{game_type.replace("_", " ").title()} Win Rates')
        plt.xticks(rotation=45, ha='right')
        for i, v in enumerate(win_rates):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
        plt.tight_layout()
        plt.savefig(f'{game_type}_win_rates.png')
        plt.close()
        
        # Time Analysis - Only create if we have timing data
        if any(avg_move_times) or any(avg_game_times):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax1.bar(matchups, avg_move_times, color='purple')
            ax1.set_ylabel('Avg Move Time (s)')
            ax1.set_title(f'{game_type.replace("_", " ").title()} Performance')
            ax2.bar(matchups, avg_game_times, color='orange')
            ax2.set_ylabel('Avg Game Time (s)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{game_type}_times.png')
            plt.close()

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    # Interactive demo (uncomment to run)
    # play_game('tic_tac_toe', 'minimax', 'default', train_episodes=1000)
    
    # Run all experiments
    all_results = run_all_experiments(
        num_games=50,
        train_episodes_ttt=1000,
        train_episodes_cf=500  # Reduced for Connect Four due to complexity
    )
    
    # Generate plots
    plot_experiment_results(all_results)
    logging.info("Experiment completed successfully")
    print("\nExperiment completed. Check log file and generated plots for detailed results.")