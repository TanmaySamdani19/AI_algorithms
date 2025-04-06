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

# Set up logging to record experiment details with a timestamped log file
logging.basicConfig(
    filename=f'experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------
# Interactive Game Play Function
# ------------------------------
def play_game(game_type='tic_tac_toe', agent1_type='minimax', agent2_type='default', train_episodes=1000, minimax_params=None):
    """
    Play a game between two agents and display detailed output.
    
    Parameters:
        game_type (str): The type of game ('tic_tac_toe' or 'connect_four').
        agent1_type (str): Type of agent for player 1 ('minimax', 'q_learning', or 'default').
        agent2_type (str): Type of agent for player 2 ('minimax', 'q_learning', or 'default').
        train_episodes (int): Number of episodes for training Q-Learning agents.
        minimax_params (dict): Parameters for the Minimax agent (e.g., {"use_alpha_beta": True}).
    
    Returns:
        int or None: The winner of the game (1 for player 1, -1 for player 2, or None if no moves were made).
    """
    # Initialize the game object based on the type
    game = TicTacToe() if game_type == 'tic_tac_toe' else ConnectFour()
    
    # Set default minimax parameters if none are provided
    if minimax_params is None:
        minimax_params = {"use_alpha_beta": True}
    
    # Initialize agents for both players
    agent1, agent2 = initialize_agents(game, agent1_type, agent2_type, train_episodes, minimax_params)
    
    logging.info(f"Starting {game_type} game: {agent1_type} vs {agent2_type}")
    print(f"\n=== {game_type.upper()} Game: {agent1_type} vs {agent2_type} ===")
    game.print_board()
    
    # Dictionary to store node evaluation counts for analysis
    node_counts = {1: [], -1: []}
    
    # Main game loop until the game is over
    while not game.game_over:
        # Select the current agent based on the current player
        current_agent = agent1 if game.current_player == 1 else agent2
        start_time = time.time()  # Record start time for move
        
        # Reset node counter if the agent is using Minimax
        if isinstance(current_agent, Minimax):
            current_agent.nodes_evaluated = 0  
            
        # Get the move from the agent
        move = get_agent_move(current_agent, game)
        move_time = time.time() - start_time  # Calculate move time
        
        # Record node count for Minimax agents
        if isinstance(current_agent, Minimax):
            node_counts[game.current_player].append(current_agent.nodes_evaluated) 
            nodes_info = f", Nodes evaluated: {current_agent.nodes_evaluated}"
        else:
            nodes_info = ""
            
        if move is None:
            logging.error("Received None move, ending game")
            break
            
        try:
            # Make the move in the game
            make_move(game, move, game_type)
            print(f"\nPlayer {game.current_player} ({type(current_agent).__name__}) move: {move}{nodes_info}")
            game.print_board()
            print(f"Move time: {move_time:.4f}s")
            logging.info(f"Player {game.current_player} move: {move}, Time: {move_time:.4f}s{nodes_info}")
        except Exception as e:
            logging.error(f"Error making move: {e}")
            break
    
    # Announce the game result
    announce_result(game)
    
    # Print summary of node evaluations if applicable
    if any(node_counts[1]) or any(node_counts[-1]):
        print("\nNode Evaluation Summary:")
        if node_counts[1]:
            print(f"Player 1: Total nodes: {sum(node_counts[1])}, Avg per move: {sum(node_counts[1])/len(node_counts[1]):.1f}")
        if node_counts[-1]:
            print(f"Player 2: Total nodes: {sum(node_counts[-1])}, Avg per move: {sum(node_counts[-1])/len(node_counts[-1]):.1f}")
    
    return game.winner

# ------------------------------
# Helper Functions
# ------------------------------
def initialize_agents(game, agent1_type, agent2_type, train_episodes, minimax_params):
    """
    Initialize agents based on the specified type.
    
    Parameters:
        game: The game instance.
        agent1_type (str): Type for player 1 agent.
        agent2_type (str): Type for player 2 agent.
        train_episodes (int): Number of training episodes for Q-Learning agents.
        minimax_params (dict): Parameters for Minimax agent.
    
    Returns:
        tuple: A tuple containing the agents for player 1 and player 2.
    """
    # Set default minimax parameters if not provided
    if minimax_params is None:
        minimax_params = {"use_alpha_beta": True}
        
    agents = {}
    for agent_type, player in [(agent1_type, 1), (agent2_type, -1)]:
        if agent_type == 'minimax':
            # For Tic Tac Toe, use max_depth 5; for Connect Four, use max_depth 3
            agents[player] = Minimax(game, max_depth=5 if isinstance(game, TicTacToe) else 3,
                                   use_alpha_beta=minimax_params.get("use_alpha_beta", True), eval_strategy='advanced')
        elif agent_type == 'q_learning':
            # Initialize and train Q-Learning agent
            agent = QLearning(game, player=player)
            agent.train(train_episodes)
            agents[player] = agent
        else:
            # Use the default opponent
            agents[player] = DefaultOpponent(game)
    return agents[1], agents[-1]

def get_agent_move(agent, game):
    """
    Obtain the move from the specified agent.
    
    Parameters:
        agent: The current agent.
        game: The game instance.
    
    Returns:
        The move selected by the agent.
    """
    try:
        if isinstance(agent, Minimax):
            # Reset node counter and get move using Minimax
            agent.nodes_evaluated = 0
            move = agent.get_best_move()
            logging.info(f"Minimax nodes evaluated: {agent.nodes_evaluated}")
            return move
        elif isinstance(agent, QLearning):
            # Choose move based on Q-Learning policy (epsilon-greedy)
            move = agent.choose_action(game.get_valid_moves())
        else:  # DefaultOpponent
            move = agent.get_move()
        
        # Fallback if move is None
        if move is None:
            logging.error(f"Agent {type(agent).__name__} returned None move. Valid moves: {game.get_valid_moves()}")
            valid_moves = game.get_valid_moves()
            move = valid_moves[0] if valid_moves else None
        return move
    except Exception as e:
        logging.error(f"Error getting agent move: {e}")
        valid_moves = game.get_valid_moves()
        return valid_moves[0] if valid_moves else None
    
def make_move(game, move, game_type):
    """
    Make the move in the game with error handling.
    
    Parameters:
        game: The game instance.
        move: The move to be executed.
        game_type (str): The type of game ('tic_tac_toe' or other).
    
    Raises:
        ValueError: If the move is invalid.
    """
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
    """
    Announce and log the result of the game.
    
    Parameters:
        game: The game instance.
    """
    result = "Player 1 wins!" if game.winner == 1 else "Player 2 wins!" if game.winner == -1 else "It's a draw!"
    print(f"\nResult: {result}")
    logging.info(f"Game result: {result}")

# ------------------------------
# Experimental Evaluation Functions
# ------------------------------
def run_experiment(game_type, matchup_name, agent1_type, agent2_type, num_games=100, train_episodes=10000):
    """
    Run a series of games for the specified matchup and collect performance statistics.
    
    Parameters:
        game_type (str): The type of game ('tic_tac_toe' or 'connect_four').
        matchup_name (str): Name of the matchup (used for logging and plotting).
        agent1_type (str): Type for player 1 agent.
        agent2_type (str): Type for player 2 agent.
        num_games (int): Number of games to run.
        train_episodes (int): Training episodes for Q-Learning agents.
    
    Returns:
        dict: A dictionary containing results such as wins, draws, losses, average times, and node counts.
    """
    game_class = TicTacToe if game_type == 'tic_tac_toe' else ConnectFour
    results = {
        'wins': 0, 
        'draws': 0, 
        'losses': 0, 
        'move_times': [], 
        'game_times': [],
        'nodes_evaluated_agent1': [],
        'nodes_evaluated_agent2': []
    }
    
    # Pre-train Q-Learning agents if necessary
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
        
        # Play one game until completion
        while not game.game_over:
            current_agent = agent1 if game.current_player == 1 else agent2
            start_time = time.time()
            
            # Reset node counter if the agent is Minimax
            if isinstance(current_agent, Minimax):
                current_agent.nodes_evaluated = 0
                
            move = get_agent_move(current_agent, game)
            move_time = time.time() - start_time
            
            # Record node evaluation data if applicable
            if isinstance(current_agent, Minimax):
                if game.current_player == 1:
                    results['nodes_evaluated_agent1'].append(current_agent.nodes_evaluated)
                else:
                    results['nodes_evaluated_agent2'].append(current_agent.nodes_evaluated)
            
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
        
        # Record the overall game time if moves were made
        if game_moves_made > 0:
            game_time = time.time() - start_game_time
            results['game_times'].append(game_time)
        
        # Update win/draw/loss counts based on game result
        if game.winner == 1:
            results['wins'] += 1
        elif game.winner == -1:
            results['losses'] += 1
        else:
            results['draws'] += 1
        
        if game_num % 10 == 0:
            logging.info(f"{game_type} - {matchup_name} Game {game_num}: W:{results['wins']}, D:{results['draws']}, L:{results['losses']}")
    
    # Calculate average move and game times
    results['avg_move_time'] = np.mean(results['move_times']) if results['move_times'] else 0.0
    results['avg_game_time'] = np.mean(results['game_times']) if results['game_times'] else 0.0
    results['win_rate'] = results['wins'] / num_games if num_games > 0 else 0.0
    
    # Calculate node evaluation metrics for Minimax agents
    if results['nodes_evaluated_agent1']:
        results['avg_nodes_agent1'] = np.mean(results['nodes_evaluated_agent1'])
        results['total_nodes_agent1'] = sum(results['nodes_evaluated_agent1'])
    else:
        results['avg_nodes_agent1'] = 0
        results['total_nodes_agent1'] = 0
        
    if results['nodes_evaluated_agent2']:
        results['avg_nodes_agent2'] = np.mean(results['nodes_evaluated_agent2'])
        results['total_nodes_agent2'] = sum(results['nodes_evaluated_agent2'])
    else:
        results['avg_nodes_agent2'] = 0
        results['total_nodes_agent2'] = 0
    
    # Print and log the results
    print(f"Results: Wins: {results['wins']}, Draws: {results['draws']}, Losses: {results['losses']}, "
          f"Avg Move Time: {results['avg_move_time']:.4f}s, Avg Game Time: {results['avg_game_time']:.4f}s")
    
    if agent1_type == 'minimax':
        print(f"Agent1 (Minimax): Avg Nodes: {results['avg_nodes_agent1']:.1f}, Total Nodes: {results['total_nodes_agent1']}")
    if agent2_type == 'minimax':
        print(f"Agent2 (Minimax): Avg Nodes: {results['avg_nodes_agent2']:.1f}, Total Nodes: {results['total_nodes_agent2']}")
    
    logging.info(f"{game_type} - {matchup_name} Results: {results}")
    return results

def run_all_experiments(num_games=100, train_episodes_ttt=10000, train_episodes_cf=10000):
    """
    Run all specified matchups for both Tic Tac Toe and Connect Four.
    
    Parameters:
        num_games (int): Number of games for each matchup.
        train_episodes_ttt (int): Training episodes for Tic Tac Toe.
        train_episodes_cf (int): Training episodes for Connect Four.
    
    Returns:
        dict: A dictionary containing the results for both games and all matchups.
    """
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
    """
    Generate and save detailed plots for experiment results.
    
    Parameters:
        all_results (dict): The dictionary containing experimental results for each game and matchup.
    """
    for game_type, matchups_data in all_results.items():
        matchups = list(matchups_data.keys())
        wins = [data['wins'] for data in matchups_data.values()]
        draws = [data['draws'] for data in matchups_data.values()]
        losses = [data['losses'] for data in matchups_data.values()]
        avg_move_times = [data['avg_move_time'] for data in matchups_data.values()]
        avg_game_times = [data['avg_game_time'] for data in matchups_data.values()]
        win_rates = [data['win_rate'] for data in matchups_data.values()]
        
        # Node count data for Minimax agents
        minimax_matchups = []
        avg_nodes_agent1 = []
        avg_nodes_agent2 = []
        
        for matchup, data in matchups_data.items():
            if 'avg_nodes_agent1' in data or 'avg_nodes_agent2' in data:
                minimax_matchups.append(matchup)
                avg_nodes_agent1.append(data.get('avg_nodes_agent1', 0))
                avg_nodes_agent2.append(data.get('avg_nodes_agent2', 0))
        
        # Plot for Wins, Draws, Losses
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
        
        # Plot for Win Rates
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
        
        # Plot for Time Analysis
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
        
        # Plot for Node Count Analysis (Minimax agents)
        if minimax_matchups:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(minimax_matchups))
            width = 0.35
            
            agent1_bars = ax.bar(x - width/2, avg_nodes_agent1, width, label='Agent 1')
            agent2_bars = ax.bar(x + width/2, avg_nodes_agent2, width, label='Agent 2')
            
            ax.set_ylabel('Avg Nodes Evaluated')
            ax.set_title(f'{game_type.replace("_", " ").title()} Minimax Node Evaluation')
            ax.set_xticks(x)
            ax.set_xticklabels(minimax_matchups, rotation=45, ha='right')
            ax.legend()
            
            # Label each bar with its value
            for i, v in enumerate(avg_nodes_agent1):
                if v > 0:
                    ax.text(i - width/2, v + 10, f'{v:.0f}', ha='center')
            
            for i, v in enumerate(avg_nodes_agent2):
                if v > 0:
                    ax.text(i + width/2, v + 10, f'{v:.0f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(f'{game_type}_node_counts.png')
            plt.close()

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    # Interactive demo (uncomment the following line to run an interactive demo)
    play_game('tic_tac_toe', 'minimax', 'default', train_episodes=1000)
    
    # Run all experiments and generate results
    all_results = run_all_experiments(
        num_games=100, 
        train_episodes_ttt=1000,
        train_episodes_cf=1000  # Reduced for Connect Four due to complexity
    )
    
    # Generate plots for the experiment results
    plot_experiment_results(all_results)
    logging.info("Experiment completed successfully")
    print("\nExperiment completed. Check log file and generated plots for detailed results.")
