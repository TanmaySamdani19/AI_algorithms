import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from typing import List, Type, Tuple, Dict, Any
from games.tic_tac_toe import TicTacToe
from games.connect_four import ConnectFour
import os
import logging
from typing import List, Type, Tuple, Dict, Any
from datetime import datetime


from games.tic_tac_toe import TicTacToe
from games.connect_four import ConnectFour
from algorithms.minimax import Minimax
from algorithms.q_learning import QLearning
from opponents.default_opponent import DefaultOpponent

class GameExperiment:
    """
    Comprehensive game algorithm performance experiment framework.
    
    Supports multiple games, algorithms, and opponents with robust performance analysis.
    """
    
    def __init__(
        self, 
        game_class: Type, 
        algorithm_classes: List[Type], 
        opponent_class: Type, 
        num_games: int = 100,
        log_level: int = logging.INFO
    ):
        """
        Initialize the experiment with configurable game, algorithms, and experiment parameters.
        
        Args:
            game_class (Type): Game class to be used (TicTacToe or ConnectFour)
            algorithm_classes (List[Type]): List of algorithm classes to compare
            opponent_class (Type): Opponent class to play against
            num_games (int): Number of games to play for each algorithm
            log_level (int): Logging level for experiment tracking
        """
        self.game_class = game_class
        self.algorithm_classes = algorithm_classes
        self.opponent_class = opponent_class
        self.num_games = num_games
        
        # Configure logging
        logging.basicConfig(
            level=log_level, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def run_experiment(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive performance experiments with robust statistical analysis.
        
        Args:
            verbose (bool): Print detailed experiment results
        
        Returns:
            Dict containing detailed performance metrics
        """
        self.logger.info(f"Starting experiment for {self.game_class.__name__}")
        results = {}
        
        for algorithm_class in self.algorithm_classes:
            algorithm_name = algorithm_class.__name__
            self.logger.info(f"Running experiment for {algorithm_name}")
            
            results[algorithm_name] = self._run_algorithm_experiment(algorithm_class)

        if verbose:
            self._print_detailed_results(results)

        return results

    def _run_algorithm_experiment(self, algorithm_class: Type) -> Dict[str, float]:
        """
        Comprehensive algorithm performance evaluation with multiple runs.
        
        Args:
            algorithm_class (Type): Algorithm to evaluate
        
        Returns:
            Detailed performance metrics
        """
        metrics = ['wins', 'losses', 'draws', 'win_rates', 'avg_moves', 'total_game_time']
        experiment_results = {metric: [] for metric in metrics}

        # Multiple statistically significant runs
        num_runs = 10
        start_time = datetime.now()

        for run in range(num_runs):
            self.logger.debug(f"Run {run + 1}/{num_runs}")
            game = self.game_class()
            opponent = self.opponent_class(game)
            
            # Adaptive algorithm initialization
            if algorithm_class == Minimax:
                max_depth = 5 if isinstance(game, TicTacToe) else 3
                algorithm = algorithm_class(game, max_depth=max_depth, use_alpha_beta=True)
            elif algorithm_class == QLearning:
                algorithm = algorithm_class(game)
                algorithm.train(verbose=False)  # Minimize training output
            else:
                algorithm = algorithm_class(game)

            run_result = self._play_games(game, algorithm, opponent)
            
            for metric in metrics:
                if metric == 'total_game_time':
                    experiment_results[metric].append(
                        (datetime.now() - start_time).total_seconds()
                    )
                else:
                    experiment_results[metric].append(run_result[metric])

        # Calculate comprehensive statistics with advanced error handling
        results_summary = {}
        for metric in metrics:
            results_summary[f'mean_{metric}'] = np.mean(experiment_results[metric])
            results_summary[f'std_{metric}'] = np.std(experiment_results[metric])
            results_summary[f'min_{metric}'] = np.min(experiment_results[metric])
            results_summary[f'max_{metric}'] = np.max(experiment_results[metric])

        return results_summary

    def _play_games(self, game, algorithm, opponent):
        """
        Enhanced game simulation with comprehensive result tracking.
        
        Returns:
            Detailed game performance metrics
        """
        results = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rates': 0,
            'avg_moves': 0,
            'total_game_time': 0
        }

        for _ in range(self.num_games):
            game.reset()
            current_player = 1
            moves_in_game = 0

            while not game.game_over:
                moves_in_game += 1
                current_player = 3 - current_player  # Alternate players

                try:
                    move = (
                        algorithm.get_best_move() 
                        if current_player == 1 
                        else opponent.get_move()
                    )
                    self._make_game_move(game, move)
                except Exception as e:
                    self.logger.error(f"Error in game move: {e}")
                    break

            # Comprehensive result tracking
            if game.winner == 1:
                results['wins'] += 1
            elif game.winner == -1:
                results['losses'] += 1
            else:
                results['draws'] += 1
            
            results['avg_moves'] += moves_in_game

        # Calculate aggregate metrics
        results['win_rates'] = results['wins'] / self.num_games
        results['avg_moves'] /= self.num_games

        return results

    def _make_game_move(self, game, move):
        """
        Robust move-making method with game-specific handling."""
        if isinstance(game, TicTacToe):
            game.make_move(move[0], move[1])
        else:  # Connect Four
            game.make_move(move)

    def _print_detailed_results(self, results):
        """
        Enhanced result visualization with comprehensive statistics.
        """
        print(f"\n{'='*50}")
        print(f"Game: {self.game_class.__name__} Performance Analysis")
        print(f"{'='*50}")
        
        for algo, stats in results.items():
            print(f"\n{algo} Comprehensive Performance:")
            for metric in ['win_rates', 'wins', 'losses', 'draws', 'avg_moves', 'total_game_time']:
                print(f"{metric.replace('_', ' ').title()}:")
                print(f"  Mean: {stats[f'mean_{metric}']:.4f}")
                print(f"  Std Dev: {stats[f'std_{metric}']:.4f}")
                print(f"  Min: {stats[f'min_{metric}']:.4f}")
                print(f"  Max: {stats[f'max_{metric}']:.4f}")

    @staticmethod
    def plot_performance(results, game_name, output_dir='results'):
        """
        Advanced performance visualization with error bars and comprehensive metrics.
        """
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(15, 8))
        algorithms = list(results.keys())
        
        # Comprehensive performance metrics
        metrics = {
            'Win Rates': 'win_rates',
            'Avg Moves': 'avg_moves'
        }

        for i, (title, metric) in enumerate(metrics.items(), 1):
            plt.subplot(1, len(metrics), i)
            values = [results[algo][f'mean_{metric}'] for algo in algorithms]
            errors = [results[algo][f'std_{metric}'] for algo in algorithms]

            plt.bar(
                algorithms, 
                values, 
                yerr=errors, 
                capsize=10, 
                alpha=0.7, 
                color=['blue', 'green']
            )
            plt.title(f'{title} in {game_name}')
            plt.xlabel('Algorithms')
            plt.ylabel(title)
            plt.xticks(rotation=45)

            # Add value labels
            for j, v in enumerate(values):
                plt.text(j, v, f'{v:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{game_name.lower().replace(" ", "_")}_performance.png'))
        plt.close()

def main():
    """
    Comprehensive game algorithm experiments with multiple configurations.
    """
    # Extensible game and algorithm configurations
    game_types = [
        (TicTacToe, [Minimax, QLearning], DefaultOpponent, 100),
        (ConnectFour, [Minimax, QLearning], DefaultOpponent, 50)
    ]

    # Run experiments and generate visualizations
    all_results = {}
    for game_class, algorithm_classes, opponent_class, num_games in game_types:
        experiment = GameExperiment(
            game_class=game_class, 
            algorithm_classes=algorithm_classes,
            opponent_class=opponent_class,
            num_games=num_games,
            log_level=logging.INFO
        )
        results = experiment.run_experiment()
        GameExperiment.plot_performance(results, game_class.__name__)
        all_results[game_class.__name__] = results

if __name__ == "__main__":
    main()