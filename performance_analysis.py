import matplotlib.pyplot as plt
from main import run_game, TicTacToe, ConnectFour, Minimax, QLearning, DefaultOpponent

def plot_performance_comparison():
    # Tic Tac Toe experiments
    ttt_games = [TicTacToe() for _ in range(3)]
    ttt_minimax = [Minimax(game) for game in ttt_games]
    ttt_q_learning = [QLearning(game) for game in ttt_games]
    ttt_default_opponent = [DefaultOpponent(game) for game in ttt_games]

    # Train Q-Learning
    for ql in ttt_q_learning:
        ql.train()

    # Run experiments
    minimax_results = [run_game(game, minimax, opponent) 
                       for game, minimax, opponent in zip(ttt_games, ttt_minimax, ttt_default_opponent)]
    q_learning_results = [run_game(game, q_learning, opponent) 
                          for game, q_learning, opponent in zip(ttt_games, ttt_q_learning, ttt_default_opponent)]

    # Prepare data for plotting
    algorithms = ['Minimax', 'Q-Learning']
    wins = [
        [result['algorithm_wins'] for result in minimax_results],
        [result['algorithm_wins'] for result in q_learning_results]
    ]
    draws = [
        [result['draws'] for result in minimax_results],
        [result['draws'] for result in q_learning_results]
    ]
    losses = [
        [result['opponent_wins'] for result in minimax_results],
        [result['opponent_wins'] for result in q_learning_results]
    ]

    # Plotting
    plt.figure(figsize=(10, 6))
    x = range(len(algorithms))
    width = 0.25

    plt.bar([i-width for i in x], [sum(wins[i])/len(wins[i]) for i in range(len(algorithms))], 
            width, label='Wins', color='green')
    plt.bar(x, [sum(draws[i])/len(draws[i]) for i in range(len(algorithms))], 
            width, label='Draws', color='yellow')
    plt.bar([i+width for i in x], [sum(losses[i])/len(losses[i]) for i in range(len(algorithms))], 
            width, label='Losses', color='red')

    plt.xlabel('Algorithms')
    plt.ylabel('Performance Ratio')
    plt.title('Tic Tac Toe Algorithm Performance')
    plt.xticks(x, algorithms)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ttt_performance.png')
    plt.close()

if __name__ == "__main__":
    plot_performance_comparison()