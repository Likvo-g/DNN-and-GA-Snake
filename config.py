# Game Configuration Settings
# ===========================

# Display and Window Settings
TITLE = "DNN&GA Snake"           # Window title for pygame display
GRID_SIZE = 30                   # Size of each grid cell in pixels
BLANK_SIZE = 40                  # Height of blank area at top for status text
ROWS = 10                        # Number of rows in game board
COLS = 10                        # Number of columns in game board
FPS = 200                        # Target frame rate for visual display
FONT_NAME = 'arial'              # Font name for text rendering

# Color Definitions (RGB tuples)
# ==============================
WHITE = (255, 255, 255)          # Pure white
WHITE1 = (220, 220, 220)         # Light gray (snake head color)
WHITE2 = (255, 255, 255)         # Pure white (alternative)
BLACK = (0, 0, 0)                # Pure black (background color)
RED = (255, 0, 0)                # Pure red (food color)
GREEN = (0, 255, 0)              # Pure green
BLUE1 = (0, 0, 255)              # Pure blue (grid line color)
BLUE2 = (0, 100, 255)            # Medium blue
YELLOW = (255, 255, 0)           # Pure yellow
LIGHTBLUE = (0, 155, 155)        # Cyan-like color
BGCOLOR = BLACK                  # Background color alias
LINE_COLOR = BLUE1               # Grid line color alias

# Mathematical Constants
# ======================
INF = 100000000                  # Large number for initialization and bounds

# Neural Network Architecture Settings
# ====================================
N_INPUT = 32                     # Number of input neurons (environmental state features)
                                # Breakdown: 4 (head direction) + 4 (tail direction) +
                                #           24 (8 directions × 3 features per direction)

N_HIDDEN1 = 20                   # Number of neurons in first hidden layer
N_HIDDEN2 = 12                   # Number of neurons in second hidden layer
N_OUTPUT = 4                     # Number of output neurons (one per movement direction)

# Calculate total number of genetic parameters needed for neural network
# Formula: (input×hidden1 + hidden1×hidden2 + hidden2×output) + (hidden1 + hidden2 + output)
#         weights for connections                              + biases for each layer
GENES_LEN = N_INPUT * N_HIDDEN1 + N_HIDDEN1 * N_HIDDEN2 + N_HIDDEN2 * N_OUTPUT + N_HIDDEN1 + N_HIDDEN2 + N_OUTPUT

# Genetic Algorithm Parameters
# ============================
P_SIZE = 100                     # Population size (number of parent individuals maintained)
C_SIZE = 400                     # Child size (number of offspring generated each generation)
MUTATE_RATE = 0.1               # Mutation rate (probability of each gene being mutated)

# Game Mechanics Settings
# =======================
# Movement directions as (dx, dy) coordinate offsets
# Index mapping: 0=Up, 1=Down, 2=Left, 3=Right
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
