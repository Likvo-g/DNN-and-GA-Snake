import pygame as pg
import random
from config import *
import numpy as np
from neuroNetwork import neuroNet
import os


class Snake:
    """
    Snake agent that can move autonomously and perceive its environment.

    This class represents an AI-controlled snake that uses a neural network
    to make movement decisions based on environmental state.

    Attributes:
        body (list): List of (x, y) tuples representing snake body segments,
                    with body[0] being the head position
        direction (tuple): Current movement direction as (dx, dy) tuple
        score (int): Current score (number of food items eaten)
        steps (int): Total number of moves made in current game
        dead (bool): Flag indicating whether snake has died
        uniq (list): History buffer to detect infinite loops during gameplay
        board_x (int): Width of the game board in grid units
        board_y (int): Height of the game board in grid units
        nn (neuroNet): Neural network instance for decision making
        color (tuple): RGB color tuple for rendering snake body
    """

    def __init__(self, head, direction, genes, board_x, board_y):
        """
        Initialize a new snake instance.

        Args:
            head (tuple): Initial position of snake head as (x, y)
            direction (tuple): Initial movement direction as (dx, dy)
            genes (np.array): Weight parameters for neural network
            board_x (int): Board width in grid units
            board_y (int): Board height in grid units
        """
        # Initialize snake body with only head segment
        self.body = [head]
        self.direction = direction

        # Initialize game statistics
        self.score = 0
        self.steps = 0
        self.dead = False

        # Initialize loop detection system - tracks unique game states
        self.uniq = [0] * board_x * board_y

        # Store board dimensions for boundary checking
        self.board_x = board_x
        self.board_y = board_y

        # Create neural network with provided genetic weights
        self.nn = neuroNet(N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT, genes.copy())

        # Generate random color for visual distinction
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def move(self, food):
        """
        Execute one movement step using neural network decision.

        The snake observes its environment, feeds the state to its neural network,
        receives a movement decision, and executes that move. Handles collision
        detection, food consumption, and infinite loop prevention.

        Args:
            food (tuple): Current food position as (x, y) coordinates

        Returns:
            bool: True if snake ate food this turn, False otherwise
        """
        # Increment step counter for this game session
        self.steps += 1

        # Get current environmental state as neural network input
        state = self.get_state(food)

        # Use neural network to predict best action (0-3 for four directions)
        action = self.nn.predict_next_action(state)

        # Convert action index to direction vector
        self.direction = DIRECTIONS[action]

        # Calculate new head position after movement
        head = (self.body[0][0] + self.direction[0], self.body[0][1] + self.direction[1])

        # Track whether food was consumed this turn
        has_eat = False

        # Check for collision conditions (wall collision or self-collision)
        if (head[0] < 0 or head[0] >= self.board_x or head[1] < 0 or head[1] >= self.board_y
                or head in self.body[:-1]):
            # Snake hit wall or collided with its own body
            self.dead = True
        else:
            # Valid move - add new head to front of body
            self.body.insert(0, head)

            if head == food:
                # Snake ate food - increase score and keep tail (snake grows)
                self.score += 1
                has_eat = True
            else:
                # Normal move - remove tail to maintain length
                self.body.pop()

                # Infinite loop detection: check if current state has been seen before
                if (head, food) not in self.uniq:
                    # New state - add to history and remove oldest entry
                    self.uniq.append((head, food))
                    del self.uniq[0]
                else:
                    # State repetition detected - kill snake to prevent infinite loop
                    self.dead = True

        return has_eat

    def get_state(self, food):
        """
        Generate environmental state vector for neural network input.

        Creates a comprehensive representation of the snake's current situation
        including head direction, tail direction, and 8-directional vision system
        that detects walls, food, and body segments.

        Args:
            food (tuple): Current food position as (x, y) coordinates

        Returns:
            list: State vector of length N_INPUT (32) containing:
                  - Head direction (4 values, one-hot encoded)
                  - Tail direction (4 values, one-hot encoded)
                  - Vision data (24 values, 3 per direction × 8 directions)
        """
        # Encode current head direction as one-hot vector
        i = DIRECTIONS.index(self.direction)
        head_dir = [0.0, 0.0, 0.0, 0.0]
        head_dir[i] = 1.0

        # Encode tail direction for body awareness
        if len(self.body) == 1:
            # Single segment - tail direction same as head direction
            tail_direction = self.direction
        else:
            # Calculate tail direction from last two body segments
            tail_direction = (self.body[-2][0] - self.body[-1][0], self.body[-2][1] - self.body[-1][1])

        i = DIRECTIONS.index(tail_direction)
        tail_dir = [0.0, 0.0, 0.0, 0.0]
        tail_dir[i] = 1.0

        # Combine directional information
        state = head_dir + tail_dir

        # Define 8-directional vision system (N, NE, E, SE, S, SW, W, NW)
        dirs = [[0, -1], [1, -1], [1, 0], [1, 1],
                [0, 1], [-1, 1], [-1, 0], [-1, -1]]

        # Cast vision rays in each direction until hitting boundary
        for dir in dirs:
            # Start from head position
            x = self.body[0][0] + dir[0]
            y = self.body[0][1] + dir[1]

            # Initialize vision variables
            dis = 1.0  # Distance to boundary (will be normalized)
            see_food = 0.0  # Binary flag for food detection
            see_self = 0.0  # Binary flag for body segment detection

            # Cast ray until hitting board boundary
            while x >= 0 and x < self.board_x and y >= 0 and y < self.board_y:
                if (x, y) == food:
                    see_food = 1.0  # Food detected in this direction
                elif (x, y) in self.body:
                    see_self = 1.0  # Body segment detected in this direction

                # Move ray one step further
                dis += 1
                x += dir[0]
                y += dir[1]

            # Add normalized distance and detection flags to state
            # 1/dis provides proximity measure (closer objects have higher values)
            state += [1.0 / dis, see_food, see_self]

        return state


class Game:
    """
    Game environment that manages multiple snakes and game mechanics.

    Handles game loop, food placement, collision detection, rendering,
    and coordination between multiple AI snakes competing simultaneously.

    Attributes:
        X (int): Number of columns in game board
        Y (int): Number of rows in game board
        show (bool): Whether to render game visually using pygame
        seed (int): Random seed for reproducible game sessions
        rand (Random): Seeded random number generator for consistent behavior
        snakes (list): List of Snake instances participating in game
        food (tuple): Current food position, None if board is full
        best_score (int): Highest score achieved by any snake in current session
        width (int): Pixel width of game window (only if show=True)
        height (int): Pixel height of game window (only if show=True)
        screen: Pygame display surface (only if show=True)
        clock: Pygame clock for frame rate control (only if show=True)
        font_name (str): Font name for text rendering (only if show=True)
    """

    def __init__(self, genes_list, seed=None, show=False, rows=ROWS, cols=COLS):
        """
        Initialize game environment with specified parameters.

        Args:
            genes_list (list): List of neural network weight arrays, one per snake
            seed (int, optional): Random seed for reproducibility. Auto-generated if None
            show (bool): Whether to enable visual rendering with pygame
            rows (int): Number of rows in game board
            cols (int): Number of columns in game board
        """
        # Store board dimensions
        self.Y = rows
        self.X = cols
        self.show = show

        # Setup random seed for reproducible games
        self.seed = seed if seed is not None else random.randint(-INF, INF)
        self.rand = random.Random(self.seed)

        # Create snake instances with random starting positions and directions
        self.snakes = []
        board = [(x, y) for x in range(self.X) for y in range(self.Y)]

        for genes in genes_list:
            # Random starting position anywhere on board
            head = self.rand.choice(board)
            # Random initial direction (up/down/left/right)
            direction = DIRECTIONS[self.rand.randint(0, 3)]
            self.snakes.append(Snake(head, direction, genes, self.X, self.Y))

        # Place initial food and initialize scoring
        self.food = self._place_food()
        self.best_score = 0

        # Initialize pygame components if visual display is enabled
        if show:
            pg.init()
            self.width = cols * GRID_SIZE
            self.height = rows * GRID_SIZE + BLANK_SIZE

            pg.display.set_caption(TITLE)
            self.screen = pg.display.set_mode((self.width, self.height))
            self.clock = pg.time.Clock()
            self.font_name = pg.font.match_font(FONT_NAME)

    def play(self):
        """
        Execute complete game session until all snakes are dead.

        Main game loop that alternates between snake movements, food management,
        score tracking, and optional visual rendering. Continues until no snakes
        remain alive or board becomes completely filled.

        Returns:
            tuple: (scores, steps, seed) where:
                  - scores: Final scores (int if single snake, list if multiple)
                  - steps: Total steps taken (int if single snake, list if multiple)
                  - seed: Random seed used for this game session
        """
        # Track living snakes using set for efficient removal
        alive_snakes_set = set(self.snakes)

        # Continue game while snakes are alive and food can be placed
        while alive_snakes_set and self.food is not None:
            # Handle pygame events and rendering if visual mode enabled
            if self.show:
                self._handle_events()
                self.render()

            # Process movement for each living snake
            for snake in alive_snakes_set:
                # Execute one movement step for this snake
                has_eat = snake.move(self.food)

                if has_eat:
                    # Snake ate food - place new food on board
                    self.food = self._place_food()
                    if self.food is None:
                        # Board completely filled - end game
                        break

                # Update best score tracking
                if snake.score > self.best_score:
                    self.best_score = snake.score

            # Remove dead snakes from active set
            alive_snakes = [snake for snake in alive_snakes_set if not snake.dead]
            alive_snakes_set = set(alive_snakes)

        # Prepare return values based on number of snakes
        if len(self.snakes) > 1:
            # Multiple snakes - return lists of scores and steps
            score = [snake.score for snake in self.snakes]
            steps = [snake.steps for snake in self.snakes]
        else:
            # Single snake - return individual values
            score, steps = self.snakes[0].score, self.snakes[0].steps

        return score, steps, self.seed

    def _place_food(self):
        """
        Find an empty position on the board to place food.

        Scans all board positions and excludes those occupied by any living snake.
        Returns None if no empty positions are available (board completely filled).

        Returns:
            tuple or None: (x, y) coordinates of new food position, or None if board full
        """
        # Generate set of all possible board positions
        board = set([(x, y) for x in range(self.X) for y in range(self.Y)])

        # Remove positions occupied by living snakes
        for snake in self.snakes:
            if not snake.dead:
                for body in snake.body:
                    board.discard(body)

        # Check if any empty positions remain
        if len(board) == 0:
            return None  # Board completely filled

        # Randomly select from available empty positions
        return self.rand.choice(list(board))

    def render(self):
        """
        Render current game state using pygame.

        Draws all visual elements including snake bodies, food, grid lines,
        and status information. Only called when show=True.
        """
        # Clear screen with black background
        self.screen.fill(BLACK)

        # Helper function to convert grid coordinates to pixel coordinates
        get_xy = lambda pos: (pos[0] * GRID_SIZE, pos[1] * GRID_SIZE + BLANK_SIZE)

        # Draw all living snakes
        num = 0  # Count of living snakes for display
        for snake in self.snakes:
            if not snake.dead:
                num += 1

                # Draw snake
                for s in snake.body[0:]:
                    x, y = get_xy(s)
                    pg.draw.rect(self.screen, snake.color, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))

        # Draw food as red square
        x, y = get_xy(self.food)
        pg.draw.rect(self.screen, RED, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))

        # Draw status text showing living snake count and best score
        text = str(self.best_score)
        font = pg.font.Font(self.font_name, 20)
        text_surface = font.render(text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.midtop = ((self.width / 2, 5))
        self.screen.blit(text_surface, text_rect)

        # Update display with all drawn elements
        pg.display.flip()

    def _handle_events(self):
        """
        Handle pygame events and maintain frame rate.

        Processes user input (mainly window close events) and controls
        game speed through frame rate limiting.
        """
        # Limit frame rate to specified FPS
        self.clock.tick(FPS)

        # Process all pending pygame events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                # User clicked window close button
                pg.quit()
                quit()


def run(score):
    """
    Load and replay the best performing neural network model.

    Loads saved neural network weights and game seed that achieved the specified
    score, then replays that exact game scenario with visual display.

    Args:
        score (int): Target score to load - specifies which saved model to use
    """
    # Load neural network weights from saved file
    genes_pth = os.path.join("genes", "best", str(score))
    with open(genes_pth, "r") as f:
        genes = np.array(list(map(float, f.read().split())))

    # Load corresponding random seed for exact game reproduction
    seed_pth = os.path.join("seed", str(score))
    with open(seed_pth, "r") as f:
        seed = int(f.read())

    # Create and run game with loaded parameters and visual display
    game = Game(show=True, genes_list=[genes], seed=seed)
    game.play()

if __name__ == '__main__':
    # Example usage - uncomment desired function
    run(310)  # Run best model that achieved score 292
