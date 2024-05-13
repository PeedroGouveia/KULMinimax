import pyspiel

import numpy as np
import time



class MinimaxTransposition:

    def __init__(self, game):
        params = game.get_parameters()
        self.game = game
        self.num_rows = params['num_rows']
        self.num_cols = params['num_cols']
        self.total_actions = len(self.game.new_initial_state().legal_actions())
        self.len_initial_tensor = len(self.game.new_initial_state().observation_tensor())

        self.player_0_index_to_action = {}
        for i in range(0, self.total_actions):
            state = game.new_initial_state()
            initial_tensor = state.observation_tensor()
            state.apply_action(i)
            final_tensor = state.observation_tensor()
            for j in range(self.len_initial_tensor // 3, self.len_initial_tensor):
                if (initial_tensor[j] != final_tensor[j]):
                    self.player_0_index_to_action[j] = i

        self.player_1_index_to_action = {int(self.len_initial_tensor * (2 / 3)): 0}
        for i in range(1, self.total_actions):
            state = game.new_initial_state()
            state.apply_action(0)
            initial_tensor = state.observation_tensor()
            state.apply_action(i)
            final_tensor = state.observation_tensor()
            for j in range(self.len_initial_tensor // 3, self.len_initial_tensor):
                if initial_tensor[j] != final_tensor[j]:
                    self.player_1_index_to_action[j] = i

        self.num_horizontal_lines = (self.num_rows + 1) * self.num_cols
        self.num_vertical_lines = self.num_rows * (self.num_cols + 1)

        self.transposition_table = {}

        self.name = "Minimax Transposition"

    def form_game_string(self, state):
        state_custom_string = "0" * self.total_actions
        state_custom_list = list(state_custom_string)
        for key in self.player_0_index_to_action.keys():
            tensor = state.observation_tensor()
            if tensor[key] == 1:
                state_custom_list[self.player_0_index_to_action[key]] = "1"

        for key in self.player_1_index_to_action.keys():
            tensor = state.observation_tensor()
            if tensor[key] == 1:
                state_custom_list[self.player_1_index_to_action[key]] = "2"

        state_custom_string = "".join(state_custom_list)
        return state_custom_string

    def state_to_key(self, state):
        # Using transposition tables with symmetries
        # Total keys: 212
        # Execution time: 0.19 seconds
        # Generate a key for the state using its canonical form and current player.
        action_string = np.array(self.form_game_string(state))
        return tuple(action_string.flatten()), state.current_player()

    def _minimax(self, state, maximizing_player_id, depth=0, iteration=0):
        if state.is_terminal():
            return state.player_return(maximizing_player_id), None

        key = self.state_to_key(state)

        if key in self.transposition_table:
            return self.transposition_table[key]

        player = state.current_player()
        if player == maximizing_player_id:
            selection = max
        else:
            selection = min
        values_children = [self._minimax(state.child(action), maximizing_player_id) for action in state.legal_actions()]

        self.transposition_table[key] = selection(values_children)

        return selection(values_children)

    def minimax_search(self, game,
                       state=None,
                       maximizing_player_id=None,
                       state_to_key=lambda state: state):
        """Solves deterministic, 2-players, perfect-information 0-sum game.

        For small games only! Please use keyword arguments for optional arguments.

        Arguments:
          game: The game to analyze, as returned by `load_game`.
          state: The state to run from.  If none is specified, then the initial state is assumed.
          maximizing_player_id: The id of the MAX player. The other player is assumed
            to be MIN. The default (None) will suppose the player at the root to be
            the MAX player.

        Returns:
          The value of the game for the maximizing player when both player play optimally.
        """

        start_time = time.time()


        game_info = game.get_type()

        if game.num_players() != 2:
            raise ValueError("Game must be a 2-player game")
        if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
            raise ValueError("The game must be a Deterministic one, not {}".format(
                game.chance_mode))
        if game_info.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
            raise ValueError(
                "The game must be a perfect information one, not {}".format(
                    game.information))
        if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError("The game must be turn-based, not {}".format(
                game.dynamics))
        if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
            raise ValueError("The game must be 0-sum, not {}".format(game.utility))

        if state is None:
            state = game.new_initial_state()
        if maximizing_player_id is None:
            maximizing_player_id = state.current_player()
        v = self._minimax(
            state.clone(),
            maximizing_player_id=maximizing_player_id,
        )
        total_keys = len(self.transposition_table)
        execution_time = time.time() - start_time
        return v, total_keys, execution_time

