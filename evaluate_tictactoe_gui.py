# evaluate_tictactoe_gui.py
import torch
import tkinter as tk
from tictactoe_env import TicTacToe
from dqn_model import DQN

class TicTacToeGUI:
    def __init__(self, model_path):
        self.env = TicTacToe()
        self.model = DQN()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.window = tk.Tk()
        self.window.title("Tic-Tac-Toe with AI")
        self.buttons = []
        self.current_player = 1  # Human starts
        self.create_ui()
        self.window.mainloop()

    def create_ui(self):
        for i in range(9):
            button = tk.Button(self.window, text="", width=10, height=3, 
                               command=lambda i=i: self.on_click(i))
            button.grid(row=i//3, column=i%3)
            self.buttons.append(button)
        reset_button = tk.Button(self.window, text="Reset", command=self.reset_game)
        reset_button.grid(row=3, column=1)

    def on_click(self, idx):
        if self.env.board[idx//3, idx%3] == 0 and not self.env.done:
            self.buttons[idx].config(text="X")
            state, reward, done = self.env.step(idx, 1)
            if done:
                self.display_result(reward)
                return

            # AI takes its turn
            ai_action = self.get_ai_action(state)
            state, reward, done = self.env.step(ai_action, -1)
            self.buttons[ai_action].config(text="O")
            if done:
                self.display_result(reward)

    def get_ai_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        q_values = q_values.detach().numpy().flatten()
        legal_actions = [i for i in range(9) if state.numpy()[0][i] == 0]
        # Choose the action with the highest Q-value
        best_action = max(legal_actions, key=lambda a: q_values[a])
        return best_action

    def display_result(self, reward):
        if reward == 1:
            result = "You win!"
        elif reward == -1:
            result = "AI wins!"
        else:
            result = "It's a draw!"
        result_label = tk.Label(self.window, text=result, font=("Arial", 24))
        result_label.grid(row=3, column=1)
        self.env.done = True

    def reset_game(self):
        self.env.reset()
        for button in self.buttons:
            button.config(text="")
        self.env.done = False

if __name__ == "__main__":
    model_path = "tic_tac_toe_dqn.pth"  # Path to your trained model
    TicTacToeGUI(model_path)
