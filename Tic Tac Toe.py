def display_board(board):
    print("\n")
    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("---|---|---")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("---|---|---")
    print(f" {board[6]} | {board[7]} | {board[8]} ")
    print("\n")

def check_winner(board, player):
    winning_combinations = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6)
    ]

    for a, b, c in winning_combinations:
        if board[a] == board[b] == board[c] == player:
            return True

    return False

def check_draw(board):
    for spot in board:
        if spot not in ["X", "O"]:
            return False
    return True

def play_game():
    board = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    current_player = "X"

    print("TIC-TAC-TOE")
    print("Player 1: X")
    print("Player 2: O")

    while True:
        display_board(board)
        print(f"Player {current_player}'s turn")

        try:
            choice = int(input("Enter position (1-9): "))
        except ValueError:
            print("Invalid input! Please enter a number between 1 and 9.")
            continue

        if choice < 1 or choice > 9:
            print("Out of range! Please choose a number from 1 to 9.")
            continue

        index = choice - 1

        if board[index] in ["X", "O"]:
            print("That position is already taken! Pick another one.")
            continue

        board[index] = current_player

        if check_winner(board, current_player):
            display_board(board)
            print(f"Player {current_player} wins!")
            break

        if check_draw(board):
            display_board(board)
            print("Game Draw!")
            break

        if current_player == "X":
            current_player = "O"
        else:
            current_player = "X"

play_game()

#Player X's turn
#Enter position (1-9): 1


# X | 2 | 3 
#---|---|---
# O | X | 6 
#---|---|---
# O | 8 | X 


#Player X wins!