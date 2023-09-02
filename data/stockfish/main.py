import logging
import os
import copy
import time
import shutil
import chess.engine
import chess.polyglot
import chess.svg
import tqdm
import random

output_root = "./"


def save_match(filename, board, moves):
    with open(f"{filename}.log", "a") as game_file:
        # game_file.write(f"\n{str(board)}\n\n")

        for i, move in enumerate(moves):
            game_file.write(f"{str(move)} ")

        game_file.write("\n")

    # boardsvg = chess.svg.board(board=board)
    # with open(f"{filename}.svg", "w") as image_file:
    #     image_file.write(boardsvg)


def play(engine: chess.engine.SimpleEngine):
    board = chess.Board()
    moves = []


    while not board.is_game_over():
        depth = random.choice([2, 3, 4])
        result = engine.play(board, chess.engine.Limit(depth=depth), ponder=False)
        board.push(result.move)
        moves.append(result.move)

    # random number
    filename = f"{output_root}/log"

    save_match(filename, board, moves)


def simulate():
    STOCKFISH_PATH = "/home/ubuntu/stockfish/src/stockfish"
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 6})
    engine.configure({"Hash": 4096})

    # delete log file
    if os.path.exists(f"{output_root}/log.log"):
        os.remove(f"{output_root}/log.log")

    for _ in tqdm.tqdm(range(1000)):
        play(
            engine,
        )

    engine.quit()


if __name__ == "__main__":
    simulate()
