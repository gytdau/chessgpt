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
from multiprocessing import Pool, Lock, Value, current_process

output_root = "./"
STOCKFISH_PATH = "/home/ubuntu/stockfish/src/stockfish"


def init(l):
    global lock
    lock = l
    global engine
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1})
    engine.configure({"Hash": 4096})


def save_match(filename, board, moves):
    with lock:
        with open(f"{filename}.log", "a") as game_file:
            game_file.write(" ".join(str(move) for move in moves) + "\n")


def play(dummy_arg):  # the argument is a dummy one, required for the imap_unordered
    board = chess.Board()
    moves = []

    while not board.is_game_over():
        depth = random.choice([2, 3, 4])
        result = engine.play(board, chess.engine.Limit(depth=depth), ponder=False)
        board.push(result.move)
        moves.append(result.move)

    filename = f"{output_root}/log"
    save_match(filename, board, moves)

    return 1  # just return 1 for every game completed


def simulate():
    l = Lock()
    num_processes = os.cpu_count() - 1
    pool = Pool(processes=num_processes, initializer=init, initargs=(l,))

    total_games = 10000
    with tqdm.tqdm(total=total_games) as pbar:
        for _ in pool.imap_unordered(play, [None] * total_games):  # passing dummy values
            pbar.update()

    pool.close()
    pool.join()


if __name__ == "__main__":
    simulate()
