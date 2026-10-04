"""Play chess against the fine-tuned Gemma from the terminal.

The model picks its move by ranking every legal move with jev_pakkio's
OptionScorer (the same one-pass scoring the server uses). You type moves in
UCI (e2e4) or SAN (Nf3, O-O). Commands: `moves` lists legal moves, `hint`
shows the model's top choices for your side, `undo` takes back a full move,
`fen` prints the position, `quit` exits.

Usage:
    .venv/bin/python finetune/play_chess.py                    # you are white
    .venv/bin/python finetune/play_chess.py --play black
    .venv/bin/python finetune/play_chess.py --fen "<fen>"      # start from a position
    .venv/bin/python finetune/play_chess.py --no-adapter       # base model, for comparison
"""

from __future__ import annotations

import argparse
import sys

import chess

from finetune.prepare_chess_data import board_ascii, make_prompt
from jev_pakkio.scorer import OptionScorer


def rank_moves(scorer: OptionScorer, board: chess.Board, top: int = 3) -> list[tuple[chess.Move, float]]:
    moves = list(board.legal_moves)
    if len(moves) == 1:
        return [(moves[0], 1.0)]
    scores = scorer.score(make_prompt(board), [f" {m.uci()}" for m in moves], norm="sum")
    ranked = sorted(zip(moves, scores), key=lambda t: t[1].probability, reverse=True)
    return [(m, s.probability) for m, s in ranked[:top]]


def parse_move(board: chess.Board, text: str) -> chess.Move | None:
    for parser in (board.parse_uci, board.parse_san):
        try:
            return parser(text)
        except ValueError:
            continue
    return None


def show(board: chess.Board) -> None:
    print()
    print(board_ascii(board))
    print(f"\n{'White' if board.turn else 'Black'} to move.", end="")
    if board.is_check():
        print(" Check!", end="")
    print()


def game_over_message(board: chess.Board) -> str:
    if board.is_checkmate():
        return f"Checkmate. {'Black' if board.turn else 'White'} wins."
    if board.is_stalemate():
        return "Stalemate."
    if board.is_insufficient_material():
        return "Draw by insufficient material."
    if board.can_claim_draw():
        return "Draw (repetition or fifty-move rule)."
    return "Game over."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gemma-3-4b-it")
    ap.add_argument("--adapter", default="adapters/chess-lora")
    ap.add_argument("--no-adapter", action="store_true", help="play the base model instead")
    ap.add_argument("--play", choices=["white", "black"], default="white", help="your colour")
    ap.add_argument("--fen", default=None, help="start position (default: initial position)")
    ap.add_argument("--top", type=int, default=3, help="candidate moves to display")
    args = ap.parse_args()

    adapter = None if args.no_adapter else args.adapter
    print(f"Loading {args.model}" + (f" + {adapter}" if adapter else "") + " ...", file=sys.stderr)
    scorer = OptionScorer(args.model, batch_size=16, adapter_path=adapter)

    board = chess.Board(args.fen) if args.fen else chess.Board()
    human = chess.WHITE if args.play == "white" else chess.BLACK
    print(f"You are {args.play}. Enter moves like e2e4 or Nf3. Type 'help' for commands.")

    while not board.is_game_over(claim_draw=True):
        show(board)
        if board.turn != human:
            ranked = rank_moves(scorer, board, args.top)
            move, p = ranked[0]
            san = board.san(move)
            alts = ", ".join(f"{board.san(m)} {q:.0%}" for m, q in ranked[1:])
            board.push(move)
            print(f"Model plays {san} ({p:.0%})" + (f"   [also considered: {alts}]" if alts else ""))
            continue

        try:
            text = input("> ").strip()
        except EOFError:
            print()
            return
        if not text:
            continue
        cmd = text.lower()
        if cmd in ("quit", "exit", "q"):
            return
        if cmd == "help":
            print("moves | hint | undo | fen | quit, or a move in UCI/SAN")
        elif cmd == "moves":
            print(" ".join(board.san(m) for m in board.legal_moves))
        elif cmd == "hint":
            for m, p in rank_moves(scorer, board, args.top):
                print(f"  {board.san(m):8s} {p:.0%}")
        elif cmd == "undo":
            if len(board.move_stack) >= 2:
                board.pop()
                board.pop()
            elif board.move_stack:
                board.pop()
            else:
                print("nothing to undo")
        elif cmd == "fen":
            print(board.fen())
        else:
            move = parse_move(board, text)
            if move is None:
                print(f"'{text}' is not a legal move here. Type 'moves' to list them.")
            else:
                board.push(move)

    show(board)
    print(game_over_message(board))


if __name__ == "__main__":
    main()
