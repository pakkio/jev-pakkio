#!/usr/bin/env python
"""Play Doom in the terminal with the jev_pakkio scorer choosing every action.

Each decision step: ViZDoom renders a frame -> the frame is described in a few lines of
text -> the jev_pakkio server scores the scenario's action menu as continuations of that text
-> the highest-probability action is pressed for `--frame-skip` tics. Nothing is generated;
the model only ranks the options, which is the jevlike / System One idea.

    make serve                                  # in another terminal
    .venv/bin/python demo/doom/play.py          # model plays defend_the_center
    .venv/bin/python demo/doom/play.py --scenario deadly_corridor --api systemone

Keys: q quit, p pause/resume, space single-step, m toggle manual control,
      w/s forward/back, a/d turn, j/l strafe, f fire (manual mode).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import select
import sys
import termios
import textwrap
import time
import tty

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import Client, Decision, ServerDown  # noqa: E402
from describe import ACTION_HELP, describe, instructions, state_dict  # noqa: E402
from game import SCENARIOS, Doom  # noqa: E402
from render import Screen, bar  # noqa: E402

MANUAL_KEYS = {"w": "move forward", "s": "move backward", "a": "turn left", "d": "turn right",
               "j": "strafe left", "l": "strafe right", "f": "attack"}


class Keys:
    """Non-blocking single-key reads from a raw-mode tty."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.tty = os.isatty(self.fd)

    def __enter__(self) -> "Keys":
        if self.tty:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc) -> None:
        if self.tty:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def poll(self) -> list[str]:
        keys = []
        if not self.tty:
            return keys
        while select.select([self.fd], [], [], 0)[0]:
            keys.append(os.read(self.fd, 1).decode(errors="ignore"))
        return keys

    def wait(self) -> str:
        if not self.tty:
            return "q"
        return os.read(self.fd, 1).decode(errors="ignore")


def pick(decision: Decision, temperature: float) -> str:
    if temperature <= 0:
        return decision.best
    acts = list(decision.probs)
    w = [max(p, 1e-9) ** (1.0 / temperature) for p in decision.probs.values()]
    return random.choices(acts, weights=w, k=1)[0]


def decide(client: Client, api: str, snap, doom: Doom, last: str | None) -> Decision:
    if api == "score":
        return client.decide_score(describe(snap, doom.goal, doom.rules, doom.examples, last), doom.actions)
    return client.decide_systemone(state_dict(snap, last), instructions(doom.goal, doom.rules, doom.examples),
                                   {a: ACTION_HELP[a] for a in doom.actions})


def panel_lines(args, snap, decision: Decision | None, chosen: str | None, source: str, mode: str,
                episode: int, step: int, total_reward: float, note: str) -> list[str]:
    W = 43
    L = [f"jev_pakkio doom  [{args.scenario}]  {mode}",
         f"episode {episode}  step {step}  tic {snap.tic}",
         f"health {snap.health:3d}  ammo {snap.ammo:3d}  kills {snap.kills}  reward {total_reward:.0f}",
         ""]
    if decision is not None:
        L.append(f"/{args.api}  {decision.latency_s * 1000:.0f} ms")
        for a, p in sorted(decision.probs.items(), key=lambda kv: -kv[1]):
            mark = ">" if a == chosen else " "
            L.append(f"{mark} {a:<14} {bar(p, 16)} {p * 100:5.1f}%")
        L.append(f"pressed: {chosen}  ({source})")
        L.append("")
        shown = decision.context
        if "\nSituation:" in shown:  # /score: drop goal, rules and examples, keep the live situation
            shown = shown[shown.rindex("\nSituation:") + 1:].removesuffix("->")
        for line in shown.splitlines():
            L.extend(textwrap.wrap(line, W) or [""])
    if note:
        L.append("")
        L.append(note)
    L.append("")
    L.append("q quit  p pause  space step  m manual")
    L.append("manual: w/s move  a/d turn  j/l strafe  f fire")
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="defend_the_center", choices=list(SCENARIOS))
    ap.add_argument("--url", default=os.environ.get("OPENJEV_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--api", default="score", choices=["score", "systemone"],
                    help="score: native /score endpoint. systemone: TypeSafe /v1/systemone choice question")
    ap.add_argument("--api-key", default=None, help="bearer token for /v1/systemone (default: $OPENJEV_API_KEY)")
    ap.add_argument("--norm", default="mean", choices=["mean", "sum", "pmi"], help="/score normalisation")
    ap.add_argument("--frame-skip", type=int, default=5, help="game tics each chosen action is held for")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = argmax; >0 samples from the option probabilities")
    ap.add_argument("--episodes", type=int, default=0, help="stop after N episodes (0 = until q)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N decision steps (0 = unlimited)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--record", default=None, help="append one JSON line per decision (context, options, probs, action)")
    ap.add_argument("--manual", action="store_true", help="start in manual mode (model still shows its ranking)")
    ap.add_argument("--wad", default=None, help="path to a real IWAD (doom2.wad, doom.wad); default is the bundled freedoom2.wad")
    ap.add_argument("--map", default=None, help="map to load with --scenario level, e.g. map01 (Doom II) or e1m1 (Doom)")
    args = ap.parse_args()

    client = Client(args.url, api=args.api, api_key=args.api_key, norm=args.norm)
    try:
        h = client.health()
    except ServerDown as e:
        sys.exit(f"{e}\nStart the server first:  make serve   (or: .venv/bin/jev_pakkio serve)")
    if not h.get("ok"):
        sys.exit("server is up but the model is still loading; try again in a moment")

    if args.seed is not None:
        random.seed(args.seed)
    if (args.wad or args.map) and args.scenario != "level":
        args.scenario = "level"  # a real level only makes sense with the full action menu
    try:
        doom = Doom(args.scenario, seed=args.seed, wad=args.wad, map_name=args.map)
    except FileNotFoundError as e:
        sys.exit(f"WAD not found: {e}")
    rec = open(args.record, "a") if args.record else None

    manual = args.manual
    paused = False
    episode, step = 1, 0
    last: str | None = None
    note = ""
    doom.new_episode()

    with Screen() as screen, Keys() as keys:
        try:
            while True:
                if doom.finished:
                    snap_note = f"episode {episode} over: kills {doom_last_kills}  reward {doom.total_reward:.0f}"
                    screen.draw(last_frame, panel_lines(args, last_snap, last_decision, last_chosen, "", "OVER",
                                                        episode, step, doom.total_reward, snap_note + "   (any key)"))
                    if args.episodes and episode >= args.episodes:
                        break
                    k = keys.wait()
                    if k == "q":
                        break
                    episode += 1
                    step = 0
                    last = None
                    doom.new_episode()
                    continue

                snap = doom.snapshot()
                if snap is None:
                    continue
                last_frame, last_snap = snap.frame, snap
                doom_last_kills = snap.kills

                try:
                    decision = decide(client, args.api, snap, doom, last)
                except ServerDown as e:
                    note = str(e)
                    break
                last_decision = decision
                chosen, source = pick(decision, args.temperature), "model"

                # keyboard
                for k in keys.poll():
                    if k == "q":
                        raise KeyboardInterrupt
                    if k == "p":
                        paused = not paused
                    elif k == "m":
                        manual = not manual
                    elif k == " " and paused:
                        pass
                    elif manual and k in MANUAL_KEYS:
                        chosen, source = MANUAL_KEYS[k], "you"
                if manual and source == "model":
                    chosen, source = None, "waiting for a key"

                mode = "MANUAL" if manual else ("PAUSED" if paused else "MODEL PLAYS")
                screen.draw(snap.frame, panel_lines(args, snap, decision, chosen, source, mode, episode, step,
                                                    doom.total_reward, note))

                if paused:
                    k = keys.wait()
                    if k == "q":
                        break
                    if k == "p":
                        paused = False
                    if k == "m":
                        manual = not manual
                    if k != " ":
                        continue  # redraw / re-decide
                if manual and chosen is None:
                    k = keys.wait()
                    if k == "q":
                        break
                    if k == "m":
                        manual = False
                        continue
                    if k == "p":
                        paused = True
                        continue
                    chosen = MANUAL_KEYS.get(k)
                    if chosen is None:
                        continue
                    source = "you"

                reward = doom.step(chosen, args.frame_skip)
                last_chosen = chosen
                step += 1
                last = chosen
                if rec:
                    rec.write(json.dumps({"episode": episode, "step": step, "api": args.api,
                                          "context": decision.context, "options": list(decision.probs),
                                          "probabilities": list(decision.probs.values()),
                                          "action": chosen, "source": source, "reward": reward}) + "\n")
                if args.max_steps and step >= args.max_steps:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            doom.close()
            if rec:
                rec.close()
    if note:
        print(note)
    print(f"episodes {episode}  steps {step}  last reward {doom.total_reward:.0f}")


if __name__ == "__main__":
    main()
