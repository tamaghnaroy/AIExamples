"""
Main entry point for running prompt_universe as a module.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prompt Universe module entrypoint')
    parser.add_argument('--ui', action='store_true', help='Run the Dash UI')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Dash host')
    parser.add_argument('--port', type=int, default=8051, help='Dash port')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Enable Dash debug')
    parser.add_argument('--no-debug', dest='debug', action='store_false', help='Disable Dash debug')
    parser.set_defaults(debug=True)
    args, remaining = parser.parse_known_args()

    if args.ui:
        from FinAnalytics.AI.prompt_universe.dash_ui import run_ui
        run_ui(host=args.host, port=args.port, debug=bool(args.debug))
    else:
        from FinAnalytics.AI.prompt_universe.runner import main as runner_main
        sys.argv = [sys.argv[0]] + remaining
        runner_main()
