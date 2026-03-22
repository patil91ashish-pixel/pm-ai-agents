"""
evals/__init__.py
─────────────────
Makes the evals/ directory a Python package so that eval_runner.py
can import from eval_functional and eval_scorer using:

    from evals.eval_functional import run_all_functional_checks
    from evals.eval_scorer import score_multiple_posts
"""
