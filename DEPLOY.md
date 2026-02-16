# Run from server

## Run with tmux

```bash
tmux new -s wiki_voyager
cd ~/wiki_voyager
conda activate wiki_voyager
python src/run.py
```

### tmux cheatsheet

| Action | Keys |
|---|---|
| Detach (leave running) | `Ctrl+B` then `D` |
| Reattach | `tmux attach -t wikibot` |
| List sessions | `tmux ls` |
| Kill session | `tmux kill-session -t wikibot` |
