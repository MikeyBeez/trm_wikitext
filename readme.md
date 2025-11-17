TRM-WikiText103

Character-level chunked autoregressive transformer with recursive refinement. Experimental implementation exploring iterative token refinement mechanisms.
Files

TRM-WikiText103.py: Main training script (character-level tokenization)
get_wtext.py: Dataset download utility
trm_wikitext.py: Alternative implementation
trm_wk3.py: Alternative implementation
.gitignore: Version control exclusions
Quick Start

python get_wtext.py     # Download WikiText-103 dataset
python TRM-WikiText103.py  # Train character-level TRM model

Notes

Character-level tokenization: each character is a token
Chunk size: 2 tokens predicted simultaneously
No experimental results available yet (first run pending)
See docstrings in TRM-WikiText103.py for parameter details
