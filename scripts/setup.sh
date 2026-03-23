#!/bin/bash
# phi-auto setup script
echo "=== phi-auto Setup ==="
echo "Checking dependencies..."

python3 -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null || {
    echo "Installing numpy..."
    pkg install python-numpy -y
}

echo "Creating cache directories..."
mkdir -p ~/.cache/phi-auto/data

echo "Testing model..."
cd "$(dirname "$0")/.."
python3 -c "
from src.engine.model import GPT, GPTConfig
cfg = GPTConfig(vocab_size=256, n_embd=64, n_head=4, n_layer=2, seq_len=32)
model = GPT(cfg)
import numpy as np
x = np.random.randint(0, 256, (1, 32))
y = np.random.randint(0, 256, (1, 32))
_, loss = model.forward(x, y)
model.backward()
print(f'Sanity check passed. Loss: {loss:.4f}')
"

echo ""
echo "Setup complete! Run: python3 src/engine/train.py"
