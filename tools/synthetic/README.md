Synthesize dataset with background pool and foreground pool.

Before you use this, please take a look at codes to see what will be done.

### 1. Split train/val/test
```bash
./1.split_pool.sh
```

### 2. Synthesize BG pool
```bash
./2.synthesize_bg.sh
```

### 3. Synthesize dataset
```bash
./3.synthesize_dataset.sh
```