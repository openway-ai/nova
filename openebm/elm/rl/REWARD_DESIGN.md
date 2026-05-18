# Sudoku RL Reward Function Design

## Overview

This document explains the multi-component reward function for Sudoku reinforcement learning training.

## Reward Components

### Total Reward Range: [0.0, 3.0]

1. **Format Reward (0.0 - 0.5)**
   - Can the model output be parsed as a valid 9x9 grid?
   - Binary: 0.5 if parseable, 0.0 otherwise
   - Purpose: Encourage structured output

2. **Clue Preservation (0.0 - 0.5)**
   - Are all given clues (non-zero cells in puzzle) unchanged?
   - Formula: `(preserved_clues / total_clues) × 0.5`
   - Purpose: **Prevent the model from changing the original puzzle**
   - Critical: Without this, model could generate valid sudoku solutions for different puzzles

3. **Blank Accuracy (0.0 - 1.5)**
   - Fraction of blank cells filled correctly
   - Formula: `(correct_blanks / total_blanks) × 1.5`
   - Purpose: Main learning signal for filling in the solution
   - Weighted highest (1.5) because this is the core task

4. **Full Solve Bonus (0.0 - 0.5)**
   - Perfect match with ground truth solution
   - Binary: 0.5 if all 81 cells match, 0.0 otherwise
   - Purpose: Extra reward for complete correctness
   - Note: This implicitly validates sudoku constraints

## Design Rationale

### Why Remove "Validity Bonus"?

The original design had a separate "validity bonus" that checked if the board satisfied sudoku constraints (no duplicates in rows/columns/boxes). We removed it because:

1. **Logical Redundancy**
   - If `clue_preservation = 1.0` AND `blank_accuracy = 1.0` → `full_solve = 1.0` → validity is guaranteed
   - A correct solution must be valid

2. **Misleading Signal**
   - A board could be "valid" but solve a different puzzle (if clues were changed)
   - Example: Model changes first clue from 5→9, then fills rest correctly for that new puzzle
   - Old design: Gets validity bonus despite solving wrong puzzle!
   - New design: Loses clue_preservation reward, no validity bonus

3. **Simplicity**
   - Fewer components = clearer learning signal
   - Clue preservation + blank accuracy already capture everything needed

### Why Clue Preservation is Critical

Consider this scenario:
```
Original puzzle: 530070000... (starts with 5,3,0,...)
Model output:    930070000... (changed 5→9)
```

Old design:
- Format: ✓ 0.5
- Accuracy: ✓ 1.5 (if it fills blanks correctly for the NEW puzzle)
- Validity: ✓ 0.5 (satisfies constraints)
- Full solve: ✗ 0.0
- **Total: 2.5** (seems good but it's completely wrong!)

New design:
- Format: ✓ 0.5
- Clue preservation: ✗ 0.48 (30/31 clues preserved)
- Blank accuracy: ✓ 1.5
- Full solve: ✗ 0.0
- **Total: 2.48** (penalized for changing clue)

More importantly, the model learns that **changing clues is bad** through the explicit clue_preservation signal.

## Example Scenarios

### Scenario 1: Perfect Solution
```python
Puzzle:  530070000600195000...
Output:  534678912672195348...
Solution: 534678912672195348...

Rewards:
  Format: 0.50 ✓
  Clue preservation: 0.50 ✓ (31/31)
  Blank accuracy: 1.50 ✓ (50/50)
  Full solve: 0.50 ✓
  Total: 3.00
```

### Scenario 2: Changed One Clue
```python
Puzzle:  530070000600195000...
Output:  934678912672195348... (5→9 at position 0)
Solution: 534678912672195348...

Rewards:
  Format: 0.50 ✓
  Clue preservation: 0.48 ✗ (30/31)
  Blank accuracy: 1.50 (depends on blanks)
  Full solve: 0.00 ✗
  Total: 2.48
```

### Scenario 3: Partial Solution (98% Blanks Correct)
```python
Puzzle:  530070000600195000...
Output:  534678912672195348... (1 blank wrong)
Solution: 534678912672195348...

Rewards:
  Format: 0.50 ✓
  Clue preservation: 0.50 ✓ (31/31)
  Blank accuracy: 1.47 ~ (49/50)
  Full solve: 0.00 ✗
  Total: 2.47
```

### Scenario 4: Invalid Format
```python
Puzzle:  530070000600195000...
Output:  "I cannot solve this puzzle" (unparseable)
Solution: 534678912672195348...

Rewards:
  Format: 0.00 ✗
  Clue preservation: 0.00 (no board to check)
  Blank accuracy: 0.00
  Full solve: 0.00
  Total: 0.00
```

## Implementation Details

### Data Formats
- **Puzzle**: 81-character string, '0' = blank (e.g., `"530070000600195000..."`)
- **Solution**: 9×9 list of lists (e.g., `[[5,3,4,...], [6,7,2,...], ...]`)
- **Completion**: Text with newlines (e.g., `"5 3 4 6 7 8 9 1 2\n6 7 2..."`)

### Code Location
```python
from openebm.elm.rl.rewards import compute_sudoku_rewards_detailed

rewards = compute_sudoku_rewards_detailed(
    completions=["5 3 4 6 7 8 9 1 2\n..."],
    puzzles=["530070000600195000..."],
    solutions=[[[5,3,4,...], [6,7,2,...], ...]]
)
# Returns: [{'total': 3.0, 'format': 0.5, 'clue_preservation': 0.5,
#           'blank_accuracy': 1.5, 'full_solve': 0.5}]
```

## Training Implications

### Reward Shaping
The weights (0.5, 0.5, 1.5, 0.5) are designed to:
1. **Format (0.5)**: Basic requirement, binary signal
2. **Clue preservation (0.5)**: Critical constraint, must be satisfied
3. **Blank accuracy (1.5)**: Main optimization target, highest weight
4. **Full solve (0.5)**: Bonus for perfection, encourages going all the way

### Expected Learning Curve
1. **Early training**: Model learns to output valid 9×9 grids (format reward)
2. **Mid training**: Model learns to preserve clues and fill some blanks correctly
3. **Late training**: Model optimizes blank accuracy, increasing full solve rate

### Baseline vs Target

**SFT Baseline** (before RL):
- Format accuracy: ~95%
- Clue preservation: ~99%
- Blank accuracy: ~85%
- Full solve rate: ~60%
- Average reward: ~2.4

**RL Target** (after GRPO):
- Format accuracy: ~99%
- Clue preservation: ~99.5%
- Blank accuracy: ~92%+
- Full solve rate: ~75%+
- Average reward: ~2.7+

## Comparison with Original Design

| Component | Old Design | New Design | Reason for Change |
|-----------|------------|------------|-------------------|
| Format | 0.5 | 0.5 | ✓ Keep |
| Accuracy | 1.5 (all cells) | 1.5 (blanks only) | ✓ More precise |
| Validity | 0.5 (constraints) | **Removed** | ✗ Redundant + misleading |
| Clue preservation | **Not present** | 0.5 | ✓ **Critical addition** |
| Full solve | 0.5 | 0.5 | ✓ Keep |

**Key Insight**: The old "validity bonus" could reward wrong solutions. The new "clue preservation" explicitly prevents this failure mode.

## Testing

See test cases in the implementation:
```bash
python -m pytest openebm/elm/rl/test_rewards.py -v
```

Or run inline test:
```bash
python -c "from openebm.elm.rl.rewards import compute_sudoku_rewards_detailed; ..."
```

---

**Last updated**: May 16, 2025
**Status**: ✅ Tested and verified
