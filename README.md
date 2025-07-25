# GPU-Tic-Tac-Toe
### 🔧 **Project Idea: GPU Tic-Tac-Toe**

* 2 GPUs act as players.
* Each GPU runs a kernel to decide its move based on some strategy.
* The host coordinates turns, validates moves, and tracks the board.
* Board is stored as a simple 3x3 array (values: 0=empty, 1=GPU1, 2=GPU2).
* Output: A series of moves that can be replayed manually (e.g., in PowerPoint or video).

---

### 🧠 Strategy Options (Different per GPU)

| GPU   | Strategy Idea                                             |
| ----- | --------------------------------------------------------- |
| GPU 0 | Random valid move                                         |
| GPU 1 | Rule-based: win-if-possible, block-if-needed, else random |

---

### 📁 Project Structure

```bash
gpu_game/
│
├── gpu_tictactoe.cu        # Main CUDA program
├── board_utils.h/.cpp      # Board logic (host-side)
├── strategies.cu           # Device functions for move strategies
├── output.txt              # Record of moves (to replay)
├── slides/                 # PowerPoint or image-based slides of each move
└── presentation.mp4        # Final video recording of explanation + demo
```

---

### 🧩 Logic Flow

#### 1. **Board Representation**

```cpp
int board[3][3]; // 0: empty, 1: GPU0, 2: GPU1
```

#### 2. **Host Loop for Turns**

```cpp
for (int turn = 0; turn < 9 && !game_over(board); turn++) {
    int current_player = turn % 2;
    cudaSetDevice(current_player); // GPU0 or GPU1
    launch_kernel_to_compute_move<<<1, 1>>>(...);
    cudaMemcpy(...); // Get move from device
    apply_move(board, move);
    save_move_to_file(board, move, turn);
}
```

#### 3. **Kernel Per GPU**

Each GPU has its own kernel and decision-making logic.

```cpp
__global__ void make_move(int* board, int* move, int player_id) {
    // Strategy: Analyze board and write best move to `move`
}
```

#### 4. **Output Moves**

* Store each board state in a file (CSV, text, etc.).
* Use these to recreate the sequence manually in PowerPoint or slides.

---

### 🎥 **Video Presentation Guidelines**

**Length:** \~5–7 minutes
**Sections:**

1. **Intro (30 sec)**: Overview of what you're building.
2. **How the game works (2 mins)**:

   * Rules of the game.
   * How GPUs make decisions.
3. **Code Walkthrough (2–3 mins)**:

   * Show board logic.
   * Show GPU strategy kernel.
   * Show turn management via CUDA.
4. **Demo (1–2 mins)**:

   * Walk through the moves.
   * Highlight where each GPU made its move.
   * Show who wins.
5. **Conclusion**:

   * Challenges faced.
   * Potential improvements (e.g., scaling to Connect-4, Minimax strategy).

---

### 📤 Submission

* Video format: `.mp4`, `.mov`, `.m4a`, etc.
* Optional: Include your code repo link or ZIP.
* Optional: Add narrated audio or captions.

---

### ✅ Tips for Success

* Use **`cudaSetDevice()`** correctly to switch between GPUs.
* Make sure both GPUs are doing **real computation** (not just random moves).
* Use **`cudaMemcpy`** or unified memory to pass board state between host and device.
* Keep logic modular – isolate board handling and strategy functions.

---

### 📌 Bonus Ideas (if you want to go further)

* Add replay via ASCII animation in the terminal.
* Use 4x4 or Connect Four for a slightly more advanced challenge.
* Use Minimax on one GPU and Random on the other.

