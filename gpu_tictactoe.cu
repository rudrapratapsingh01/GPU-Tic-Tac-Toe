#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define EMPTY 0
#define GPU0 1
#define GPU1 2

__device__ int get_random_move(int board[9], int seed) {
    int valid_moves[9];
    int count = 0;
    for (int i = 0; i < 9; i++) {
        if (board[i] == EMPTY)
            valid_moves[count++] = i;
    }
    if (count == 0) return -1;
    int index = seed % count;
    return valid_moves[index];
}

__global__ void gpu0_random_strategy(int* board, int* move, unsigned int seed) {
    *move = get_random_move(board, seed);
}

__device__ bool is_winning_move(int board[9], int player, int move) {
    int test_board[9];
    for (int i = 0; i < 9; i++) test_board[i] = board[i];
    test_board[move] = player;

    int wins[8][3] = {
        {0,1,2},{3,4,5},{6,7,8}, // rows
        {0,3,6},{1,4,7},{2,5,8}, // cols
        {0,4,8},{2,4,6}          // diags
    };
    for (int i = 0; i < 8; i++) {
        int a = wins[i][0], b = wins[i][1], c = wins[i][2];
        if (test_board[a] == player && test_board[b] == player && test_board[c] == player)
            return true;
    }
    return false;
}

__global__ void gpu1_rule_based(int* board, int* move, unsigned int seed) {
    // Try to win
    for (int i = 0; i < 9; i++) {
        if (board[i] == EMPTY && is_winning_move(board, GPU1, i)) {
            *move = i;
            return;
        }
    }
    // Try to block opponent
    for (int i = 0; i < 9; i++) {
        if (board[i] == EMPTY && is_winning_move(board, GPU0, i)) {
            *move = i;
            return;
        }
    }
    // Else random
    *move = get_random_move(board, seed);
}

bool check_winner(int board[9], int player) {
    int wins[8][3] = {
        {0,1,2},{3,4,5},{6,7,8},
        {0,3,6},{1,4,7},{2,5,8},
        {0,4,8},{2,4,6}
    };
    for (int i = 0; i < 8; i++) {
        if (board[wins[i][0]] == player &&
            board[wins[i][1]] == player &&
            board[wins[i][2]] == player)
            return true;
    }
    return false;
}

void print_board(int board[9]) {
    const char* symbols = ".XO";
    for (int i = 0; i < 9; i++) {
        printf(" %c ", symbols[board[i]]);
        if (i % 3 == 2) printf("\n");
    }
    printf("\n");
}

int main() {
    int board[9] = {0};
    int* d_board0;
    int* d_board1;
    int* d_move0;
    int* d_move1;
    int move;

    cudaSetDevice(0);
    cudaMalloc(&d_board0, 9 * sizeof(int));
    cudaMalloc(&d_move0, sizeof(int));
    cudaSetDevice(1);
    cudaMalloc(&d_board1, 9 * sizeof(int));
    cudaMalloc(&d_move1, sizeof(int));

    srand(time(NULL));
    printf("Starting GPU vs GPU Tic Tac Toe\n\n");

    for (int turn = 0; turn < 9; turn++) {
        int current_gpu = turn % 2;
        int player = current_gpu == 0 ? GPU0 : GPU1;

        cudaMemcpy(current_gpu == 0 ? d_board0 : d_board1, board, 9 * sizeof(int), cudaMemcpyHostToDevice);
        if (current_gpu == 0) {
            cudaSetDevice(0);
            gpu0_random_strategy<<<1, 1>>>(d_board0, d_move0, rand());
            cudaMemcpy(&move, d_move0, sizeof(int), cudaMemcpyDeviceToHost);
        } else {
            cudaSetDevice(1);
            gpu1_rule_based<<<1, 1>>>(d_board1, d_move1, rand());
            cudaMemcpy(&move, d_move1, sizeof(int), cudaMemcpyDeviceToHost);
        }

        if (move < 0 || move >= 9 || board[move] != EMPTY) {
            printf("Invalid move by GPU%d! Skipping turn.\n", current_gpu);
            continue;
        }

        board[move] = player;
        printf("Turn %d - GPU%d played at position %d\n", turn + 1, current_gpu, move);
        print_board(board);

        if (check_winner(board, player)) {
            printf("GPU%d wins!\n", current_gpu);
            break;
        }
    }

    if (!check_winner(board, GPU0) && !check_winner(board, GPU1))
        printf("Game ends in a draw.\n");

    cudaFree(d_board0);
    cudaFree(d_move0);
    cudaFree(d_board1);
    cudaFree(d_move1);

    return 0;
}
