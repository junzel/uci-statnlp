import numpy as np
import pdb


def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    # Initialize R and bp
    R = np.full((N, L), -np.inf)
    bp = np.zeros((N, L)).astype(int)
    # Initialize output sequence
    y = [0] * N

    # Added scores starting from <s>. 
    for i in range(L):
        R[0, i] = start_scores[i] + emission_scores[0][i]

    # Iterative Computation (forward)
    for i in range(1, N):
        for j in range(L):
            for j_pre in range(L): # Loop through preceding labels
                score = R[i-1][j_pre] + trans_scores[j_pre][j] + emission_scores[i,j]
                if score > R[i,j]:
                    R[i,j] = score
                    bp[i,j] = j_pre

    # Add scores of <e>
    for j in range(L):
        R[N-1,j] += end_scores[j]

    # Back propogation
    # Get the last label of sequence
    y = [np.argmax(R[-1])]
    # Loop through all positions backwards
    for i in reversed(range(1, N)):
        y.append(bp[i, y[-1]])

    # Output the largest final score and the sequence (y is backwards, needs to be reversed)
    return (np.max(R[-1]), list(reversed(y)))

    # y = []
    # for i in range(N):
    #     # stupid sequence
    #     y.append(i % L)
    # # score set to 0
    # return (0.0, y)
