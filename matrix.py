def matrix_multiply(a, b):
    # Check if the input matrices are valid
    if not isinstance(a, list) or len(a) == 0 or not isinstance(b, list) or len(b) == 0:
        raise ValueError("Invalid input matrices")

    # Get the dimensions of the matrices
    m = len(a)
    n = len(a[0])
    o = len(b[0])

    if n != len(b):
      raise ValueError("Can't do multipy")

    # Initialize the result matrix m*o, m row, o column
    c = [[0 for _ in range(o)] for _ in range(m)]

    # Perform the multiplication
    for i in range(m):
        for j in range(o):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]

    return c

def generate_identity_matrtix(n):
    c = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                c[i][j] = 1
    return c

#Gauss-Jordan elimination method
def inverse_matrix(matrix):
    n = len(matrix)
    # Create an identity matrix of the same size as the input matrix.
    identity = [[float(i == j) for i in range(n)] for j in range(n)]
    augmented_matrix = [row[:] for row in matrix]

    # Append the identity matrix to the right of the input matrix.
    for i in range(n):
        augmented_matrix[i] += identity[i]

    # Perform Gauss-Jordan elimination
    for i in range(n):
        # Make the diagonal contain all 1's
        div = augmented_matrix[i][i]
        if div == 0:
            return None  # This indicates that the matrix is not invertible
        for j in range(n * 2):
            augmented_matrix[i][j] /= div

        # Make all rows below this one 0 in the current column
        for j in range(n):
            if i != j:
                sub = augmented_matrix[j][i]
                for k in range(n * 2):
                    augmented_matrix[j][k] -= (sub * augmented_matrix[i][k])

    # Extract the right half of the matrix as the inverse
    for i in range(n):
        augmented_matrix[i] = augmented_matrix[i][n:]

    return augmented_matrix

# A3*2
a = [[1, 2], [3, 4], [5, 6]]
# B2*3
b = [[5, 6, 7], [7, 8, 9]]
# b = generate_identity_matrtix(2)
# 3*3
c = matrix_multiply(a, b)
print(c)

d = inverse_matrix(c)
if d is not None:
    print(d)
else:
    print("The matrix {} is not invertible.".format(c))