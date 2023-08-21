#ifndef NNC_H
#define NNC_H
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


#define AT(m, i, j) ((m)->data[(i) * (m)->row_stride + (j) * (m)->col_stride])
#define MATRIX(rows, cols) create_matrix(rows, cols)
#define MATRIX_ROWS(m) ((m)->size/(m)->row_stride)
#define MATRIX_COLS(m) ((m)->row_stride)

double sigmoidf(double value){
    return 1/(1+exp(-value));
}

typedef struct{
    double *data;
    size_t row_stride;
    size_t col_stride;
    size_t size;
} Matrix;

Matrix* create_matrix(size_t rows, size_t cols) {

    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) {
        perror("Failed to allocate memory for the matrix structure.");
        return NULL;
    }

    mat->data = (double*)malloc(rows * cols * sizeof(double));
    if (!mat->data) {
        perror("Failed to allocate memory for the matrix data.");
        free(mat); 
        return NULL;
    }
    mat->size = rows*cols;
    mat->row_stride = cols; 
    mat->col_stride = 1;    

    return mat;
}

void print_matrix(const Matrix* matrix) {
    printf("array([\n");
    for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
        printf("[");
        for (size_t j = 0; j < MATRIX_COLS(matrix); j++) {
            double value = AT(matrix,i,j);
            printf("%f", value);
            if (j < MATRIX_COLS(matrix) - 1) {
                printf(", ");
            }
        }
        printf("]");
        if (i < MATRIX_ROWS(matrix) - 1) {
            printf(",");
        }

        printf("\n");
    }
    printf("])\n");
}

// Matrix* transpose(Matrix* matrix){
//     size_t rows = matrix->row_stride;
//     matrix->row_stride = ;           
//     matrix->col_stride = rows; 
//     return matrix;
// }

void init_matrix_with_random_values(const Matrix* matrix){
    size_t rows = matrix->size/matrix->row_stride;
    for(size_t i=0; i< rows;i++){
        for(size_t j=0; j< matrix->row_stride;j++){
            int value = (int)rand()%10;
            AT(matrix,i,j) = value;
        }
    }
}

Matrix* dot(const Matrix* matrix1, const Matrix* matrix2){
    if(MATRIX_COLS(matrix1) != MATRIX_ROWS(matrix2)){
        perror("Incorrect matrix shape");
        return NULL;
    }
    Matrix* result = MATRIX(MATRIX_ROWS(matrix1), MATRIX_COLS(matrix2));
    for(size_t i=0; i<MATRIX_ROWS(matrix1); i++){
        for(size_t j=0; j<MATRIX_COLS(matrix2); j++){            
            double sum = 0;
            for (size_t k = 0; k < MATRIX_COLS(matrix1); k++) {
                sum += AT(matrix1, i, k) * AT(matrix2, k, j);
            }
            AT(result, i, j) = sum;
        }
    }
    return result;
}

// Matrix* slice(const Matrix* matrix, size_t i_rows, size_t j_rows, size_t i_cols, size_t j_cols){
//     // Check if indices are valid
//     if (i_rows >= j_rows || i_cols >= j_cols || 
//         j_rows > matrix->rows || j_cols > matrix->cols) {
//         perror("Invalid slice indices");
//         return NULL;
//     }

//     Matrix* result = (Matrix*)malloc(sizeof(Matrix));
//     if (!result) {
//         perror("Memory allocation failed");
//         return NULL;
//     }

//     result->rows = j_rows - i_rows;
//     result->cols = j_cols - i_cols;

//     // Set the starting point to the beginning of the slice
//     result->data = &AT(matrix, i_rows, i_cols);

//     // Strides are inherited, as you're just viewing the original data in a different way
//     result->row_stride = matrix->row_stride;
//     result->col_stride = matrix->col_stride;

//     return result;
// }


Matrix* open_dataset(const char* name){
    FILE* fp = fopen(name,"r");
    if(fp == NULL){
        return NULL;
    }
    char line[256];

    size_t data_set_rows = 0;
    size_t data_set_cols = 0;
    
    fgets(line, sizeof(line), fp);
    data_set_rows++;
    
    char* token = strtok(line, ",");

    while (token) {
        token = strtok(NULL, ",");
        data_set_cols++;
    }
    while(fgets(line, sizeof(line), fp)) {data_set_rows++;};

    fp = fopen(name,"r");
    Matrix* result = MATRIX(data_set_rows, data_set_cols);

    size_t i = 0;
    while (fgets(line, sizeof(line), fp)) {
        size_t j = 0;
        char* token = strtok(line, ",");
        while (token) {
            double value = atof(token);
            AT(result, i, j) = value;
            j++;
            token = strtok(NULL, ",");
        }
        i++;
    }
    return result;
}

double inference(double** dataset, double* vector, size_t dataset_size_1,size_t dataset_size_2, size_t vector_size){
    double scalar=0;
    for(size_t i = 0; i<dataset_size_1; i++){
    }
    return scalar;
}
#endif // NNC_H