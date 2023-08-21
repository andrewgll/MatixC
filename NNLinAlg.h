#ifndef NNLinAlg
#define NNLinAlg
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define AT(m, i, j) ((m)->data[(i) * (m)->row_stride + (j) * (m)->col_stride])
#define MATRIX(rows, cols) create_matrix(rows, cols)

double sigmoidf(double value){
    return 1/(1+exp(-value));
}

typedef struct{
    size_t size;
    size_t rows;
    size_t cols;
    size_t row_stride;
    size_t col_stride;
    double *data;
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
    mat->rows = rows;
    mat->cols = cols;
    mat->row_stride = cols; 
    mat->col_stride = 1;    

    return mat;
}

Matrix* transpose(const Matrix* matrix){
    Matrix* transposed = (Matrix*)malloc(sizeof(Matrix));
    if (!transposed) {
        perror("Failed to allocate memory for transposed view.");
        return NULL;
    }

    transposed->data = matrix->data;
    transposed->rows = matrix->cols;
    transposed->cols = matrix->rows;
    transposed->row_stride = matrix->col_stride;          
    transposed->col_stride = matrix->row_stride; 

    return transposed;
}

Matrix* rand_matrix(size_t rows, size_t cols){
    Matrix* matrix = MATRIX(rows, cols);
    for(size_t i=0; i< matrix->rows;i++){
        for(size_t j=0; j< matrix->cols;j++){
            double value = (double)rand()/RAND_MAX*100;
            AT(matrix,i,j) = value;
        }
    }
    return matrix;
}

Matrix* dot(const Matrix* matrix1, const Matrix* matrix2){
    if(matrix1->cols != matrix2->rows){
        perror("Incorrect matrix shape");
        return NULL;
    }
    Matrix* result = MATRIX(matrix1->rows, matrix2->cols);
    for(size_t i=0; i<matrix1->rows; i++){
        for(size_t j=0; j<matrix2->cols; j++){            
            double sum = 0;
            for (size_t k = 0; k < matrix1->cols; k++) {
                sum += AT(matrix1, i, k) * AT(matrix2, k, j);
            }
            AT(result, i, j) = sum;
        }
    }
    return result;
}

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

void print_matrix(const Matrix* matrix) {
    printf("array([\n");
    for (size_t i = 0; i < matrix->rows; i++) {
        printf("[");
        for (size_t j = 0; j < matrix->cols; j++) {
            double value = AT(matrix,i,j);
            printf("%f", value);
            if (j < matrix->cols - 1) {
                printf(", ");
            }
        }
        printf("]");
        if (i < matrix->rows - 1) {
            printf(",");
        }

        printf("\n");
    }
    printf("])\n");
}

#endif // NNLinAlg