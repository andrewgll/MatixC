
#include "mx.h"

#define APPLY_TO_BOTH_INPLACE(matrix1, matrix2, function) __mx_apply_function_to_both(matrix1, matrix2, function, 0)
#define APPLY_TO_BOTH_COPY(matrix1, matrix2, function) __mx_apply_function_to_both(matrix1, matrix2, function, 1)

float sigmoidf(float value){
    return 1.0/(1+expf(-value));
}

void swap(float *a, float *b) {
    *a = *a + *b;
    *b = *a - *b;
    *a = *a - *b;
}

void mx_free(Matrix *matrix) {
    if (matrix)  {
        if(matrix->container){
            matrix->container->ref_count--;
            if (matrix->container->ref_count == 0) {
                if (matrix->container->data) {
                    free(matrix->container->data);
                    matrix->container->data = NULL;
                }
                free(matrix->container);
                matrix->container = NULL;
            }
        }
        free(matrix);  
    }
}

float __add_elements(float a, float b) {
    return a + b;
}

void mx_apply_sigmoid(Matrix* matrix){
    mx_apply_function(matrix, sigmoidf);
}

void mx_apply_function(Matrix* matrix, float (*func)(float)) {
    if(CHECK_MATRIX_VALIDITY(matrix) == -1){
        errno = EINVAL;
        perror("Got an ivalid matrix when tried to apply function.");
    }

    for(size_t i = 0; i < matrix->rows; ++i) {
        for(size_t j = 0; j < matrix->cols; ++j) {
            AT(matrix,i,j)= func(AT(matrix,i,j));
        }
    }
}

void* __mx_apply_function_to_both(Matrix* matrix1,Matrix* matrix2, float (*func)(float, float), uint8_t flags) {
    if(CHECK_MATRIX_VALIDITY(matrix1) == -1){
        return NULL;
    }
    if(matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols) {
        printf("Error: matrices have different dimensions.\n");
        return NULL;
    }
    Matrix* result = matrix1;
    if(CHECK_FLAG(flags,0) == 1){
        result = MATRIX(matrix1->rows, matrix1->cols);
    }
    for(size_t i = 0; i < matrix1->rows; ++i) {
        for(size_t j = 0; j < matrix1->cols; ++j) {
            AT(result, i, j) = func(AT(matrix1, i, j), AT(matrix2, i, j));
        }
    }
    return NULL;
}

__matrix_container* __init_container(float* array, size_t size) {
    if(size == 0){
        return NULL;
    }
    
    __matrix_container* container = MX_MALLOC(sizeof(__matrix_container));
    if (!container) {
        return NULL;
    }

    container->ref_count = 1;

    // Always allocate memory on the heap
    container->data = calloc(size, sizeof(*container->data));
    
    if (!container->data) {
        free(container);
        return NULL;
    }

    // If an external array is provided, copy the data over
    if (array) {
        memcpy(container->data, array, size * sizeof(*container->data));
    }

    return container;
}

Matrix* mx_copy(const Matrix* src){
    if(CHECK_MATRIX_VALIDITY(src)==-1){
        return NULL;
    }

    Matrix* copy = MATRIX(src->rows, src->cols);
    if(!copy){
        printf("Failed to create matrix.");
        return NULL;
    }
    copy->col_stride = src->col_stride;
    copy->row_stride = src->row_stride;
    memcpy(copy->container->data, src->container->data, sizeof(float) * src->rows*src->cols);
    return copy;
    
}
Matrix* __mx_init(float* array, size_t rows, size_t cols, float init_value) {

    if(!VALID_DIMENSIONS(rows, cols)){
        printf("Invalid matrix dimensions.");
        return NULL;
    }
    
    Matrix* mat = (Matrix*)MX_MALLOC(sizeof(Matrix));
    if (!mat) {
        return NULL;
    }

    // Pass array to init_container
    if(array){
        mat->container = __init_container(array, cols * rows);
    }
    else{
        mat->container = __init_container(NULL, cols * rows);
    }
    if (!mat->container) {
        free(mat);
        return NULL;
    }

    mat->rows = rows;
    mat->cols = cols;
    mat->row_stride = cols; 
    mat->col_stride = 1;

    // TODO How about lazy matrix view by default?
    mat->default_value = init_value;
    mat->flags = 0;

    // Initialize only if the value is non-zero and if an external array was not provided
    if(init_value != 0 && !array){
        for(size_t i = 0; i < mat->rows; ++i){
            for(size_t j = 0; j < mat->cols; ++j){
                AT(mat, i, j) = init_value;
            }
        }        
    }

    return mat;
}

void mx_set_to_rand(Matrix* m, float min, float max)
{
    if(CHECK_MATRIX_VALIDITY(m) == -1){
        return;
    }
    for(size_t i = 0; i < m->rows; ++i){
        for(size_t j = 0; j < m->cols; ++j){
            AT(m,i,j) = ((float)rand()/(float)RAND_MAX)*(max-min)+min;

        }
    }
}
Matrix* mx_view(const Matrix* matrix, size_t rows, size_t cols, float default_value){
    Matrix* view = (Matrix*)MX_MALLOC(sizeof(Matrix));
    if (!view) {
        printf("Failed to allocate memory for the matrix structure.");
        return NULL;
    }
    if(matrix){
        matrix->container->ref_count++;
        view->col_stride = matrix->col_stride;
        view->row_stride = matrix->row_stride;
        view->cols = rows;    // corrected to use passed rows and cols
        view->rows = cols;    // corrected to use passed rows and cols
        view->container = matrix->container;
        view->default_value = default_value;
        view->flags = 0;
    }
    else{
        view->col_stride = 1;
        view->row_stride = cols;
        view->cols = cols;
        view->rows = rows;
        view->container = NULL;
        view->default_value = default_value;
        view->flags = 0;
        SET_FLAG(view->flags, 0); // lazy matrix
    }
    return view;   
}
// TODO this can be replaced with macros
Matrix* mx_identity(size_t rows){
    return mx_diagonal(rows, 1);
}

Matrix* mx_diagonal(size_t rows, float value){
    if(!VALID_DIMENSIONS(rows, rows)){
        return NULL;
    }
    
    Matrix* m = MATRIX(rows, rows);

    for(size_t i = 0; i < rows; ++i){
        AT(m,i,i) = value;
    }
    return m;
}

Matrix* mx_cross_product(const Matrix* A, const Matrix* B) {
    // Ensure that both matrices are 3x1 vectors
    if (A->rows != 3 || A->cols != 1 || B->rows != 3 || B->cols != 1) {
        return NULL;  // Invalid vectors for cross product
    }
    
    Matrix* result = MATRIX(3, 1);  // Replace with your function to create a 3x1 matrix

    AT(result, 0, 0) = AT(A, 1, 0) * AT(B, 2, 0) - AT(A, 2, 0) * AT(B, 1, 0);
    AT(result, 1, 0) = AT(A, 2, 0) * AT(B, 0, 0) - AT(A, 0, 0) * AT(B, 2, 0);
    AT(result, 2, 0) = AT(A, 0, 0) * AT(B, 1, 0) - AT(A, 1, 0) * AT(B, 0, 0);

    return result;
}

float mx_cosine_between_two_vectors(Matrix* matrix1, Matrix* matrix2){
    if(matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols){
        errno = EINVAL;
        perror("Matrices must have the same dimensionality.");
        return -1;
    }

    Matrix* product;
    if(matrix1->rows == 1 && matrix2->rows == 1){
        Matrix* mx2_transposed = TRANSPOSE_VIEW(matrix2);
        product = DOT(matrix1, mx2_transposed);
        mx_free(mx2_transposed);
    }
    else if( matrix1->cols == 1 && matrix2->cols ==1){
        Matrix* mx1_transposed = TRANSPOSE_VIEW(matrix1);
        product = DOT(mx1_transposed, matrix2);
        mx_free(mx1_transposed);
    }
    else{
        errno = EINVAL;
        perror("Both matrices must be vectors.");
        return -1;
    }

    float length1 = mx_length(matrix1);
    float length2 = mx_length(matrix2);

    if(length1 == 0 || length2 == 0) {
        perror("One or both of the vectors have zero length.");
        mx_free(product);
        return -1;
    }

    float result = AT(product,0,0) / (length1 * length2);
    mx_free(product);
    return result;
}

Matrix* mx_perpendicular(const Matrix* matrix){
    // Check if matrix is valid
    if (CHECK_MATRIX_VALIDITY(matrix) == -1) {
        errno = EINVAL;
        perror("Error in 'mx_unit_vector_from'. Invalid matrix for this operation");
        return NULL;
    }
    // Check if matrix is in vector form
    if (matrix->cols != 1 && matrix->rows != 1) {
        errno = EINVAL;
        perror("Matrix should be in vector form in order to calculate unit vector");
        return NULL;    
    }
    Matrix* m_copy = MATRIX_COPY(matrix);
    if(m_copy->rows == 2 || m_copy->cols == 2){
        swap(&AT(m_copy,0,0), &AT(m_copy,0,1));
        AT(m_copy,0,0) *= -1;
        return m_copy;
    }
    if(m_copy->rows == 3 || m_copy->cols == 3){
        // Base vectors
        Matrix* x_base = MATRIX(1, 3);
        AT(x_base, 0, 0) = 1; AT(x_base, 0, 1) = 0; AT(x_base, 0, 2) = 0;

        Matrix* y_base = MATRIX(1, 3);
        AT(y_base, 0, 0) = 0; AT(y_base, 0, 1) = 1; AT(y_base, 0, 2) = 0;

        // If vec is not parallel to x_base, use x_base
        Matrix* base_to_use = x_base;
        if (fabs(AT(m_copy, 0, 0)) > 0.99) {  // Here 0.99 is an arbitrary threshold, adjust as needed
            base_to_use = y_base;  // Use y_base if vec is nearly parallel to x_base
        }

        Matrix* perpendicular = mx_cross_product(m_copy, base_to_use);
        mx_free(m_copy);
        mx_free(y_base);
        mx_free(x_base);
        return perpendicular;
    }
    errno = EINVAL;
    perror("ERROR: infinite many perpendiculars for more then 3 dimensional vector");
    return NULL;

}

Matrix* mx_unit_vector_from(const Matrix* matrix) {
    // Check if matrix is valid
    if (CHECK_MATRIX_VALIDITY(matrix) == -1) {
        errno = EINVAL;
        perror("Error in 'mx_unit_vector_from'. Invalid matrix for this operation");
        return NULL;
    }

    // Check if matrix is in vector form
    if (matrix->cols != 1 && matrix->rows != 1) {
        errno = EINVAL;
        perror("Matrix should be in vector form in order to calculate unit vector");
        return NULL;    
    }

    Matrix* result = MATRIX_COPY(matrix);
    float length = mx_length(matrix);

    // Check if length is close to zero
    if (fabs(length) < 1e-6) {
        errno = EINVAL;
        perror("Error in 'mx_unit_vector_from'. Vector length is too close to zero");
        mx_free(result);  // Free the allocated matrix before returning
        return NULL;
    }

    for (size_t i = 0; i < result->rows; ++i) {
        for (size_t j = 0; j < result->cols; ++j) {
            AT(result,i,j) /= length;
        }
    }

    return result;
}


uint8_t mx_equal(Matrix* matrix1, Matrix* matrix2){
    if(!VALID_MATRIX(matrix1) || !VALID_MATRIX(matrix2)) {
        perror("Invalid matrix dimensions.");
        return 0;
    }
    if(matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols){
        return 0;
    }
    for(size_t i = 0; i < matrix1->rows; ++i){
        for(size_t j =0; j<matrix1->cols; ++j){
            if(AT(matrix1,i,j) != AT(matrix2,i,j)){
                return 0;
            }
        }
    }
    return 1;
}

Matrix* mx_transpose(Matrix* matrix, uint8_t flags){
    Matrix* mx_transposed;
    if(CHECK_MATRIX_VALIDITY(matrix) == -1){
        return NULL;
    }

    if(CHECK_FLAG(flags,0) == 1){
        float rows = matrix->rows;
        matrix->rows = matrix->cols;
        matrix->cols = rows;
        
        float row_stride = matrix->row_stride;    
        matrix->row_stride = matrix->col_stride;      
        matrix->col_stride = row_stride;
        return matrix; 

    }
    else if(CHECK_FLAG(flags,2) == 1){
        mx_transposed = MATRIX_COPY(matrix);
        if(!mx_transposed){
            return NULL;
        }
    }
    else {
        mx_transposed = MATRIX_VIEW(matrix);
        if(!mx_transposed){
            return NULL;
        }
    }

    mx_transposed->rows = matrix->cols;
    mx_transposed->cols = matrix->rows;
    mx_transposed->row_stride = matrix->col_stride;          
    mx_transposed->col_stride = matrix->row_stride; 

    return mx_transposed;
}

Matrix* mx_arrange(size_t rows, size_t cols, float start_arrange) {
    Matrix* matrix = MATRIX(rows, cols);
    if (!matrix) {
        printf("Failed to allocate memory for the matrix.\n");
        return NULL;
    }

    for(size_t i=0; i< matrix->rows; ++i){
        for(size_t j=0; j< matrix->cols; ++j, start_arrange++){
            AT(matrix, i, j) = start_arrange;
        }
    }

    return matrix;
}

Matrix* mx_inverse(const Matrix* matrix){
    // pass
    Matrix* m = MATRIX_VIEW(matrix);
    return m;
}

Matrix* mx_scale(Matrix* matrix, float scalar) {
    Matrix* result = MATRIX_COPY(matrix);
    if (!result) {
        printf("Failed to allocate memory for the scaled matrix.\n");
        return NULL;
    }

    for(size_t i = 0; i < matrix->rows; ++i) {
        for(size_t j = 0; j < matrix->cols; ++j) {
            AT(result, i, j) = AT(result, i, j) * scalar;
        }
    }

    return result;
}

Matrix* mx_rand(size_t rows, size_t cols) {
    Matrix* matrix = MATRIX(rows, cols);
    if (!matrix) {
        printf("Failed to allocate memory for the matrix.\n");
        return NULL;
    }

    for(size_t i = 0; i < matrix->rows; ++i) {
        for(size_t j = 0; j < matrix->cols; ++j) {
            float value = (float)((double)rand() / RAND_MAX);
            AT(matrix, i, j) = value;
        }
    }

    return matrix;
}

Matrix* mx_add(Matrix* matrix1, Matrix* matrix2, uint8_t flags){ 
    if(CHECK_MATRIX_VALIDITY(matrix1) == -1|| CHECK_MATRIX_VALIDITY(matrix2)==-1){
        return NULL;
    }

    if (matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols) {
        printf("ERROR when 'mx_add': Sizes of two matrices should be equal.\n");
        return NULL;
    }
    Matrix* result = matrix1;
    if(CHECK_FLAG(flags,0) == 1){
        result = MATRIX(matrix1->rows, matrix1->cols);
    }
    
    if (!result) {
        printf("Failed to allocate memory for the resultant matrix.\n");
        return NULL;
    }

    APPLY_TO_BOTH_INPLACE(result, matrix2, __add_elements);

    return result;
}

float mx_length(const Matrix* matrix) {
    if (matrix == NULL) {
        errno = EINVAL;
        perror("ERROR when 'mx_length': Matrix is NULL.\n");
        return -1;
    }

    float value = 0;
    for(size_t i = 0; i < matrix->rows; ++i) {
        for(size_t j = 0; j < matrix->cols; ++j) {
            float element = AT(matrix, i, j);
            value += element * element;  // sum of squares of elements
        }
    }
    return sqrt(value);
}

Matrix* mx_subtract(const Matrix* matrix1, const Matrix* matrix2){
    if(CHECK_MATRIX_VALIDITY(matrix1)==-1|| CHECK_MATRIX_VALIDITY(matrix2)==-1){
        return NULL;
    }

    if (matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols) {
        printf("ERROR when 'mx_subtract': Sizes of two matrices should be equal.\n");
        return NULL;
    }

    Matrix* result = MATRIX(matrix1->rows, matrix1->cols);
    if (!result) {
        printf("Failed to allocate memory for the resultant matrix.\n");
        return NULL;
    }

    for (size_t i = 0; i < matrix1->rows; ++i) {
        for (size_t j = 0; j < matrix1->cols; ++j) {
            AT(result, i, j) = AT(matrix1, i, j) - AT(matrix2, i, j);
        }
    }

    return result;
}

Matrix* mx_dot(const Matrix* matrix1, const Matrix* matrix2, float scalar, uint8_t flags){

    Matrix* m1_copy;
    Matrix* m2_copy;
    if(CHECK_FLAG(flags,0)){
        if(CHECK_MATRIX_VALIDITY(matrix1) == -1 || CHECK_MATRIX_VALIDITY(matrix2) == -1){
            return NULL;
        }   
        m1_copy = MATRIX_COPY(matrix1);
        m2_copy = MATRIX_COPY(matrix2);
    }
    else if(CHECK_FLAG(flags,1)){
    // it's a vector-scalar multiplication
        if(CHECK_MATRIX_VALIDITY(matrix1) == -1){
            return NULL;
        }  
        m1_copy = MATRIX_COPY(matrix1);
        m2_copy = MATRIX_DIAGONAL(matrix1->cols, scalar);
        
    }
    else{
        errno = EINVAL;
        perror("ERROR when 'mx_dot': Unspecified flags.");
        return NULL;
    }
    if(m1_copy->cols != m2_copy->rows){
        if (m1_copy->cols == m2_copy->cols) {
            TRANSPOSE(m2_copy);
        }
        else if(m1_copy->rows == m2_copy->rows){
            TRANSPOSE(m1_copy);
        }
        else {
            errno = EINVAL;
            perror("ERROR when 'mx_dot': Matrices are not compatible for dot product.");
            mx_free((Matrix*) m1_copy);
            mx_free((Matrix*) m2_copy);
            return NULL;
        }
    }
    Matrix* result = MATRIX(m1_copy->rows, m2_copy->cols);
    if (!result) {
        mx_free((Matrix*) m1_copy);
        mx_free((Matrix*) m2_copy);
        errno = EINVAL;
        perror("ERROR when 'mx_dot': Unable to allocate memory for result matrix.");
        return NULL;
    }
    for(size_t i = 0; i < m1_copy->rows; ++i){
        for(size_t j = 0; j < m2_copy->cols; ++j){            
            float sum = 0;
            for(size_t k = 0; k < m1_copy->cols; k++){
                sum += AT(m1_copy, i, k) * AT(m2_copy, k, j);
            }
            AT(result, i, j) = sum;
        }
    }

    mx_free((Matrix*) m1_copy);
    mx_free((Matrix*) m2_copy);
    return result;
}

float mx_self_dot_product(Matrix* vector) {
    if(CHECK_MATRIX_VALIDITY(vector)==-1)
    {
        return -1;
    }

    // Ensure it's a vector
    if (vector->rows != 1 && vector->cols != 1) {
        errno = EINVAL;
        perror("ERROR when 'mx_self_dot_product': Input matrix is not a vector.");
        return -1; // or any other error value or behavior
    }

    float result = 0;
    size_t length = (vector->rows == 1) ? vector->cols : vector->rows;

    for (size_t i = 0; i < length; ++i) {
        float value = (vector->rows == 1) ? AT(vector, 0, i) : AT(vector, i, 0);
        result += value * value;
    }

    return result;
}

Matrix* mx_slice(const Matrix* src, size_t start_row, size_t end_row, size_t start_col, size_t end_col) {

    if(CHECK_MATRIX_VALIDITY(src)==-1){
        return NULL;
    }
    
    // Check for valid indices
    if (start_row > end_row || start_col > end_col || 
        end_row >= src->rows || end_col >= src->cols) {
        printf("ERROR when 'mx_slice': Invalid slice indices.\n");
        return NULL;  // Return NULL if the requested slice is invalid
    }

    size_t rows = end_row - start_row + 1;
    size_t cols = end_col - start_col + 1;

    Matrix* slice = MATRIX(rows,cols);
    if (!slice) {
        printf("ERROR when 'mx_slice': Unable to allocate memory for slice matrix.\n");
        return NULL;
    }

    // Directly copy rows from the source matrix to the slice matrix
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            AT(slice, i, j) = AT(src, start_row + i, start_col + j);
        }
    }

    return slice;
}

Matrix* open_dataset(const char* name){
    FILE* fp = fopen(name,"r");
    if(fp == NULL){
        printf("ERROR: Incorrect filename");
        return NULL;
    }
    char line[256];

    size_t data_set_rows = 0;
    size_t data_set_cols = 0;
    
    if(!fgets(line, sizeof(line), fp)){
        printf("ERROR: Something went wrong during file reading.");
        fclose(fp); // Close the file before returning
        return NULL;
    }
    data_set_rows++;
    
    char* token = strtok(line, ",");

    while (token) {
        token = strtok(NULL, ",");
        data_set_cols++;
    }
    while(fgets(line, sizeof(line), fp)) {data_set_rows++;};

    fclose(fp);

    fp = fopen(name,"r");
    if(fp == NULL){
        printf("ERROR: Couldn't reopen the file.");
        return NULL;
    }
    
    Matrix* result = MATRIX(data_set_rows, data_set_cols);

    size_t i = 0;
    while (fgets(line, sizeof(line), fp)) {
        size_t j = 0;
        char* token = strtok(line, ",");
        while (token) {
            float value = atof(token);
            if (i < data_set_rows && j < data_set_cols) {
                AT(result, i, j) = value;
            } else {
                printf("ERROR: Trying to access out-of-bounds index. i: %zu, j: %zu\n", i, j);
            }
            ++j;
            token = strtok(NULL, ",");
        }
        ++i;
    }
    
    fclose(fp);

    return result;
}

void* mx_print(const Matrix* matrix, const char* name) {
    if(CHECK_MATRIX_VALIDITY(matrix)==-1){
        return NULL;
    }
    printf("%s=([\n",name);
    for (size_t i = 0; i < matrix->rows; ++i) {
        printf("[");
        for (size_t j = 0; j < matrix->cols; ++j) {
            float value = AT(matrix,i,j);
            printf("%f", value);
            if (j < (size_t)(matrix->cols)-1) {
                printf(", ");
            }
        }
        printf("]");
        if (i < (size_t)(matrix->rows - 1)) {
            printf(",");
        }

        printf("\n");
    }
    printf("]);\n");
    return 0;
}
