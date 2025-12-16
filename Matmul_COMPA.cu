#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <omp.h>
#include <chrono>
#include <random>
#include <mma.h>
#include <cuda_fp16.h>

#define BSIZE2D 16
#define TILE_SIZE 16

using namespace std;
namespace wmma = nvcuda::wmma;

// Generador aleatorio global
std::mt19937 gen(std::random_device{}());
std::uniform_real_distribution<> dis(0.0, 1.0);

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void kernel_matmul(int n, float *A, float *B, float *C){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(tx < n && ty < n){
        float sum = 0.0f;
        for(int k=0; k<n; k++){
            sum += A[ty*n + k] * B[k*n + tx];
        }
        C[ty*n + tx] = sum;
    }
}

// Kernel con memoria compartida
__global__ void kernel_matmul_gpusm(int n, float *A, float *B, float *C){
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calcular la fila y columna de C para este thread
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Iterar sobre los tiles de A y B
    for(int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++){
        // Cargar tile de A en memoria compartida
        if(row < n && (t * TILE_SIZE + tx) < n){
            As[ty][tx] = A[row * n + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Cargar tile de B en memoria compartida
        if((t * TILE_SIZE + ty) < n && col < n){
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Multiplicar los dos tiles
        for(int k = 0; k < TILE_SIZE; k++){
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Escribir el resultado en C
    if(row < n && col < n){
        C[row * n + col] = sum;
    }
}

__global__ void wmma_matmul_fp16(half *A, half *B, half *C, int n){
    // Implementacion del kernel WMMA para multiplicacion de matrices en FP16
    // Calcular la posición del warp (cada warp procesa un tile de 16x16)
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Calcular la fila y columna base para este warp
    int row = warpM * WMMA_M;
    int col = warpN * WMMA_N;
    
    // Verificar límites
    if(row >= n || col >= n) return;

    // Declarar fragmentos WMMA
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

    // forzar a que la suma interna se haga en 16 bits
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // Inicializar el fragmento acumulador a cero
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    // bucle principal: iterar sobre la dimensión K
    for (int k = 0; k < n; k += WMMA_K){
        // Cargar los fragmentos de A y B desde memoria global
        wmma::load_matrix_sync(a_frag, A + row * n + k, n);
        wmma::load_matrix_sync(b_frag, B + k * n + col, n);

        // Realizar la multiplicacion de matrices en los Tensor Cores
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Almacenar el resultado en C
    wmma::store_matrix_sync(C + row * n + col, c_frag, n, wmma::mem_row_major);
}


void matmul_block(float *A, float *B, float *C, int n){
    int blockSize = 64; // Tamaño de bloque fijo para este ejemplo
    #pragma omp parallel for collapse(2)
        for(int ii = 0; ii < n; ii += blockSize){
            for(int jj = 0; jj < n; jj += blockSize){
                for(int kk = 0; kk < n; kk += blockSize){

                    // multiplicar bloque A[ii][kk] con bloque B[kk][jj]
                    for(int i = ii; i < std::min(ii + blockSize, n); i++){
                        for(int j = jj; j < jj + blockSize; j++){
                            float sum = 0.0f;
                            for(int k = kk; k < kk + blockSize; k++){
                                sum += A[i*n + k] * B[k*n + j];
                            }
                            C[i*n + j] += sum;
                        }
                    }
                }
            }
        }  
}

void matmul_gpu(float *A, float *B, float *C, int n){
    // Implementacion de la version GPU basica
    float *d_A, *d_B, *d_C;
    float milliseconds = 0;
    
    cudaMalloc(&d_A, sizeof(float)*n*n);
    cudaMalloc(&d_B, sizeof(float)*n*n);
    cudaMalloc(&d_C, sizeof(float)*n*n);
    cudaMemcpy(d_A, A, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeof(float)*n*n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 block(BSIZE2D, BSIZE2D, 1);
    dim3 grid((n+BSIZE2D-1)/BSIZE2D, (n+BSIZE2D-1)/BSIZE2D, 1);
    cudaEventRecord(start);
    // Llamada al kernel
    kernel_matmul<<<grid, block>>>(n, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Tiempo de ejecucion GPU (ms): " << milliseconds << endl;
    
    cudaMemcpy(C, d_C, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matmul_gpusm(float *A, float *B, float *C, int n){
    // Implementacion de la version GPU con memoria compartida (GPUsm)
    float *d_A, *d_B, *d_C;
    float milliseconds = 0;
    
    cudaMalloc(&d_A, sizeof(float)*n*n);
    cudaMalloc(&d_B, sizeof(float)*n*n);
    cudaMalloc(&d_C, sizeof(float)*n*n);
    cudaMemcpy(d_A, A, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeof(float)*n*n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Usar tiles de TILE_SIZE x TILE_SIZE
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE, 1);
    
    cudaEventRecord(start);
    // Llamada al kernel con memoria compartida
    kernel_matmul_gpusm<<<grid, block>>>(n, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Tiempo de ejecucion GPU con memoria compartida (ms): " << milliseconds << endl;
    
    cudaMemcpy(C, d_C, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matmul_gpwmm(float *A, float *B, float *C, int n){
    // Implementacion de la version GPU con WMMA (Tensor Cores)
    size_t size_bytes = n*n*sizeof(half);
    half *h_A = new half[n*n];
    half *h_B = new half[n*n];
    half *h_C = new half[n*n];

    // Convertir las matrices float a half precision
    for(int i = 0; i < n*n; i++){
        h_A[i] = __float2half(A[i]);
        h_B[i] = __float2half(B[i]);
        h_C[i] = __float2half(0.0f);
    }
    
    half *d_A, *d_B, *d_C;
    float milliseconds = 0;
    cudaMalloc(&d_A, size_bytes);
    cudaMalloc(&d_B, size_bytes);
    cudaMalloc(&d_C, size_bytes);

    cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Configuración de grid y bloques
    // Cada warp (32 threads) procesa un tile WMMA de 16x16
    dim3 blockDim(32, 4, 1);  // 128 threads por bloque (4 warps)
    dim3 gridDim((n + WMMA_M - 1) / WMMA_M, (n + WMMA_N - 1) / (WMMA_N * 4), 1);

    cudaEventRecord(start);
    wmma_matmul_fp16<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Tiempo de ejecucion GPU con WMMA (tensor cores) (ms): " << milliseconds << endl;
    
    // Copiar resultado de vuelta y convertir a float
    cudaMemcpy(h_C, d_C, size_bytes, cudaMemcpyDeviceToHost);
    for(int i = 0; i < n*n; i++){
        C[i] = __half2float(h_C[i]);
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cout << "Forma de uso: ./prog <n> <nt> <ALG>" << endl;
        cout << "ALG: algoritmo (1=CPU multicore, 2=GPU basica, 3=GPUsm, 4=WMMA)" << endl;
        return -1;
    }

    int n = atoi(argv[1]);
    int nt = atoi(argv[2]);
    int ALG = atoi(argv[3]);

    while(n % 16 != 0) {
        cout << "n debe ser multiplo de 16 para WMMA (tensor cores)." << endl;
        cout << "Ingrese un nuevo valor para n: ";
        cin >> n;
        cout << endl;
    }

    float *A, *B, *C;
    A = new float[n * n];
    B = new float[n * n];
    C = new float[n * n];

    // Valores aleatorios en A y B
    /*for (int i=0; i<n*n; i++){
        A[i] = dis(gen);
        B[i] = dis(gen);
        C[i] = 0.0;
    }*/
   for (int i=0; i<n*n; i++){
        A[i] = i;
        B[i] = i;
        C[i] = 0.0;
    }

    switch(ALG){
        case 1:{
            // Version CPU multicore
            omp_set_num_threads(nt);
            auto start = std::chrono::high_resolution_clock::now();
            matmul_block(A, B, C, n);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            cout << "Tiempo de ejecucion CPU (ms): " << elapsed.count() << endl;
            break;
        }
        case 2:{
            // Version GPU basica
            matmul_gpu(A, B, C, n);
            break;
        }
        case 3: {
            // Version GPU con memoria compartida (GPUsm)
            matmul_gpusm(A, B, C, n);
            break;
        }
        case 4: {
            // Version GPU con WMMA (Tensor Cores)
            matmul_gpwmm(A, B, C, n);
            break;
        }
        default:
            cout << "Algoritmo no valido. Use 1, 2, 3 o 4." << endl;
            break;
    }
    
    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
}