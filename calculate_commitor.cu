#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <iostream>

extern "C" {
  #include "mc_cpu.h"
}

#include "mc_gpu.h"
#include "gpu_tools.h"

void read_input_grid(FILE *ptr, char *bitgrid, int L, int *ising_grids, int grids_per_slice, int islice, int igrid);
double calc_commitor(int tot_nsweeps, int ngrids, int mag_output_int, int grid_output_int, int threadsPerBlock, int gpu_device, int gpu_method, float beta, float h, int *ising_grids, int *grid_fate_host);

int main() {

    // Define input variables
    int L = 64;
    int grids_per_slice = 1248;
    int Maxcon = 4;
    int tot_sweeps = 50000;
    int mag_output_int = 100;
    int grid_output_int = 100;
    int threadsPerBlock = 32;
    int gpu_device = 0;
    int gpu_method = 0;
    float beta = 0.54;
    float h = 0.07;

    // Standard deviation calculation parameters
    int m = 100;
    double standard_deviation = 0.0, standard_deviation_numerator = 0.0;
    int *ngrids_rand_ind = (int*)malloc(grids_per_slice*sizeof(int)); // Array that stores indices randomly sampled ngrid times from 0 to ngrids
    double *commitor_store = (double*)malloc(m*sizeof(double)); // Array to store commitors for standard deviation calcstandard_deviation_numeratorulation
    int *grid_fate_host = (int *)malloc(grids_per_slice*sizeof(int)); // Array to store fate of grids from grid propagation
    

    // Define bytes
    int bytes_per_slice = 12;

    // Define loop variables
    int im, irand, igrid, islice;
    int nA=0, nB=0;
    int i = 0, j = 0, progress = 0, tot_sample = 0;

    // Populate grid_fate_host array
    for(i=0;i<grids_per_slice;i++) {grid_fate_host[i] = 0;}

    // Set filenames
    const char *filename1 = "cluster_index_commitor.bin";
    const char *filename2 = "gridstates.bin";
    const char *filename3 = "clusters_to_commitor.bin";
    
    // open write cluster file
    FILE *ptr1 = fopen(filename1,"wb"); // open for write if not available for append 
    if (ptr1==NULL){
        fprintf(stderr,"Error opening %s for write!\n",filename1);
        exit(EXIT_FAILURE);
    }

    // open read grid file
    FILE *ptr2 = fopen(filename2, "rb");
    if (ptr2==NULL){
        fprintf(stderr, "Error opening %s for input!\n", filename2);
        exit(EXIT_FAILURE);
    }

    // open read index file
    FILE *ptr3 = fopen(filename3, "rb");
    if (ptr3==NULL){
        fprintf(stderr, "Error opening %s for input!\n", filename3);
        exit(EXIT_FAILURE);
    }

    // Ising grid configuration
    int *ising_grids = (int *)malloc(L*L*sizeof(int));
    if (ising_grids==NULL){
        fprintf(stderr,"Error allocating memory for Ising grids!\n");
        exit(EXIT_FAILURE);
    } 
    
    // Host copy of Ising grid configurations
    int *ising_grids_full = (int *)malloc(L*L*grids_per_slice*sizeof(int));
    if (ising_grids_full==NULL){
        fprintf(stderr,"Error allocating memory for host copy of Ising grids!\n");
        exit(EXIT_FAILURE);
    }

    // Create array to store index
    int *index = (int *)malloc(3*sizeof(int));
    if (index==NULL){
        fprintf(stderr,"Error allocating memory for index!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate space to read a single grid as bits
    int nbytes = L*L/8;
    if ( (L*L)%8 !=0 ) { nbytes++; }
    char *bitgrid = (char *)malloc(nbytes);
    if (bitgrid==NULL){
        fprintf(stderr,"Error allocating input buffer!");
        exit(EXIT_FAILURE);
    }


    // Create variable to store commitor
    double commitor = 0.0;

    while (1) {
        fread(index, sizeof(int), 3, ptr3);
        if ( feof(ptr3) ) { break;}
        tot_sample += 1;
    }

    fseek(ptr3, 0, SEEK_SET);

    while (1) {
        fread(index, sizeof(int), 3, ptr3);
        if ( feof(ptr3) ) { break;}
        islice = index[0]/100;
        igrid = index[1];
        // Loops over slices i.e. sweep snapshots
        read_input_grid(ptr2, bitgrid, L, ising_grids, grids_per_slice, islice, igrid);
        for (i = 0; i<grids_per_slice; i++) {
          for (j = 0; j<L*L; j++) {
            ising_grids_full[i*L*L+j] = ising_grids[j];
          }
        }
        //printf("Finished writing\n");
        //printf("Read grid\n");
        commitor = calc_commitor(tot_sweeps, grids_per_slice, mag_output_int, grid_output_int, threadsPerBlock, gpu_device, gpu_method,  beta, h, ising_grids_full, grid_fate_host);
        // Standard deviation of commitor, take d_ising_grid and generate a random list of samples of size 1248 from d_ising_grids, calculate commitor for random sample
        // Repeat above m and calculate standard deviation of m commitors
        for (im=0;im<m;im++) {
          nA=0; nB=0;
          for (igrid=0;igrid<grids_per_slice;igrid++){
            ngrids_rand_ind[igrid] = rand()%grids_per_slice;
          }
          for (igrid=0;igrid<grids_per_slice;igrid++){
            irand = ngrids_rand_ind[igrid];
            if ((int)grid_fate_host[irand]==0 ) {
              nA++;
            } else if ((int)grid_fate_host[irand]==1 ) {
              nB++;
            } // fate
          } //grids
          commitor_store[im] = (double)nB/(double)(nA+nB); // Store commitor
        } // m
        double standard_deviation_numerator = 0.0, mean_commitor = 0.0;            
        // Standard deviation calculation
        for (im=0;im<m;im++) {
          mean_commitor = mean_commitor + commitor_store[im];
        } 
        mean_commitor = mean_commitor/m;
        for (im=0;im<m;im++) {
          standard_deviation_numerator = standard_deviation_numerator + (commitor_store[im]-mean_commitor)*(commitor_store[im]-mean_commitor);
        }
        standard_deviation = sqrt(standard_deviation_numerator/m);
        //printf("Found commitor\n");
        fwrite(index, sizeof(int), 3, ptr1);
        //printf("Write 1\n");
        fwrite(&commitor, sizeof(commitor), 1, ptr1);
        fwrite(&standard_deviation, sizeof(standard_deviation), 1, ptr1);
        //printf("Write 2\n");
        //printf("%d %d %d %g\n", index[0], index[1], index[2], commitor[0]);
        progress += 1;
        printf("\rNumber of commitors written: %d/%d", progress, tot_sample); // Print progress
        fflush(stdout);
    }
    //printf("Number of clusters: %d\n", num_clust);
    printf("\n");
    // Free memory
    free(bitgrid); free(ising_grids); free(index); free(ising_grids_full); free(grid_fate_host); free(ngrids_rand_ind); free(commitor_store);


    // Close files
    fclose(ptr1); fclose(ptr2); fclose(ptr3);

    return EXIT_SUCCESS;
}

void read_input_grid(FILE *ptr, char *bitgrid, int L, int *ising_grids, int grids_per_slice, int islice, int igrid){
    
    // bytes per slice to move through gridfile
    int bytes_per_slice = 12+grids_per_slice*(L*L/8);

    // converts [0,1] to [-1,1]
    const int blookup[2] = {-1, 1};

    uint32_t one = 1U;

    int nbytes = L*L/8;
    // Read the grid
    fseek(ptr, 12+bytes_per_slice*islice+(L*L/8)*(igrid), SEEK_SET);
    fread(bitgrid, sizeof(char), nbytes, ptr);

    // Loop over grid points
    int ibit=0, ibyte=0;
    int isite=0;
    for (ibyte=0;ibyte<nbytes;ibyte++){
        for (ibit=0;ibit<8;ibit++){
            ising_grids[isite] = blookup[(bitgrid[ibyte] >> ibit) & one];
            isite++;
            if (isite>L*L) {break;}
        }
    }
}

double calc_commitor(int tot_nsweeps, int ngrids, int mag_output_int, int grid_output_int, int threadsPerBlock, int gpu_device, int gpu_method, float beta, float   h, int *ising_grids, int *grid_fate_host) {

/*=================================
   Constants and variables
  =================================*/ 

  double dn_threshold = -0.90;         // Magnetisation at which we consider the system to have reached spin up state
  double up_threshold =  0.90;         // Magnetisation at which we consider the system to have reached spin down state

  const bool run_gpu = true;      // Run using GPU
  const bool run_cpu = false;     // Run using CPU


  int itask = 1;
  int L = 64;
  int blocksPerGrid   = 1;             // Total number of threadBlocks

  //unsigned long rngseed = 2894203475;  // RNG seed (fixed for development/testing)
  unsigned long rngseed = (long)time(NULL);

/*=================================
   Initialise simulations
  =================================*/ 

  int i;
  int *grid_fate;  // stores pending(-1), reached B first (1) or reached A first (0)
  double pB;
  grid_fate = (int *)malloc(ngrids*sizeof(int));
  if (grid_fate==NULL) {
    printf("Error allocating memory for grid fates\n");
    exit(EXIT_FAILURE);
  }
  for (i=0;i<ngrids;i++) { grid_fate[i] = -1; } // all pending

  // Initialise host RNG
  init_genrand(rngseed);

  // Precompute acceptance probabilities for flip moves
  preComputeProbs_cpu(beta, h);

  int *d_ising_grids;                    // Pointer to device grid configurations
  curandState *d_state;                  // Pointer to device RNG states
  int *d_neighbour_list;                 // Pointer to device neighbour lists

  // How many sweeps to run in each call
  int sweeps_per_call;
  sweeps_per_call = mag_output_int < grid_output_int ? mag_output_int : grid_output_int;

  if (run_gpu==true) {
    
    gpuInit(gpu_device); // Initialise GPU device(s)

    // Allocate threads to thread blocks
    blocksPerGrid = ngrids/threadsPerBlock;
    if (ngrids%threadsPerBlock!=0) { blocksPerGrid += 1; }

    // Device copy of Ising grid configurations
    gpuErrchk( cudaMalloc(&d_ising_grids,L*L*ngrids*sizeof(int)) );
    
    // Populate from host copy
    gpuErrchk( cudaMemcpy(d_ising_grids,ising_grids,L*L*ngrids*sizeof(int),cudaMemcpyHostToDevice) );

    // Initialise GPU RNG
    gpuErrchk (cudaMalloc((void **)&d_state, ngrids*sizeof(curandState)) );
    unsigned long long gpuseed = (unsigned long long)rngseed;
    //std::cout << ngrids << " " << threadsPerBlock << "\n";
    init_gpurand<<<blocksPerGrid,threadsPerBlock>>>(gpuseed, ngrids, d_state);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //fprintf(stderr, "threadsPerBlock = %d, blocksPerGrid = %d\n",threadsPerBlock, blocksPerGrid);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_gpu(beta, h);

    // Neighbours
    gpuErrchk (cudaMalloc((void **)&d_neighbour_list, L*L*4*sizeof(int)) );
    preComputeNeighbours_gpu(L, d_ising_grids, d_neighbour_list);
  }
/*=================================
    Run simulations - CPU version
  =================================*/ 

  clock_t t1,t2;  // For measuring time taken
  int isweep;     // MC sweep loop counter
  int igrid;      // counter for loop over replicas



  if (run_cpu==true) {

    // Magnetisation of each grid
    double *magnetisation = (double *)malloc(ngrids*sizeof(double));
    if (magnetisation==NULL){
      fprintf(stderr,"Error allocating magnetisation array!\n");
      exit(EXIT_FAILURE);
    }

    t1 = clock();  // Start timer

    isweep = 0;
    while (isweep < tot_nsweeps){

      // Output grids to file
      //if (isweep%grid_output_int==0){
      //  write_ising_grids(L, ngrids, ising_grids, isweep);  
      //}

      // Report magnetisations
      if (isweep%mag_output_int==0){
        for (igrid=0;igrid<ngrids;igrid++){
          compute_magnetisation_cpu(L, ising_grids, igrid, magnetisation);
          //printf("Magnetisation of grid %d at sweep %d = %8.4f\n",igrid, isweep, magnetisation[igrid]);
        }
        if ( itask == 0 ) { // Report how many samples have nucleated.
          int nnuc = 0;
          for (igrid=0;igrid<ngrids;igrid++){
            if ( magnetisation[igrid] > up_threshold ) nnuc++;
          }
          //printf("%10d  %12.6f\n",isweep, (double)nnuc/(double)ngrids);
          if (nnuc==ngrids) break; // Stop if everyone has nucleated
        } else if ( itask == 1 ){

          // Statistics on fate of trajectories
          int nA=0, nB=0;
          for (igrid=0;igrid<ngrids;igrid++){
            //printf("grid_fate[%d] = %d\n",igrid, grid_fate[igrid]);
            if (grid_fate[igrid]==0 ) {
              nA++;
            } else if (grid_fate[igrid]==1 ) {
              nB++;
            } else {
              if ( magnetisation[igrid] > up_threshold ){
                grid_fate[igrid] = 1;
                nB++;
              } else if (magnetisation[igrid] < dn_threshold ){
                grid_fate[igrid] = 0;
                nA++;
              }
            } // fate
          } //grids

          // Monitor progress
          pB = (double)nB/(double)(nA+nB);

          //printf("\r Sweep : %10d, Reached m = %6.2f : %4d , Reached m = %6.2f : %4d , Unresolved : %4d, pB = %10.6f",
          // isweep, dn_threshold, nA, up_threshold, nB, ngrids-nA-nB,pB);
          //fflush(stdout);
          if (nA + nB == ngrids) break; // all fates resolved
        } // task
      } 

      // MC Sweep - CPU
      for (igrid=0;igrid<ngrids;igrid++) {
        mc_sweep_cpu(L, ising_grids, igrid, beta, h, sweeps_per_call);
      }
      isweep += sweeps_per_call;

    }

    t2 = clock();  // Stop Timer

    //printf("\n# Time taken on CPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);
    //printf("pB estimate : %10.6f\n",pB);

    // Release memory
    free(magnetisation);

  }

  /*=================================
    Run simulations - GPU version
  =================================*/ 
  if (run_gpu==true){

    // Host copy of magnetisation
    float *magnetisation = (float *)malloc(ngrids*sizeof(float));
    if (magnetisation==NULL){
      fprintf(stderr,"Error allocating magnetisation host array!\n");
      exit(EXIT_FAILURE);
    }

    // Device copy of magnetisation
    float *d_magnetisation;
    gpuErrchk( cudaMalloc(&d_magnetisation,ngrids*sizeof(float)) );

    // Streams
    cudaStream_t stream1;
    gpuErrchk( cudaStreamCreate(&stream1) );

    cudaStream_t stream2;
    gpuErrchk( cudaStreamCreate(&stream2) );


    t1 = clock();  // Start Timer

    isweep = 0;

    while(isweep < tot_nsweeps){

      // Output grids to file
      if (isweep%grid_output_int==0){
        // Asynchronous - can happen while magnetisation is being computed in stream 2
        gpuErrchk( cudaMemcpyAsync(ising_grids,d_ising_grids,L*L*ngrids*sizeof(int),cudaMemcpyDeviceToHost,stream1) );
      }

      // Can compute manetisation while grids are copying
      if (isweep%mag_output_int==0){
        compute_magnetisation_gpu<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(L, ngrids, d_ising_grids, d_magnetisation);    
        gpuErrchk( cudaMemcpyAsync(magnetisation,d_magnetisation,ngrids*sizeof(float),cudaMemcpyDeviceToHost, stream2) );
      } 

      // MC Sweep - GPU
      gpuErrchk( cudaStreamSynchronize(stream1) ); // Make sure copy completed before making changes
      if (gpu_method==0){
        mc_sweep_gpu<<<blocksPerGrid,threadsPerBlock,0,stream1>>>(L,d_state,ngrids,d_ising_grids,d_neighbour_list, (float)beta,(float)h,sweeps_per_call);
      } else if (gpu_method==1){
          size_t shmem_size = L*L*threadsPerBlock*sizeof(uint8_t)/8; // number of bytes needed to store grid as bits
          mc_sweep_gpu_bitrep<<<blocksPerGrid,threadsPerBlock,shmem_size,stream1>>>(L,d_state,ngrids,d_ising_grids, d_neighbour_list, (float)beta,(float)h,sweeps_per_call);
      } else if (gpu_method==2){
          size_t shmem_size = L*L*threadsPerBlock*sizeof(uint8_t)/8; // number of bytes needed to store grid as bits
          if (threadsPerBlock==32){
            mc_sweep_gpu_bitmap32<<<blocksPerGrid,threadsPerBlock,shmem_size,stream1>>>(L,d_state,ngrids,d_ising_grids, d_neighbour_list, (float)beta,(float)h,sweeps_per_call);
          } else if (threadsPerBlock==64){
            mc_sweep_gpu_bitmap64<<<blocksPerGrid,threadsPerBlock,shmem_size,stream1>>>(L,d_state,ngrids,d_ising_grids, d_neighbour_list, (float)beta,(float)h,sweeps_per_call);
          } else {
            printf("Invalid threadsPerBlock for gpu_method=2\n");
            exit(EXIT_FAILURE);
          } 
      } else {
        printf("Unknown gpu_method in ising.cu\n");
        exit(EXIT_FAILURE);
      }
      
      // Writing of the grids can be happening on the host while the device runs the mc_sweep kernel
      //if (isweep%grid_output_int==0){
      //  write_ising_grids(L, ngrids, ising_grids, isweep);  
      //}

      // Write and report magnetisation - can also be happening while the device runs the mc_sweep kernel
      if (isweep%mag_output_int==0){
        gpuErrchk( cudaStreamSynchronize(stream2) );  // Wait for copy to complete
        //for (igrid=0;igrid<ngrids;igrid++){
        //  printf("    %4d     %10d      %8.6f\n",igrid, isweep, magnetisation[igrid]);
        //}
        if ( itask == 0 ) { // Report how many samples have nucleated.
          int nnuc = 0;
          for (igrid=0;igrid<ngrids;igrid++){
            if ( magnetisation[igrid] > up_threshold ) nnuc++;
          }
          //printf("%10d  %12.6f\n",isweep, (double)nnuc/(double)ngrids);
          if (nnuc==ngrids) break; // Stop if everyone has nucleated
        } else if ( itask == 1 ){

            // Statistics on fate of trajectories
            int nA=0, nB=0;
            for (igrid=0;igrid<ngrids;igrid++){
              //printf("grid_fate[%d] = %d\n",igrid, grid_fate[igrid]);
              //printf("magnetisation[%d] = %d\n",igrid, magnetisation[igrid]);
              if (grid_fate[igrid]==0 ) {
                nA++;
              } else if (grid_fate[igrid]==1 ) {
                nB++;
              } else {
                if ( magnetisation[igrid] > up_threshold ){
                  grid_fate[igrid] = 1;
                  nB++;
                } else if (magnetisation[igrid] < dn_threshold ){
                  grid_fate[igrid] = 0;
                  nA++;
                }
              } // fate
            } //grids

            // Monitor progress
            pB = (double)nB/(double)(nA+nB);
            //printf("\r Sweep : %10d, Reached m = %6.2f : %4d , Reached m = %6.2f : %4d , Unresolved : %4d, pB = %10.6f",
            //isweep, dn_threshold, nA, up_threshold, nB, ngrids-nA-nB,pB);
            //fflush(stdout);
            //printf("%d %d\n", nA, nB);
            if (nA + nB == ngrids) break; // all fates resolved
        } // task 
      }

      // Increment isweep
      isweep += sweeps_per_call;

      // Make sure all kernels updating the grids are finished before starting magnetisation calc
      gpuErrchk( cudaStreamSynchronize(stream1) );
      gpuErrchk( cudaPeekAtLastError() );
    }

    // Ensure all threads finished before stopping timer
    gpuErrchk( cudaDeviceSynchronize() )

    t2 = clock();

    //printf("\n# Time taken on GPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);
    //printf("pB estimate : %10.6f\n",pB);


    // Destroy streams
    gpuErrchk( cudaStreamDestroy(stream1) );
    gpuErrchk( cudaStreamDestroy(stream2) );


    // Free magnetisation arrays
    free(magnetisation);
    gpuErrchk( cudaFree(d_magnetisation) );
  }

  for(i=0;i<ngrids;i++) {grid_fate_host[i] = grid_fate[i];}

/*=================================================
    Tidy up memory used in both GPU and CPU paths
  =================================================*/ 
  if (run_gpu==true) {
    gpuErrchk( cudaFree(d_ising_grids) );
    gpuErrchk( cudaFree(d_state) );
    gpuErrchk( cudaFree(d_neighbour_list) );
  }

  return pB;

}
