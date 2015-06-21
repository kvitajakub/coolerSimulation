/**
 * @file        proj02.cpp
 * @author      Jiri Jaros and Radek Hrbacek\n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       Parallelisation of Heat Distribution Method in Heterogenous
 *              Media using OpenMP
 *
 * @version     2015
 * @date        10 April 2015, 10:22 (created) \n
 * @date        10 April 2015, 10:22 (last revised) \n
 *
 * @detail
 * This is the main file of the project. Add all code here.
 */


#include <mpi.h>

#include <string.h>
#include <string>
#include <cmath>

#include <hdf5.h>

#include <sstream>
#include <immintrin.h>

#include "MaterialProperties.h"
#include "BasicRoutines.h"

using namespace std;


//----------------------------------------------------------------------------//
//---------------------------- Global variables ------------------------------//
//----------------------------------------------------------------------------//

/// Temperature data for sequential version.
float *seqResult = NULL;
/// Temperature data for parallel method.
float *parResult = NULL;

/// Parameters of the simulation
TParameters parameters;

/// Material properties
TMaterialProperties materialProperties;


//----------------------------------------------------------------------------//
//------------------------- Function declarations ----------------------------//
//----------------------------------------------------------------------------//

/// Sequential implementation of the Heat distribution
void SequentialHeatDistribution(float                     *seqResult,
                                const TMaterialProperties &materialProperties,
                                const TParameters         &parameters,
                                string                     outputFileName);

/// Parallel Implementation of the Heat distribution (Non-overlapped file output)
void ParallelHeatDistribution(float                     *parResult,
                              const TMaterialProperties &materialProperties,
                              const TParameters         &parameters,
                              string                     outputFileName);

/// Store time step into output file
void StoreDataIntoFile(hid_t         h5fileId,
                       const float * data,
                       const size_t  edgeSize,
                       const size_t  snapshotId,
                       const size_t  iteration);


//----------------------------------------------------------------------------//
//------------------------- Function implementation  -------------------------//
//----------------------------------------------------------------------------//


void ComputePoint(float  *oldTemp,
                  float  *newTemp,
                  float  *params,
                  int    *map,
                  size_t  i,
                  size_t  j,
                  size_t  edgeSize,
                  float   airFlowRate,
                  float   coolerTemp)
{
  // [i] Calculate neighbor indices
  const int center = i * edgeSize + j;
  const int top    = center - edgeSize;
  const int bottom = center + edgeSize;
  const int left   = center - 1;
  const int right  = center + 1;

  // [ii] The reciprocal value of the sum of domain parameters for normalization
  const float frac = 1.0f / (params[top]    +
                             params[bottom] +
                             params[left]   +
                             params[center] +
                             params[right]);

  // [iii] Calculate new temperature in the grid point
  float pointTemp = 
        oldTemp[top]    * params[top]    * frac +
        oldTemp[bottom] * params[bottom] * frac +
        oldTemp[left]   * params[left]   * frac +
        oldTemp[right]  * params[right]  * frac +
        oldTemp[center] * params[center] * frac;

  // [iv] Remove some of the heat due to air flow (5% of the new air)
  pointTemp = (map[center] == 0)
              ? (airFlowRate * coolerTemp) + ((1.0f - airFlowRate) * pointTemp)
              : pointTemp;

  newTemp[center] = pointTemp;
}

/**
 * Sequential version of the Heat distribution in heterogenous 2D medium
 * @param [out] seqResult          - Final heat distribution
 * @param [in]  materialProperties - Material properties
 * @param [in]  parameters         - parameters of the simulation
 * @param [in]  outputFileName     - Output file name (if NULL string, do not store)
 *
 */
void SequentialHeatDistribution(float                      *seqResult,
                                const TMaterialProperties &materialProperties,
                                const TParameters         &parameters,
                                string                     outputFileName)
{
  // [1] Create a new output hdf5 file
  hid_t file_id = H5I_INVALID_HID;
  
  if (outputFileName != "")
  {
    if (outputFileName.find(".h5") == string::npos)
      outputFileName.append("_seq.h5");
    else
      outputFileName.insert(outputFileName.find_last_of("."), "_seq");
    
    file_id = H5Fcreate(outputFileName.c_str(),
                        H5F_ACC_TRUNC,
                        H5P_DEFAULT,
                        H5P_DEFAULT);
    if (file_id < 0) ios::failure("Cannot create output file");
  }


  // [2] A temporary array is needed to prevent mixing of data form step t and t+1
  float *tempArray = (float *)_mm_malloc(materialProperties.nGridPoints * 
                                         sizeof(float), DATA_ALIGNMENT);

  // [3] Init arrays
  for (size_t i = 0; i < materialProperties.nGridPoints; i++)
  {
    tempArray[i] = materialProperties.initTemp[i];
    seqResult[i] = materialProperties.initTemp[i];
  }

  // [4] t+1 values, t values
  float *newTemp = tempArray;
  float *oldTemp = seqResult;

  if (!parameters.batchMode) 
    printf("Starting sequential simulation... \n");

  //---------------------- [5] start the stop watch ------------------------------//
  double elapsedTime = MPI_Wtime();
  size_t i, j;
  size_t iteration;
  float middleColAvgTemp = 0.0f;

  // [6] Start the iterative simulation
  for (iteration = 0; iteration < parameters.nIterations; iteration++)
  {
    // [a] calculate one iteration of the heat distribution (skip the grid points at the edges)
    for (i = 1; i < materialProperties.edgeSize - 1; i++)
      for (j = 1; j < materialProperties.edgeSize - 1; j++)
        ComputePoint(oldTemp,
                     newTemp,
                     materialProperties.domainParams,
                     materialProperties.domainMap,
                     i, j,
                     materialProperties.edgeSize, 
                     parameters.airFlowRate,
                     materialProperties.coolerTemp);

    // [b] Compute the average temperature in the middle column
    middleColAvgTemp = 0.0f;
    for (i = 0; i < materialProperties.edgeSize; i++)
      middleColAvgTemp += newTemp[i*materialProperties.edgeSize +
                          materialProperties.edgeSize/2];
    middleColAvgTemp /= materialProperties.edgeSize;

    // [c] Store time step in the output file if necessary
    if ((file_id != H5I_INVALID_HID)  && ((iteration % parameters.diskWriteIntensity) == 0))
    {
      StoreDataIntoFile(file_id,
                        newTemp,
                        materialProperties.edgeSize,
                        iteration / parameters.diskWriteIntensity,
                        iteration);
    }

    // [d] Swap new and old values
    swap(newTemp, oldTemp);

    // [e] Print progress and average temperature of the middle column
    if ((iteration % (parameters.nIterations / 10l)) == 
        ((parameters.nIterations / 10l) - 1l) && !parameters.batchMode)
    {
      printf("Progress %ld%% (Average Temperature %.2f degrees)\n", 
             iteration / (parameters.nIterations / 100) + 1, 
             middleColAvgTemp);
    }
  } // for iteration

  //-------------------- stop the stop watch  --------------------------------//  
  double totalTime = MPI_Wtime() - elapsedTime;

  // [7] Print final result
  if (!parameters.batchMode)
    printf("\nExecution time of sequential version %.5f\n", totalTime);
  else
    printf("%s;%s;%f;%e;%e\n", outputFileName.c_str(), "seq",
                               middleColAvgTemp, totalTime,
                               totalTime / parameters.nIterations);   

  // Close the output file
  if (file_id != H5I_INVALID_HID) H5Fclose(file_id);

  // [8] Return correct results in the correct array
  if (iteration & 1)
    memcpy(seqResult, tempArray, materialProperties.nGridPoints * sizeof(float));

  _mm_free(tempArray);
} // end of SequentialHeatDistribution
//------------------------------------------------------------------------------


/**
 * Parallel version of the Heat distribution in heterogenous 2D medium
 * @param [out] parResult          - Final heat distribution
 * @param [in]  materialProperties - Material properties
 * @param [in]  parameters         - parameters of the simulation
 * @param [in]  outputFileName     - Output file name (if NULL string, do not store)
 *
 * @note This is the function that students should implement.                                                  
 */
void ParallelHeatDistribution(float                     *parResult,
                              const TMaterialProperties &materialProperties,
                              const TParameters         &parameters,
                              string                     outputFileName)
{
  //===========================================================================
  // udelame novou kartezskou topologii s pocty procesoru
  MPI_Comm communicator_cart; //kartezsky komunikator
  int period[2] = {0,0};
  int coord[2];  //pomocne promenne pro souradnice procesoru a rank procesoru
  int rank, size;    //poradi procesoru, rozmery mrizky
  int rank_cart[2], size_cart[2];  //souradnice procesoru, rozmery mrizky

  // Get MPI rank and size
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(sqrt(size) == floor(sqrt(size))){
      //suda mocnina dvojky jako pocet parametru
      size_cart[0] = sqrt(size);
      size_cart[1] = size_cart[0];
  }
  else{
      //licha mocnina dvojky jako pocet parametru
      size_cart[1] = sqrt(size/2);
      size_cart[0] = size_cart[1]*2;
  }

  MPI_Cart_create(MPI_COMM_WORLD, 2, size_cart, period, 0, &communicator_cart);
  MPI_Cart_coords(communicator_cart, rank, 2, rank_cart);

  //===========================================================================
  //otevreme soubor pro zapis
  hid_t file_id = H5I_INVALID_HID;
  if (rank == 0){
    //Create a new output hdf5 file  
    if (outputFileName != ""){
      if (outputFileName.find(".h5") == string::npos)
        outputFileName.append("_par.h5");
      else
        outputFileName.insert(outputFileName.find_last_of("."), "_par");

      file_id = H5Fcreate(outputFileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      if (file_id < 0)
        ios::failure("Cannot create output file");
    }   
  }
  //reknu ostatnim jestli neco mame nebo ne
  MPI_Bcast(&file_id,sizeof(hid_t),MPI_BYTE,0,communicator_cart);

  //===========================================================================
  //vytvorime nove datove typy pro posilani dat
  //FLOAT
  int tile_size[2];   //velikost dlazdice
    tile_size[0] = materialProperties.edgeSize/size_cart[0];
    tile_size[1] = materialProperties.edgeSize/size_cart[1];

  MPI_Datatype tileInFull;      //dlazdice v prvnim, celem prostoru
    MPI_Type_vector (tile_size[0], tile_size[1],
                    materialProperties.edgeSize,
                    MPI_FLOAT, &tileInFull);

  MPI_Datatype tileInTile;
    MPI_Type_vector (tile_size[0], tile_size[1],
                    tile_size[1]+2,
                    MPI_FLOAT, &tileInTile);

  MPI_Datatype haloRow;
    MPI_Type_contiguous(tile_size[1], MPI_FLOAT, &haloRow);

  MPI_Datatype haloColumn;
    MPI_Type_vector (tile_size[0], 1,
                    tile_size[1]+2,
                    MPI_FLOAT, &haloColumn);

    //===================================================
    //INT
  MPI_Datatype tileInFullINT;      //dlazdice v prvnim, celem prostoru
    MPI_Type_vector (tile_size[0], tile_size[1],
                    materialProperties.edgeSize,
                    MPI_INT, &tileInFullINT);

  MPI_Datatype tileInTileINT;
    MPI_Type_vector (tile_size[0], tile_size[1],
                    tile_size[1]+2,
                    MPI_INT, &tileInTileINT);

  MPI_Datatype haloRowINT;
    MPI_Type_contiguous(tile_size[1], MPI_INT, &haloRowINT);

  MPI_Datatype haloColumnINT;
    MPI_Type_vector (tile_size[0], 1,
                    tile_size[1]+2,
                    MPI_INT, &haloColumnINT);

  MPI_Type_commit(&tileInFull);
  MPI_Type_commit(&tileInTile);
  MPI_Type_commit(&haloRow);
  MPI_Type_commit(&haloColumn);
  MPI_Type_commit(&tileInFullINT);
  MPI_Type_commit(&tileInTileINT);
  MPI_Type_commit(&haloRowINT);
  MPI_Type_commit(&haloColumnINT);

  //===========================================================================
  //alokace pameti pro data
  float *oldData = (float *)_mm_malloc((tile_size[0] +2) * //vyska +2 na okoli
                                       (tile_size[1] +2) * //sirka +2 na okoli
                                       sizeof(float), DATA_ALIGNMENT);
  float *newData = (float *)_mm_malloc((tile_size[0] +2) * //vyska +2 na okoli
                                       (tile_size[1] +2) * //sirka +2 na okoli
                                       sizeof(float), DATA_ALIGNMENT);
  float *domainParams = (float *)_mm_malloc((tile_size[0] +2) * //vyska +2 na okoli
                                            (tile_size[1] +2) * //sirka +2 na okoli
                                             sizeof(float), DATA_ALIGNMENT);
  int *domainMap = (int *)_mm_malloc((tile_size[0] +2) * //vyska +2 na okoli
                                     (tile_size[1] +2) * //sirka +2 na okoli
                                     sizeof(int), DATA_ALIGNMENT);
  //inicializace
  for(int i=0;i<tile_size[0]+2;i++){
    for(int j=0;j<tile_size[1]+2;j++){
      oldData[(tile_size[1]+2)*i+j] = -1.0f;
      newData[(tile_size[1]+2)*i+j] = -1.0f;
      domainParams[(tile_size[1]+2)*i+j] = -1.0f;
      domainMap[(tile_size[1]+2)*i+j] = -1;
    }
  }

  //===========================================================================
  //distribuce hodnot do ostatnich procesu
  #define TILESTARTFULL(coord, tile_size, edgeSize) ((coord[0]*tile_size[0]*edgeSize)+(coord[1]*tile_size[1]))
  #define TILESTART(tile_size) (tile_size[1]+3)
  #define TILERIGHT(tile_size) (tile_size[1]*2+2)
  #define TILEBOTTOM(tile_size) ((tile_size[0])*(tile_size[1]+2)+1)
  #define HALOTOP(tile_size) (1)
  #define HALOBOTTOM(tile_size) ((tile_size[0]+1)*(tile_size[1]+2)+1)
  #define HALOLEFT(tile_size) (tile_size[1]+2)
  #define HALORIGHT(tile_size) (tile_size[1]*2+3)

  if(rank==0){
    MPI_Request req[size];  
    //poslu data sobe
    MPI_Isend(materialProperties.initTemp, 1, tileInFull, 0, 0, communicator_cart, req);
    MPI_Recv(&(oldData[TILESTART(tile_size)]), 1, tileInTile, 0, 0, communicator_cart, MPI_STATUS_IGNORE);

    MPI_Isend(materialProperties.domainParams, 1, tileInFull, 0, 0, communicator_cart, req);
    MPI_Recv(&(domainParams[TILESTART(tile_size)]), 1, tileInTile, 0, 0, communicator_cart, MPI_STATUS_IGNORE);

    MPI_Isend(materialProperties.domainMap, 1, tileInFullINT, 0, 0, communicator_cart, req);
    MPI_Recv(&(domainMap[TILESTART(tile_size)]), 1, tileInTileINT, 0, 0, communicator_cart, MPI_STATUS_IGNORE);

    //poslu data ostatnim
    //DATA
    for(int i=1;i<size;i++){
      MPI_Cart_coords(communicator_cart, i, 2, coord);
      //odeslani dlazdice dat
      MPI_Isend(&(materialProperties.initTemp[TILESTARTFULL(coord,tile_size,materialProperties.edgeSize)]),
                1, tileInFull, i, 0, communicator_cart, &(req[i]));
    }
    //DOMAINPARAMS
    for(int i=1;i<size;i++){
      MPI_Wait(&(req[i]), MPI_STATUS_IGNORE);  //kontrola ze jsme dokoncili predchozi operaci
      MPI_Cart_coords(communicator_cart, i, 2, coord);
      //odeslani dlazdice dat
      MPI_Isend(&(materialProperties.domainParams[TILESTARTFULL(coord,tile_size,materialProperties.edgeSize)]),
                1, tileInFull, i, 0, communicator_cart, &(req[i]));
    }
    //DOMAINMAP
    for(int i=1;i<size;i++){
      MPI_Wait(&(req[i]), MPI_STATUS_IGNORE);  //kontrola ze jsme dokoncili predchozi operaci
      MPI_Cart_coords(communicator_cart, i, 2, coord);
      //odeslani dlazdice dat
      MPI_Isend(&(materialProperties.domainMap[TILESTARTFULL(coord,tile_size,materialProperties.edgeSize)]),
                1, tileInFullINT, i, 0, communicator_cart, &(req[i]));
    } 
  }
  else{
    //prijmeme data od nulteho
    MPI_Recv(&(oldData[TILESTART(tile_size)]), 1, tileInTile, 0, 0, communicator_cart, MPI_STATUS_IGNORE);
    MPI_Recv(&(domainParams[TILESTART(tile_size)]), 1, tileInTile, 0, 0, communicator_cart, MPI_STATUS_IGNORE);
    MPI_Recv(&(domainMap[TILESTART(tile_size)]), 1, tileInTileINT, 0, 0, communicator_cart, MPI_STATUS_IGNORE);
  }

  //===========================================================================
  //predpocitani source a destination hodnot pro posilani na mrizce
  int source[4];
  int destination[4];
  MPI_Cart_shift(communicator_cart, 0, -1, &(source[0]), &(destination[0]));
  MPI_Cart_shift(communicator_cart, 0, 1, &(source[1]), &(destination[1]));
  MPI_Cart_shift(communicator_cart, 1, -1, &(source[2]), &(destination[2]));
  MPI_Cart_shift(communicator_cart, 1, 1, &(source[3]), &(destination[3]));

  //===========================================================================
  //distribuce HALO zon - posilat NAHORU
  MPI_Sendrecv(&(oldData[TILESTART(tile_size)]), 1, haloRow, destination[0], 0,
               &(oldData[HALOBOTTOM(tile_size)]), 1, haloRow, source[0], 0,
               communicator_cart, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(domainParams[TILESTART(tile_size)]), 1, haloRow, destination[0], 0,
               &(domainParams[HALOBOTTOM(tile_size)]), 1, haloRow, source[0], 0,
               communicator_cart, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(domainMap[TILESTART(tile_size)]), 1, haloRowINT, destination[0], 0,
               &(domainMap[HALOBOTTOM(tile_size)]), 1, haloRowINT, source[0], 0,
               communicator_cart, MPI_STATUS_IGNORE);  
  // posilat DOLU
  MPI_Sendrecv(&(oldData[TILEBOTTOM(tile_size)]), 1, haloRow, destination[1], 0,
               &(oldData[HALOTOP(tile_size)]), 1, haloRow, source[1], 0,
               communicator_cart, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(domainParams[TILEBOTTOM(tile_size)]), 1, haloRow, destination[1], 0,
               &(domainParams[HALOTOP(tile_size)]), 1, haloRow, source[1], 0,
               communicator_cart, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(domainMap[TILEBOTTOM(tile_size)]), 1, haloRowINT, destination[1], 0,
               &(domainMap[HALOTOP(tile_size)]), 1, haloRowINT, source[1], 0,
               communicator_cart, MPI_STATUS_IGNORE);  
  // posilat VLEVO
  MPI_Sendrecv(&(oldData[TILESTART(tile_size)]), 1, haloColumn, destination[2], 0,
               &(oldData[HALORIGHT(tile_size)]), 1, haloColumn, source[2], 0,
               communicator_cart, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(domainParams[TILESTART(tile_size)]), 1, haloColumn, destination[2], 0,
               &(domainParams[HALORIGHT(tile_size)]), 1, haloColumn, source[2], 0,
               communicator_cart, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(domainMap[TILESTART(tile_size)]), 1, haloColumn, destination[2], 0,
               &(domainMap[HALORIGHT(tile_size)]), 1, haloColumn, source[2], 0,
               communicator_cart, MPI_STATUS_IGNORE);  
  // posilat VPRAVO
  MPI_Sendrecv(&(oldData[TILERIGHT(tile_size)]), 1, haloColumn, destination[3], 0,
               &(oldData[HALOLEFT(tile_size)]), 1, haloColumn, source[3], 0,
               communicator_cart, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(domainParams[TILERIGHT(tile_size)]), 1, haloColumn, destination[3], 0,
               &(domainParams[HALOLEFT(tile_size)]), 1, haloColumn, source[3], 0,
               communicator_cart, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(domainMap[TILERIGHT(tile_size)]), 1, haloColumn, destination[3], 0,
               &(domainMap[HALOLEFT(tile_size)]), 1, haloColumn, source[3], 0,
               communicator_cart, MPI_STATUS_IGNORE);  

  //===========================================================================
  //spusteni hodin
  double elapsedTime = 0.0;
  if(rank==0)
    elapsedTime = MPI_Wtime();

  //===========================================================================
  //novy komunikator pro prumernou hodnotu
  MPI_Comm avgcomm;
    if(rank==0 || rank_cart[1]==size_cart[1]/2)
      MPI_Comm_split(MPI_COMM_WORLD, 1, rank_cart[0]+1, &avgcomm);
    else
      MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 0, &avgcomm);

  //===========================================================================
  //inicializace newData
  for(int i=0;i<(tile_size[0] +2)*(tile_size[1] +2);i++)
    newData[i] = oldData[i];

  //===========================================================================
  // spusteni simulace
  MPI_Request req;
  int i, j;
  size_t iteration;
  float middleColAvgTemp, avg;

  for (iteration = 0; iteration < parameters.nIterations; iteration++){
    //=========================================================================
    //spocitam HALO - nahore
    if(rank_cart[0]!=0){
      //spocitam roh vlevo
      if(rank_cart[1]!=0)
        ComputePoint(oldData, newData, domainParams, domainMap, 1, 1, tile_size[1]+2, 
                     parameters.airFlowRate, materialProperties.coolerTemp); 
      //spocitam roh vpravo
      if(rank_cart[1]!=size_cart[1]-1)
        ComputePoint(oldData, newData, domainParams, domainMap, 1, tile_size[1], tile_size[1]+2, 
                     parameters.airFlowRate, materialProperties.coolerTemp);
      //spocitam prostredek
      for(i=2;i<tile_size[1];i++)
        ComputePoint(oldData, newData, domainParams, domainMap, 1, i, tile_size[1]+2, 
                     parameters.airFlowRate, materialProperties.coolerTemp);
    }
    //spocitam halo - dole
    if(rank_cart[0]!=size_cart[0]-1){
      //spocitam roh vlevo
      if(rank_cart[1]!=0)
        ComputePoint(oldData, newData, domainParams, domainMap, tile_size[0], 1, tile_size[1]+2, 
                     parameters.airFlowRate, materialProperties.coolerTemp); 
      //spocitam roh vpravo
      if(rank_cart[1]!=size_cart[1]-1)
        ComputePoint(oldData, newData, domainParams, domainMap, tile_size[0], tile_size[1], tile_size[1]+2, 
                     parameters.airFlowRate, materialProperties.coolerTemp);
      //spocitam prostredek
      for(i=2;i<tile_size[1];i++)
        ComputePoint(oldData, newData, domainParams, domainMap, tile_size[0], i, tile_size[1]+2, 
                     parameters.airFlowRate, materialProperties.coolerTemp);
    }
    //spocitam halo - vlevo
    if(rank_cart[1]!=0){
      //spocitam prostredek
      for(i=2;i<tile_size[0];i++)
        ComputePoint(oldData, newData, domainParams, domainMap, i, 1, tile_size[1]+2, 
                     parameters.airFlowRate, materialProperties.coolerTemp);
    }
    //spocitam halo - vlevo
    if(rank_cart[1]!=size_cart[1]-1){
      //spocitam prostredek
      for(i=2;i<tile_size[0];i++)
        ComputePoint(oldData, newData, domainParams, domainMap, i, tile_size[1], tile_size[1]+2, 
                     parameters.airFlowRate, materialProperties.coolerTemp);
    }

    //=========================================================================
    //komunikace a vymena hodnot - posilat NAHORU
    MPI_Isend(&(newData[TILESTART(tile_size)]), 1, haloRow, destination[0], 0, communicator_cart, &req);
    // posilat DOLU  
    MPI_Isend(&(newData[TILEBOTTOM(tile_size)]), 1, haloRow, destination[1], 0, communicator_cart, &req);
    // posilat VLEVO    
    MPI_Isend(&(newData[TILESTART(tile_size)]), 1, haloColumn, destination[2], 0, communicator_cart, &req);
    // posilat VPRAVO    
    MPI_Isend(&(newData[TILERIGHT(tile_size)]), 1, haloColumn, destination[3], 0, communicator_cart, &req);

    //=========================================================================
    //spocitani PROSTREDKU
    for(i=2;i<tile_size[0];i++)
      for(j=2;j<tile_size[1];j++)
        ComputePoint(oldData, newData, domainParams, domainMap, i, j, tile_size[1]+2, 
                     parameters.airFlowRate, materialProperties.coolerTemp);

    //=========================================================================
    //prijeti hodnot
    MPI_Recv(&(newData[HALOBOTTOM(tile_size)]), 1, haloRow, source[0], 0, communicator_cart, MPI_STATUS_IGNORE);
    MPI_Recv(&(newData[HALOTOP(tile_size)]), 1, haloRow, source[1], 0, communicator_cart, MPI_STATUS_IGNORE);
    MPI_Recv(&(newData[HALORIGHT(tile_size)]), 1, haloColumn, source[2], 0, communicator_cart, MPI_STATUS_IGNORE);
    MPI_Recv(&(newData[HALOLEFT(tile_size)]), 1, haloColumn, source[3], 0, communicator_cart, MPI_STATUS_IGNORE);

    //=========================================================================
    //vypocet prumerne teploty prostredku
    middleColAvgTemp = 0.0f;
    avg=0.0f;
    
    if(size_cart[1]>1){
      //pokud mam vic nez jeden sloupec tak je to normalni a pocitame nejlevejsi
      //sloupec prostrednich procesu
       if(rank_cart[1]==size_cart[1]/2){
        //pokud jsem uprostred
        for(i=1;i<tile_size[0]+1;i++){
            avg+=newData[i*(tile_size[1]+2)+1];
        }
        avg/=tile_size[0];    
        MPI_Reduce(&avg, &middleColAvgTemp, 1, MPI_FLOAT, MPI_SUM, 0, avgcomm);               
      }
      else if(rank==0){
        //nula prijme redukci a pocita prumer
        MPI_Reduce(&avg, &middleColAvgTemp, 1, MPI_FLOAT, MPI_SUM, 0, avgcomm);               
        middleColAvgTemp/=size_cart[0];      
      }
    }
    else{
      //mam jen jeden proces na radku, takze sem se dostanou vsechny procesy
      //a nepocitaji nejlevejsi sloupec ale ten uprostred
      for(i=1;i<tile_size[0]+1;i++){
          avg+=newData[i*(tile_size[1]+2)+tile_size[1]/2+1];
      }
      avg/=tile_size[0];    
      MPI_Reduce(&avg, &middleColAvgTemp, 1, MPI_FLOAT, MPI_SUM, 0, avgcomm);            
      middleColAvgTemp/=size_cart[0];      
    }

    //=========================================================================
    //ulozeni do souboru            
    if((file_id != H5I_INVALID_HID)  && ((iteration % parameters.diskWriteIntensity) == 0)){
      if(rank==0){
        //poslani sobe
        MPI_Isend(&(newData[TILESTART(tile_size)]), 1, tileInTile, 0, 0, communicator_cart, &req);
        MPI_Recv(parResult, 1, tileInFull, 0, 0, communicator_cart, MPI_STATUS_IGNORE);
        //prijeti od ostatnich
        for(i=1;i<size;i++){
            MPI_Cart_coords(communicator_cart, i, 2, coord);
            MPI_Recv(&(parResult[TILESTARTFULL(coord,tile_size,materialProperties.edgeSize)]), 1, tileInFull,
                     i, 0, communicator_cart,MPI_STATUS_IGNORE);
        }
        // ulozeni
        StoreDataIntoFile(file_id, parResult, materialProperties.edgeSize,
                          iteration / parameters.diskWriteIntensity, iteration);
      } 
      else{   //poslani dat nultemu
        MPI_Isend(&(newData[TILESTART(tile_size)]), 1, tileInTile, 0, 0, communicator_cart, &req);
      } 
    }

    //=========================================================================
    // Print progress and average temperature of the middle column
    if(rank==0){
      if ((iteration % (parameters.nIterations / 10l)) == 
        ((parameters.nIterations / 10l) - 1l) && !parameters.batchMode){
        printf("Progress %ld%% (Average Temperature %.2f degrees)\n", 
               iteration / (parameters.nIterations / 100) + 1, middleColAvgTemp);
      }
    }

    //=========================================================================
    swap(newData, oldData);
  } //for iteration end

  //===========================================================================
  //zastaveni hodin
  double totalTime;
  if(rank==0){
    totalTime = MPI_Wtime() - elapsedTime;

    //tisk vysledku
    if (!parameters.batchMode){
        printf("\nExecution time of parallel version %.5f\n", totalTime);
    }
    else{
        printf("%s;%s;%f;%e;%e\n", outputFileName.c_str(), "par",
                 middleColAvgTemp, totalTime, totalTime / parameters.nIterations);
    }
    //uzavreni souboru
    if (file_id != H5I_INVALID_HID)
      H5Fclose(file_id);
  }

  //zpracovani spravnych vysledku
  if(rank==0){
    //poslani sobe
    MPI_Isend(&(oldData[TILESTART(tile_size)]), 1, tileInTile, 0, 0, communicator_cart, &req);
    MPI_Recv(parResult, 1, tileInFull, 0, 0, communicator_cart, MPI_STATUS_IGNORE);
    //prijeti od ostatnich
    for(i=1;i<size;i++){
        MPI_Cart_coords(communicator_cart, i, 2, coord);
        MPI_Recv(&(parResult[TILESTARTFULL(coord,tile_size,materialProperties.edgeSize)]), 1, tileInFull,
                 i, 0, communicator_cart,MPI_STATUS_IGNORE);
    }
  } 
  else{   //poslani dat nultemu
    MPI_Isend(&(oldData[TILESTART(tile_size)]), 1, tileInTile, 0, 0, communicator_cart, &req);
  } 

  MPI_Type_free(&tileInFull);
  MPI_Type_free(&tileInTile);
  MPI_Type_free(&haloRow);
  MPI_Type_free(&haloColumn);
  MPI_Type_free(&tileInFullINT);
  MPI_Type_free(&tileInTileINT);
  MPI_Type_free(&haloRowINT);
  MPI_Type_free(&haloColumnINT);
} // end of ParallelHeatDistribution
//------------------------------------------------------------------------------


/**
 * Store time step into output file (as a new dataset in Pixie format
 * @param [in] h5fileID   - handle to the output file
 * @param [in] data       - data to write
 * @param [in] edgeSize   - size of the domain
 * @param [in] snapshotId - snapshot id
 * @param [in] iteration  - id of iteration);
 */
void StoreDataIntoFile(hid_t         h5fileId,
                       const float  *data,
                       const size_t  edgeSize,
                       const size_t  snapshotId,
                       const size_t  iteration)
{
  hid_t   dataset_id, dataspace_id, group_id, attribute_id;
  hsize_t dims[2] = {edgeSize, edgeSize};

  string groupName = "Timestep_" + to_string((unsigned long long) snapshotId);

  // Create a group named "/Timestep_snapshotId" in the file.
  group_id = H5Gcreate(h5fileId,
                       groupName.c_str(),
                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


  // Create the data space. (2D matrix)
  dataspace_id = H5Screate_simple(2, dims, NULL);

  // create a dataset for temperature and write data
  string datasetName = "Temperature";
  dataset_id = H5Dcreate(group_id,
                         datasetName.c_str(),
                         H5T_NATIVE_FLOAT,
                         dataspace_id,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset_id,
           H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           data);

  // close dataset
  H5Sclose(dataspace_id);


  // write attribute
  string atributeName="Time";
  dataspace_id = H5Screate(H5S_SCALAR);
  attribute_id = H5Acreate2(group_id, atributeName.c_str(),
                            H5T_IEEE_F64LE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT);

  double snapshotTime = double(iteration);
  H5Awrite(attribute_id, H5T_IEEE_F64LE, &snapshotTime);
  H5Aclose(attribute_id);


  // Close the dataspace.
  H5Sclose(dataspace_id);

  // Close to the dataset.
  H5Dclose(dataset_id);
} // end of StoreDataIntoFile
//------------------------------------------------------------------------------


/**
 * Main function of the project
 * @param [in] argc
 * @param [in] argv
 * @return
 */
int main(int argc, char *argv[])
{
  int rank, size;

  ParseCommandline(argc, argv, parameters);

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Get MPI rank and size
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  if (rank == 0)
  {
    // Create material properties and load from file
    materialProperties.LoadMaterialData(parameters.materialFileName, true);
    parameters.edgeSize = materialProperties.edgeSize;

    parameters.PrintParameters();
  }
  else
  {
    // Create material properties and load from file
    materialProperties.LoadMaterialData(parameters.materialFileName, false);
    parameters.edgeSize = materialProperties.edgeSize;
  }

  if (parameters.edgeSize % size)
  {
    if (rank == 0)
      printf("ERROR: number of MPI processes is not a divisor of N\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (parameters.IsRunSequntial())
  {
    if (rank == 0)
    {
      // Memory allocation for output matrices.
      seqResult = (float*)_mm_malloc(materialProperties.nGridPoints * sizeof(float), DATA_ALIGNMENT);

      SequentialHeatDistribution(seqResult,
                                 materialProperties,
                                 parameters,
                                 parameters.outputFileName);
    }
  }

  if (parameters.IsRunParallel())
  {
    // Memory allocation for output matrix.
    if (rank == 0)
      parResult = (float*) _mm_malloc(materialProperties.nGridPoints * sizeof(float), DATA_ALIGNMENT);
    else
      parResult = NULL;

    ParallelHeatDistribution(parResult,
                             materialProperties,
                             parameters,
                             parameters.outputFileName);
  }

  // Validate the outputs
  if (parameters.IsValidation() && rank == 0)
  {
    if (parameters.debugFlag)
    {
      printf("---------------- Sequential results ---------------\n");
      PrintArray(seqResult, materialProperties.edgeSize);

      printf("----------------- Parallel results ----------------\n");
      PrintArray(parResult, materialProperties.edgeSize);
    }

    if (VerifyResults(seqResult, parResult, parameters, 0.001f))
      printf("Verification OK\n");
    else
      printf("Verification FAILED\n");
  }

  /* Memory deallocation*/
  _mm_free(seqResult);
  _mm_free(parResult);

  MPI_Finalize();

  return EXIT_SUCCESS;
} // end of main
//------------------------------------------------------------------------------
