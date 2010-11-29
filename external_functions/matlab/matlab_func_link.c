#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "engine.h"
#define  BUFSIZE 8192
#define NaN (0.0/0.0)		/* Anyone know a better way? */

static void convertToDouble(float *data, double *ddata, float bad_flag,
			    int ysize, int zsize, int maxy, int maxz);
static void convertToFloat(double *ddata, float *data, float bad_flag,
			    int ysize, int zsize, int maxy, int maxz);


static Engine *ep = 0;
int matlab_func_demo_(int *getFunction,
		      float *sal, float *temp, float *pres, float *res,
			 int *ysize, int *zsize, float *bad_flag,
		         int *maxy, int *maxz){
  mxArray *mSal = NULL, *mTemp = NULL, *mPres = NULL, *mRes = NULL;
  double *dSal, *dTemp, *dPres, *dRes;
  char buffer[BUFSIZE], cbuffer[BUFSIZE];
  int size = *ysize * *zsize;
  static int choice = 0;

				/* Open MATLAB engine */
  if (ep == 0){
    if (!(ep = engOpen("\0"))) {
      fprintf(stderr, "\nCan't start MATLAB engine\n");
      return EXIT_FAILURE;
    }
  }

				/* Set up the MATLAB output buffer */
  engOutputBuffer(ep, buffer, BUFSIZE);

  mSal = mxCreateDoubleMatrix(*ysize, *zsize, mxREAL);
  mxSetName(mSal, "SAL");
  dSal = mxGetPr(mSal);
  mTemp = mxCreateDoubleMatrix(*ysize, *zsize, mxREAL);
  mxSetName(mTemp, "TEMP");
  dTemp = mxGetPr(mTemp);
  mPres = mxCreateDoubleMatrix(1, *zsize, mxREAL);
  mxSetName(mPres, "PRES");
  dPres = mxGetPr(mPres);

				/* Convert from float->double */
  convertToDouble(temp, dTemp, *bad_flag, *ysize, *zsize, *maxy, *maxz);
  convertToDouble(sal, dSal, *bad_flag, *ysize, *zsize, *maxy, *maxz);
  convertToDouble(pres, dPres, *bad_flag, *zsize, 1, *zsize, 1);

  /* Place the variables into the MATLAB workspace */
  engPutArray(ep, mSal);
  engPutArray(ep, mPres);
  engPutArray(ep, mTemp);

  /* Get the function choice */
  if (*getFunction){
    choice = 0;
    while (choice < 1 || choice > 5){
      printf("1) sw_adtg (adiabatic temperature gradient)\n");
      printf("2) sw_cndr (conductivity ratio)\n");
      printf("3) sw_cp (heat capacity)\n");
      printf("4) sw_dens (density)\n");
      printf("5) sw_gpan (geopotential anomaly)\n");
      printf("Select a function: ");
      scanf("%d", &choice);
    }
  }


  sprintf(cbuffer, "res = matlab_func_test(%d,SAL,TEMP,PRES);", choice);
  engEvalString(ep, cbuffer);
  /*
   * Echo the output from the command.  First two characters are
   * always the double prompt (>>).
   */
  if (strlen(buffer) > 1)
    printf("%s", buffer+2);

  /* Get the result array */
  mRes = engGetArray(ep,"res");
  assert(mRes);
  dRes = mxGetPr(mRes);
  convertToFloat(dRes, res, *bad_flag, *ysize , *zsize, *maxy, *maxz);
 

  mxDestroyArray(mSal);
  mxDestroyArray(mPres);
  mxDestroyArray(mTemp);
  mxDestroyArray(mRes);
  
#if 0
   engClose(ep);
#endif
	
  return EXIT_SUCCESS;
}

static void convertToFloat(double *ddata, float *data, float bad_flag,
			    int ysize, int zsize, int maxy, int maxz)
{

  int i=0,j,k;
  for (k=0; k < zsize; ++k){
    for (j=0; j < ysize; ++j){
      double theData = ddata[i++];
      int index = k * maxy + j;
      if (isnan(theData)){
	data[index] = bad_flag;
      } else {
	data[index] = theData;
      }
#if 0
      printf("Converting: %d %lf -> %f\n", i, theData, data[i]);
#endif
    }
  }
}

static void convertToDouble(float *data, double *ddata, float bad_flag,
			    int ysize, int zsize, int maxy, int maxz)
{

  int i=0,j,k;
  for (k=0; k < zsize; ++k){
    for (j=0; j < ysize; ++j){
      int index = k * maxy + j;
      float theData = data[index];
      if (theData == bad_flag){
	ddata[i++] = NaN;
      } else {
	ddata[i++] = theData;
      }
#if 0
      printf("Converting: %d %f -> %lf\n", i-1, theData, ddata[i-1]);
#endif
    }
  }
}








