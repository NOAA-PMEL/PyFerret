#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "engine.h"
#define  BUFSIZE 8192
#define NaN (0.0/0.0)		/* Anyone know a better way? */

static Engine *ep = 0;
/*
 * JS Changed coord arguments to double as external function API changed
 */
int matlab_demo_(float *data,
		 double *xcoords, int *xsize,
		 double *ycoords, int *ysize,
		 float *bad_flag) {
  mxArray *T = NULL, *result = NULL;
  mxArray *mX = NULL, *mY = NULL;
  char buffer[BUFSIZE], cbuffer[BUFSIZE];
  double *ddata = 0, *mxData = 0, *myData = 0;;

				/* Open MATLAB engine */
  if (ep == 0){
    if (!(ep = engOpen("\0"))) {
      fprintf(stderr, "\nCan't start MATLAB engine\n");
      return EXIT_FAILURE;
    }
  }

  engOutputBuffer(ep, buffer, BUFSIZE);
				/* Convert from float->double */
  T = mxCreateDoubleMatrix(*xsize, *ysize, mxREAL);
  mxSetName(T, "FERRET_DATA");
  ddata = mxGetPr(T);

  mX = mxCreateDoubleMatrix(*xsize, 1, mxREAL);
  mxSetName(mX, "FERRET_XDATA");
  mxData = mxGetPr(mX);
  mY = mxCreateDoubleMatrix(*ysize, 1, mxREAL);
  mxSetName(mY, "FERRET_YDATA");
  myData = mxGetPr(mY);
  
  {
    int i;
    for (i=0; i < *xsize * *ysize; ++i){
      float theData = data[i];
      if (theData == *bad_flag){
	ddata[i] = NaN;
      } else {
	ddata[i] = data[i];
      }
    }

    memcpy(mxData, xcoords, sizeof(double)* *xsize);
    memcpy(myData, ycoords, sizeof(double)* *ysize);
  }

  /* Place the variables into the MATLAB workspace */
  engPutArray(ep, T);
  engPutArray(ep, mX);
  engPutArray(ep, mY);

  
  sprintf(cbuffer, "ferretdemo(FERRET_DATA', FERRET_XDATA', FERRET_YDATA');");
  engEvalString(ep, cbuffer);
  /*
   * Echo the output from the command.  First two characters are
   * always the double prompt (>>).
   */
  printf("%s", buffer+2);

  mxDestroyArray(T);
/*   engEvalString(ep, "close;"); */
/*   engClose(ep); */
	
  return EXIT_SUCCESS;
}







