/*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.  
*
*/



/* whdf.c - containing 

   void wHDF(char *file, XImage *image,int r[],int g[], int b[]);

 NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

 JAN. '95 - Kevin O'Brien

 Routine for writing out HDF files */

/* revision history:
   9/6/95 *sh* - repaired memory leak ("free" not called)

   5/30/96 *kob* - write error messages to stderr
                   check validity of height and width

*/
/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
/* *acm   9/06 v600 - add stdlib.h wherever there is stdio.h for altix build*/ 
#include <wchar.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <signal.h>
#include <X11/Xlib.h>


void wHDF(file, image,r,g,b)
     char *file;
     XImage *image;
     int r[],g[],b[];
{
#ifdef __CYGWIN__
  fprintf(stderr, "wHDF not supported on this platform\n");
#else

  char *data,*pdata;
  int hheight, hwidth,istat,i,j;
  char palette[768], *p;
  int rle=11; /* set compression value for hdf encoding */

/* first convert r,g,b values to palette for hdf */
  p = palette;
  for (i=0; i<256; i++) {
    *p++ = r[i];
    *p++ = g[i];
    *p++ = b[i];
  }

/* set up some parameters */

  data = image->data;
  hheight = image->height;
  hwidth= image->bytes_per_line;

/* check height and width *kob* 5/96 */
  if (hheight < 0 || hheight > 5000 || hwidth < 0 || hwidth >5000) {
    fprintf (stderr, "\n height/width out of range: %d\t%d",hheight,hwidth);
    exit(1);
  }

  pdata = (char *)malloc(sizeof(char) * (hheight * hwidth));
  for (j=0;j<image->height; j++)
    for (i=0;i<image->width; i++) {
      pdata[i+j*hwidth] = data[j*hwidth+i];
    }

/* write out the palette */

/* write errors out to stderr *kob* 5/96 */
  istat = DFR8setpalette(palette); 
  if (istat != 0) {
    printf ("\nError writing palette....\n");
    free(pdata);
    exit(1);
  }
/* write out the image */
   istat = DFR8addimage(file,pdata,hwidth,hheight,rle);
   free(pdata);
      if (istat != 0) {
	printf("*****Error writing HDF file*****\n");
	exit (1);
      }
#endif /* #ifdef __CYGWIN__ */
}



