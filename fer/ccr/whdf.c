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

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <signal.h>
#include <malloc.h>
#include <X11/Xlib.h>


void wHDF(file, image,r,g,b)
     char *file;
     XImage *image;
     int r[],g[],b[];
{

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
}


