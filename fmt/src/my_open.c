#include <stdio.h>

void my_open_(    void  **lun,
                   char *fname,
                   int  *len_fname,
                   int  *is_success){

   int true_len_fname, i;
   char *c_fname,ch;
   FILE * fp;

   tm_get_strlen_(&true_len_fname, len_fname, fname);
   if(true_len_fname<=0) return;

   c_fname =(char *)malloc(true_len_fname+1);
   for(i=0;i<true_len_fname;i++){
      c_fname[i] = fname[i];
   }
   c_fname[true_len_fname] = 0;

   fp = fopen(c_fname,"r");
   //      printf("\nopen fp=%p",fp);
   if(fp)
     *is_success = 1;
   else
     *is_success = 0;

   free(c_fname);
 
   (FILE*)(*lun) = fp;
}

