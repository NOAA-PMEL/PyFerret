#include <stdio.h>

void my_readline_(  void **lun,
                     char *out_string,
                     int *len_out_string,
                     int *status
                   )
{

   int i, buff_size,fill;
   FILE *fp;
   char *pch,ch;

   fp = *((FILE**)lun);

   if(fp==NULL){
     *status = -1;
     return;
   }
   buff_size = *len_out_string;
   
   pch = fgets(out_string, buff_size, fp);
   
   if(pch==NULL){
      *status = 0;
   }
   else{
      *status = 1;
      fill = 0;
      for(i=0;i<buff_size;i++){
	if(out_string[i]==0){
	   fill =1;
	}
        if(fill||out_string[i]==10)
	   out_string[i]=' ';
      }
   } 
   
}


