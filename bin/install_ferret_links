#! /bin/csh
# install FERRET links in the /usr/local/bin area - eliminating the need to
# modify the PATH variable
# ver 1.0 4/92 *sh* - based on INSTALL_BIN routine

# procedure options include
# "i" (install) - copies sources, links and other files to installation area
# "r" (remove)  - removes the "i" files

ask:
echo "Install (i) or remove (r)?"
set activity = $<
switch($activity)
case "i":
  echo "Installing FERRET links"
  breaksw
case "r":
  echo "Removing FERRET links"
  breaksw
default:
  echo "You must answer i or r"
  goto ask
endsw
date

# basic definitions
set source_area = $FER_DIR/bin
set dest_area = /usr/local/bin

cd $source_area
set files = *
foreach file ($files)
  switch($activity)
  case "i":
    if ($file !~ *~ ) ln -s $source_area/$file $dest_area/$file
    breaksw
  case "r":
    rm -f $dest_area/$file
    breaksw
  endsw
end

date