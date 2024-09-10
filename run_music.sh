echo "Files in initial data folder...."
ls -R initial_data_GSR/

echo "Insert input file path:"
read path

echo "Copying input file to temporary output folder..."
cp $path ./temp-output/

echo "MUSIC is running ..."

logfile='temp-output/log_music.txt' 
./MUSIChydro $path > $logfile

echo "MUSIC ended."

