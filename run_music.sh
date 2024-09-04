echo "Files in initial data folder...."
ls -R initial_data_GSR/

echo "Insert input file path:"
read path

echo "Copying input file to temporary output folder..."
cp $path ./temp-output/

./MUSIChydro $path


