echo "transfering data from temp to final folder ..."

echo "list of folders in final data folder"
ls final_data_files 

echo "Insert project name" 
read project_folder  

echo "Insert folder name" 
read new_folder_name  

path="final_data_files/${project_folder}/${new_folder_name}"

echo $path
mkdir -p $path

mv temp-output/evolution_* $path
mv temp-output/input_* $path
mv temp-output/log_* $path

echo "removing stuff from temp-output folder..."
rm temp-output/*
