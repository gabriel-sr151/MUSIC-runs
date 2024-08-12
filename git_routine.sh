git add .

echo "Insert update message:"
read msg

if [ -z "$msg"]; then 
   echo "No message. Continuing....."
   exit 1
fi

git commit -m "$msg" 

git push -u origin public_stable

