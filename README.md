# Digit_recognition
A program created from scratch to recognise handwritten digits after training on a huge set of digits from mnist's handwritten digit dataset.
This program is trained over 19506 handwritten digits and is tested over 2048 handwritten samples. You can train your own ANN(Artificial Neural network) for your own 
dataset of digits of 28 by 28 size after deleting the values inside the weights and biases.
The current weights and biases in this folder are trained by me and works pretty accurately, although it's not 100% correct. You can check the trained program by adding your own handwritten digits images in the folder :"image_folder" of size 28 by 28
<img width="1502" height="1017" alt="image" src="https://github.com/user-attachments/assets/62f4f265-177c-4be5-a11d-54ae2e762488" />
In the image above it is displayed that how the output would look after running the code. First few lines are the location of images themselves which were being augmented during the process in order to develop a better generalisation for the neural network and then there are the details of the images it tried to predict along with the accuracy and losses and then in the last line it shows the digits that the ANN failed to predict correctly which in this case is 3 out of 29 images.
