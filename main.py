import cv2
import numpy
import os
import random
import datetime
import ast
import time

class image_info:
    def __init__(self,image,number,true_value):
        self.images = image
        self.number = number
        self.true_value = true_value

        
class Network:
    def __init__(self,folder,learn = True,test_folder = None,weights = [[],[],[]],biases = [[],[],[]]):
        self.folder = folder
        self.learn = learn
        with open('weights.txt','r') as file:
            weight = file.read()
            try:
                weights = ast.literal_eval(weight)
            except:
                pass
            # print(weights[1])
        with open('biases.txt','r') as file:
            biase = file.read()
            try:
                biases = ast.literal_eval(biase)
            except:
                pass
        # print(self.folder_list)
        self.weight1,self.weight2,self.weight3 = numpy.array(weights[0]),numpy.array(weights[1]),numpy.array(weights[2])
        self.bias1,self.bias2,self.bias3 = numpy.array(biases[0]),numpy.array(biases[1]),numpy.array(biases[2])
        # print(self.true_val)

        count = 0
        threshold = 2100
        wrong_len = threshold
        ratio = 10
        previous_validation = 0
        validation = 0
        wrong_epoches = 0
        training = True
        validation_losses = numpy.array([])
        minimum_validation = 10
        self.folder_list = os.listdir(self.folder)
        main_image_list = self.augmentation(folder)
        if test_folder:
            test_image_list = self.augmentation(test_folder)
        self.image_list = main_image_list
        # for i in range(1):
        while validation - previous_validation <= 0 or wrong_epoches <= 3:
            folder_len = len(self.folder_list)
            wrong_answer = []
            # while (ratio > 1/5 and wrong_len < 100) or wrong_len < 100:
            random.shuffle(self.image_list)
            for image in self.image_list:
                self.true_val = image.true_value
                num = image.number
                images = image.images
                self.true_val = numpy.array(self.true_val)
                # print(image)
                pixel_list = self.first_layer(images)
                hidden_layer1,self.bias1,self.weight1,l1 = self.hidden_layer(pixel_list,256,self.bias1,self.weight1)
                hidden_layer2,self.bias2,self.weight2,l2 = self.hidden_layer(hidden_layer1[0],128,self.bias2,self.weight2)
                output_layer,self.bias3,self.weight3,l3 = self.output_layer(hidden_layer2[0],10,self.bias3,self.weight3)
                loss = self.cross_entropy_loss(self.true_val,output_layer)
                if training == True:
                    weights,biases = self.back_prop(self.true_val,[l3,l2,l1,[pixel_list]])
                else:
                    print(loss)
                    validation_losses = numpy.append(validation_losses,loss)
                    print(validation_losses)
                self.weight3,self.weight2,self.weight1 = weights[0],weights[1],weights[2]
                self.bias3,self.bias2,self.bias1 = biases[0],biases[1],biases[2]
                result = max(output_layer[0])
                for i,j in enumerate(output_layer[0]):
                    if output_layer[0][i] == result:
                        answer = i
                if answer != num:
                    wrong_answer.append((answer,num))
                print(f'count = {count},predicted_answer = {answer},true_answer = {num},percent_eval = {result * 10},loss = {loss},validation = {validation}')
                count +=1
            wrong_len = len(wrong_answer)
            ratio = wrong_len/ count
            print(wrong_answer)
            print(wrong_len)
            print(ratio,1/5)
            if test_folder :
                if training == True:
                    self.image_list = test_image_list
                    training = False
                else:
                    self.image_list = main_image_list
                    previous_validation = validation
                    validation = self.validation_eval(validation_losses)
                    print(validation- previous_validation,'<<<<< Here',minimum_validation)
                    if validation < minimum_validation:
                        minimum_validation = validation
                        weights,biases = self.weights_biases()
                        if self.learn == True:
                            with open('weights.txt','w') as file:
                                file.write(f'{weights}')
                            with open('biases.txt','w') as file:
                                file.write(f'{biases}')
                        self.w1,self.w2,self.w3,self.b1,self.b2,self.b3 = self.weight1,self.weight2,self.weight3,self.bias1,self.bias2,self.bias3
                    if validation - previous_validation > 0:
                        wrong_epoches += 1
                    else:
                        wrong_epoches = 0
                    training = True
            time.sleep(2.5)
            # print(loss)
        # print(output_layer,output_layer.sum())
        # print(len(hidden_layer1[0]),len(hidden_layer2[0]))
        if test_folder:
            self.weight1,self.weight2,self.weight3,self.bias1,self.bias2,self.bias3 = self.w1,self.w2,self.w3,self.b1,self.b2,self.b3

    def augmentation(self,folder):
        folder_list = os.listdir(folder)
        image_list = []
        for image in folder_list:
            true_val = []
            if 'Zero' in image:
                num = 0
            elif 'One' in image:
                num = 1
            elif 'Two' in image:
                num = 2
            elif 'Three' in image:
                num = 3
            elif 'Four' in image:
                num = 4
            elif 'Five' in image:
                num = 5
            elif 'Six' in image:
                num = 6
            elif 'Seven' in image:
                num = 7
            elif 'Eight' in image:
                num = 8
            elif 'Nine' in image:
                num = 9
            im = cv2.imread(f'{folder}\\{image}',cv2.IMREAD_GRAYSCALE)
            im_rotate = cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE)
            im_flip = cv2.flip(im,32)
            
            for i in range(10):
                if i != num:
                    true_val.append(0)
                else:
                    true_val.append(1)
            im_info = image_info(im,num,true_val)
            im_rotate_info = image_info(im_rotate,num,true_val)
            im_flip_info = image_info(im_flip,num,true_val)
            image_list.append(im_info)
            if self.learn == True:
                image_list.append(im_rotate_info)
                image_list.append(im_flip_info)
            print(f'{folder}\\{image}')
        return image_list
    def weights_biases(self):
        return [self.weight1.tolist(),self.weight2.tolist(),self.weight3.tolist()],[self.bias1.tolist(),self.bias2.tolist(),self.bias3.tolist()]
    def first_layer(self,image):
        img = image
        img = cv2.resize(img,(28,28))
        # cv2.imshow(image,img)
        img = img/255.0
        img = img.flatten()
        # cv2.waitKey(0)
        return img

    def relu(self,x):
        return numpy.maximum(0,x)

    def softmax(self,matrix):
        exp_values = numpy.exp(matrix)
        # print(exp_values)
        return exp_values / numpy.sum(exp_values, axis=1, keepdims=True)
    def validation_eval(self,matrix):
        print(len(matrix))
        return numpy.sum(matrix, axis=0, keepdims=True)/len(matrix)

    def hidden_layer(self,parent_layer,length,biases, weights = None):
        parent_length = len(parent_layer)
        parent_layer = numpy.array(parent_layer)
        # print(parent_layer.shape,parent_layer)
        if biases is None or biases.size == 0:
            biases = numpy.zeros((1, length))
        if weights is None or weights.size == 0:
            weights = numpy.random.randn(parent_length, length) * numpy.sqrt(1.0 / parent_length)
        preactivation_layer = parent_layer.dot(weights) + biases
        activation_layer = self.relu(preactivation_layer)
        layer = [activation_layer[0],preactivation_layer,weights,biases,"relu",parent_layer]
        return activation_layer,biases,weights,layer

    def output_layer(self,parent_layer,length,biases,weights = None):
        parent_length = len(parent_layer)
        parent_layer = numpy.array(parent_layer)
        if biases is None or biases.size == 0:
            biases = numpy.zeros((1, length))
        if weights is None or weights.size == 0:
            weights = numpy.random.randn(parent_length, length) * numpy.sqrt(1.0 / parent_length)
            # print(weights,'<<<<<here')
        preactivation_layer = parent_layer.dot(weights) + biases
        activation_layer = self.softmax(preactivation_layer)
        # print(numpy.tile(activation_layer,(parent_length,1)).T)
        # print(activation_layer)
        layer = [activation_layer[0],preactivation_layer,weights,biases,"softmax",parent_layer]
        return activation_layer,biases,weights,layer
    
    def cross_entropy_loss(self,true_val,pred_val):
        pred_val = numpy.clip(pred_val,1e-9,1. - 1e-9)
        return -numpy.sum(true_val * numpy.log(pred_val))
    
    def relu_derivative(self,z):
        return (z > 0).astype(float)

    def back_prop(self,true_val,layers = [],times = 1):
        # give input in the backward direction 
        y = true_val
        weights = []
        biases = []
        learning_rate = 0.0005
        for j,i in enumerate(layers[:-1]):
            '''
            #1 a = activation
            #2 z = preactivation
            #3 w = weights
            #4 b = biases
            #5 f = activation function 
            #6 p = parent layers
            '''
            a,z,w,b,f,p = i
            tiled_a = numpy.tile(layers[j + 1][0],(len(a),1)).T
            # print(tiled_a,'<<<<here')
            # print(b.shape,'HERE TOO')
            a_previous = layers[j + 1][0]
            if f == "softmax":
                # print(a,y)
                dz = (a-y)
                dlw = numpy.outer(a_previous,dz)
                times = dz.dot(w.T)
            else:
                # print(tiled_a.shape,times.shape,self.relu_derivative(z).shape)
                dz = times * self.relu_derivative(z)
                # print(a_previous.shape)
                dlw = numpy.outer(a_previous,dz)
                # print(a_previous.shape,dlw.shape,dz.shape)
                times = dz.dot(w.T)
                pass
            dlb = dz
            # print(dlb.shape,'<<<<<<<<<<<<<<HEHRHHEHR')
            w =  w - learning_rate * dlw
            b = b - learning_rate * dlb
            # print(b.shape)
            # print(b,'<<<<<<<<<HERE')
            weights.append(w)
            biases.append(b)
        return weights,biases



    '''
    z = a0 * w + b
    a1 = softmax(z)
    l = true_val * log(a1)
    dl/dw = dl/da * da/dz * dz/dw
    da/dz = d(e^z/sum(e^i))/dz

    FOR SOFTMAX :-
    dla = d(ylog(a))/da
    dla = y/a

    dazk = d(e^i/sum(e^j))/dzk
    if i == k:
        
    
    '''


if __name__ == '__main__':

    a = (datetime.datetime.now().hour,datetime.datetime.now().minute,datetime.datetime.now().second)
    net = Network('image_folder',learn = False)
    weights,biases = net.weights_biases()
    b = (datetime.datetime.now().hour,datetime.datetime.now().minute,datetime.datetime.now().second)

    print('starting_time :', a)
    print('ending_time : ', b)