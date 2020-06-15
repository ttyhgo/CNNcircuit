
import sys
import os
import random
import math
from PIL import Image
import numpy as np
import time

if len(sys.argv) < 7:
    print(".py input_height input_width kernel_height kernel_width depth_size bit_size")
    exit(1)

f = open("./test.arith", "w")
f_in = open("./test.in", "w")

f2 = open("./nconv.arith", "w")
f2_in = open("./nconv.in", "w")

#f3 = open("./lego.arith", "w")
#f3_in = open("./lego.in", "w")

#f4 = open("./clego.arith", "w")
#f4_in = open("./clego.in", "w")

#f4 = open("./clego.arith", "w")
#all_in = open("./all.in", "w")

zeroHex = "30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001"
negOneHex = "30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000000"

inv4 = str('244B3AD628E5381F4A3C3448E1210245DE26EE365B4B146CF2E9782EF4000001')
inv2 = str('183227397098D014DC2822DB40C0AC2E9419F4243CDCB848A1F0FAC9F8000001')

input_h = int(sys.argv[1])
input_w = int(sys.argv[2])
kernel_h = int(sys.argv[3])
kernel_w = int(sys.argv[4])
depth_size_input = int(sys.argv[5])
bit_size_input = int(sys.argv[6])

#224x224x3 =>224x224x64 : input 150,528       
#kernel 3x3x3x64 : 1728

#224x224x64 => 224x224x64 : input 3,211,264
#kernel 3x3x64x64 : 36,864

#pool 224x224x64 => 112x112x64

#112x112x64 => 112x112x128 : input 802,816
#kernel 3x3x64x128 : 73728

#112x112x128 => 112x112x128 : input 1,605,632
#kernel 3x3x128x128 : 147,456

#pool 112x112x128 => 56x56x128 

#56x56x128 => 56x56x256 :401,408
#kernel 3x3x128x256 : 294,912

#56x56x256 => 56x56x256 : 802,816
#kernel 3x3x256x256 : 589,824

#pool 56x56x256 => 28x28x256

#28x28x256 => 28x28x512 :input 200,704
#kernel 3x3x256x512 : 1,179,648

#28x28x512 => 28x28x512 : 401,408
#kernel 3x3x512x512 : 2,359,296

#pool 28x28x512 => 14x14x512

#14x14x512 => 14x14x512 : 100,352
#kernel 3x3x512x512 : 2,359,296

#14x14x512 => 14x14x512 : 100,352
#kernel 3x3x512x512 : 2,359,296

#14x14x512 => 14x14x512 : 100,352
#kernel 3x3x512x512 : 2,359,296

#pool 14x14x512 => 7x7x512

#7x7x512 => 1x1x4096 : 25,088
#kernel 7x7x512x4096 : 102,760,448

#4096 => 1000 : 4096
#kernel 4096x1000 : 4,096,000

# input_h = 150528
# input_w = 1
# kernel_h = 27
# kernel_w = 1
# depth_size_input = 64


f.write("input 0\n") # test.arith 
f_in.write("0 1\n") # test.in w0 = 1
f.write("const-mul-0 in 1 <0> out 1 <1>\n")# w1 = 0
f.write("#Layer1\n")

f2.write("input 0\n") #nconv.arith
f2_in.write("0 1\n") #nconv.in
f2.write("const-mul-0 in 1 <0> out 1 <1>\n")

'''
f3.write("input 0\n") #lego.arith
f3_in.write("0 1\n") #lego.in
f3.write("#Layer1\n")
f3.write("const-mul-0 in 1 <0> out 1 <1>\n")

f4.write("input 0\n") #lego.arith
f4_in.write("0 1\n") #lego.in
f4.write("#Layer1\n")
f4.write("const-mul-0 in 1 <0> out 1 <1>\n")

all_in.write("0 1\n")
'''

const10Idx=2
constvalue = 2**bit_size_input
constvalue = hex(constvalue)
print(constvalue)
f2.write("const-mul-"+str(constvalue).split("x")[1]+" in 1 <0> out 1 <2>\n") #w2 = 2^bit
f2.write("#Layer1\n")
#outs = []
inputs = []
cidx =3

connect_idx = []
connect_val = []

input_val=[]
#output_val=[]
start = time.time()
dummlistval = []
dummlist = []

for depth in range(0,depth_size_input):
    depthinput=[]
    depthinputval=[]
    for i in range(3, 3+input_h*input_w):
        f.write("input "+str(cidx)+"\n")#inputs
        #f2.write("input "+str(cidx)+"\n")#smae inputs for connection
        rand = random.randrange(0,2**((int)(sys.argv[6])-4))*4#myim[(i-2)/input_height][(i-2)%input_width]*4#(i-1)*4#(random.randrange(0,256)) * 4#i-1
        depthinputval.append(rand)
        f_in.write(str(cidx)+" "+str(rand)+"\n")
        #all_in.write(str(cidx)+" "+str(rand)+"\n")
        connect_idx.append(cidx)
        connect_val.append(rand)
        depthinput.append(cidx)
        cidx = cidx + 1
    input_val.append(depthinputval)
    inputs.append(depthinput)
#print(inputs)

# input_val.append(dummlistval)
# inputs.append(dummlist)

print("input time ", time.time()-start)

def convolution(startIdx, inputarray, inputvalarray, input_height, input_width, kernel_height, kernel_width, depth_size):
    #print(inputvalarray)
    num_input = input_height * input_width + (kernel_height*kernel_width)
    num_output = (input_height + kernel_height -1) * (input_width + kernel_width-1)
    start = time.time()
    outarray=[]
    outvalarray=[]
    for depth in range(0, depth_size):
        f.write("#depth"+str(depth)+"\n")
        f2.write("#depth"+str(depth)+"\n")
        depth_out = []
        depth_check_out = [] # depth_out_val[i] == depth_check_out_val[i]
        depth_val_out = []
        depth_val_check_out = []
        kcnt = 0 #for test kernel -1 0 1
        kernels =[]
        ker_vals=[]
        for i in range(0, kernel_height * kernel_width):
            f.write("input "+str(startIdx)+"\n") # kernels
            #f2.write("input "+str(startIdx)+"\n") # kernels for checking connection
            kernels.append(startIdx)
            rand = random.randrange(-1,2)
            ker_vals.append(rand)
            f_in.write(str(startIdx)+" "+str(rand)+"\n")
            #all_in.write(str(startIdx)+" "+str(rand)+"\n")
            #connect_idx.append(startIdx)
            startIdx = startIdx + 1

        '''
        for i in range(0, num_output):
            f.write("input "+str(startIdx)+"\n") #convol output
            depth_cehck_out.append(startIdx)
            startIdx = startIdx + 1
        '''

        f.write("convol in "+str(num_input) + " <")
        for i in inputarray[depth]:
            f.write(str(i) + " ")

        for i in kernels:
            f.write(str(i) + " ")
        f.write("> out "+str(num_output)+" <")

        for i in range(0, num_output):
            depth_out.append(startIdx)
            f.write(str(startIdx)+" ")
            connect_idx.append(startIdx)
            startIdx = startIdx+1
        f.write("> state 4 <"+str(input_height)+" "+str(input_width)+" "+str(kernel_height)+" "+str(kernel_width)+">\n")
        #outs.append(depth_out)
        outarray.append(depth_out)
        for i in range(0, num_output):
            #f.write("output "+str(depth_out[i])+"\n")
            f.write("output "+str(depth_out[i])+"\n") #convol output
            f2.write("input "+str(depth_out[i])+"\n") #check connetion
            #f3.write("input "+str(depth_out[i])+"\n") #lego for nconv output
            #f4.write("input "+str(depth_out[i])+"\n") #lego for conv output

        print("conv input time ", time.time()-start)
        start = time.time()
        for j in range(0, input_width + kernel_width -1):
            for k in range(0, input_height + kernel_height -1):
                y =0
                for w in range(0, input_width):
                    for h in range(0, input_height):
                        k1 = j - w
                        k2 = k - h
                        if( k1 >=0 and k1 < kernel_width) and ( k2 >=0 and k2< kernel_height):
                            '''
                            print("dep : ",depth)
                            print(ker_vals[ k1*(kernel_height)+k2])
                            print(w, input_height, h)
                            print(inputvalarray[depth][(w*input_height+h)])
                            print(w, input_height, h)
                            '''
                            y = y+ ker_vals[ k1*(kernel_height)+k2]*inputvalarray[depth][(w*input_height+h)]
                                            
                #output_val.append(y)
                depth_val_out.append(y)
                f2_in.write(str(depth_out[j*(input_height+kernel_height-1)+k])+" "+str(y)+"\n")
                '''
                all_in.write(str(depth_out[j*(input_height+kernel_height-1)+k])+" "+str(y)+"\n")
                f3_in.write(str(depth_out[j*(input_height+kernel_height-1)+k])+" "+str(y)+" #conv output\n")
                f4_in.write(str(depth_out[j*(input_height+kernel_height-1)+k])+" "+str(y)+" #nconv input\n")
                '''

                connect_val.append(y)
        outvalarray.append(depth_val_out)
    lastIdx = outarray[depth_size-1][num_output-1]
    print("end convol")
    print("conv output time ", time.time()-start)
    return outarray, outvalarray, lastIdx+1

def ReLU(startIdx, inputarray, inputvalarray, bit_size, depth_size):
    print("begin ReLU")
    #bit_size = int(sys.argv[6])
    start = time.time()
    convOutputList = [] 
    convOutputValList = [] 

    lastIdx = startIdx #last index

    outarray=[]
    outvalarray=[]
    for depth in range(0, depth_size):
        f2.write("#L"+str(depth+1)+" ReLU\n")
        #outputIdx = inputarray[depth][0] # start idx of output
        inputidx=0
        for idx in inputarray[depth]:
            f2.write("add in 2 <"+str(idx)+" "+str(const10Idx)+"> out 1 <"+str(lastIdx)+">\n")
            f2.write("split in 1 <"+str(lastIdx)+"> out "+str(bit_size+1)+" <")
            lastIdx = lastIdx+1
            for j in range(0, bit_size):
                f2.write(str(lastIdx) + " ")
                lastIdx = lastIdx +1
            f2.write(str(lastIdx)+">\n")
            f2.write("mul in 2 <" + str(idx) + " "+str(lastIdx)+">")
            lastIdx = lastIdx + 1
            f2.write(" out 1 <"+str(lastIdx)+">\n")
            #f2.write("output "+str(lastIdx)+"\n")  # convol output
            #f.write("input "+str(lastIdx)+"\n")
            #f3.write("input "+str(lastIdx)+"\n")
            if inputvalarray[depth][inputidx] > 0:
                convOutputValList.append(inputvalarray[depth][inputidx])
                #f_in.write(str(lastIdx)+" "+str(inputvalarray[depth][inputidx])+"\n")
            else:
                convOutputValList.append(0)
                #f_in.write(str(lastIdx)+" "+str(0)+"\n")
            convOutputList.append(lastIdx)
            lastIdx = lastIdx+1
            inputidx = inputidx+1
        outarray.append(convOutputList)
        outvalarray.append(convOutputValList)
    print("end ReLU")
    print("ReLU output time ", time.time()-start)
    return outarray, outvalarray, lastIdx

def pooling(startIdx, inputarray, inputvalarray, depth_size):
    print("begin pool")
    start= time.time()
    poolDepthOutput=[]
    poolDepthValOutput=[]
    outarray=[]
    outvalarray=[]
    lastIdx = startIdx
    for depth in range(0, depth_size):
        add_cnt = 0
        f2.write("#L"+str(depth+1)+" AVGPool\n")
        ind = 0
        for idx in inputarray[depth]:
            
            if add_cnt %4 is 0:
                f2.write("add in 4 <" +str(idx)+ " ")
                add_cnt = 1
            elif add_cnt%4 is 3:
                f2.write(str(idx)+"> out 1 <"+str(lastIdx)+">\n")
                poolDepthOutput.append(lastIdx)
                f2.write("const-mul-"+inv4+" in 1 <"+str(lastIdx)+"> out 1 <"+str(lastIdx+1)+">\n")
                f2.write("output "+str(lastIdx+1)+"\n")
                f.write("input "+str(lastIdx+1)+"\n")
                #f3.write("input "+str(lastIdx+1)+"\n")
                #f4.write("input "+str(lastIdx+1)+"\n")
                connect_idx.append(lastIdx+1)
                avgval = inputvalarray[depth][ind-3] + inputvalarray[depth][ind-2] + inputvalarray[depth][ind-1] + inputvalarray[depth][ind]
                avgval = avgval / 4
                f_in.write(str(lastIdx+1)+" "+str(avgval)+"\n")
                connect_val.append(avgval)
                '''
                all_in.write(str(lastIdx+1)+" "+str(avgval)+"\n")
                f3_in.write(str(lastIdx+1)+" "+str(avgval)+" #conv input\n")
                f4_in.write(str(lastIdx+1)+" "+str(avgval)+" #nconv output\n")
                '''


                poolDepthValOutput.append(avgval)
                lastIdx = lastIdx+2
                add_cnt = add_cnt+1
            else:
                f2.write(str(idx)+" ")
                add_cnt = add_cnt+1
            
            ind = ind+1
        if add_cnt%4 is not 0:
            if add_cnt%4 is 1:
                f2.write(str(1)+" "+str(1)+" "+str(1)+"> out 1 <"+str(lastIdx)+">\n")
                avgval = inputvalarray[depth][ind-3]
                avgval = avgval / 4
                poolDepthValOutput.append(avgval)
                f_in.write(str(lastIdx)+" "+str(avgval)+"\n")
                connect_val.append(avgval)
                #all_in.write(str(lastIdx)+" "+str(avgval)+"\n")
                #f3_in.write(str(lastIdx)+" "+str(avgval)+"\n")
                #f2.write("output "+str(lastIdx))
            elif add_cnt%4 is 2:
                f2.write(str(1)+" "+str(1)+"> out 1 <"+str(lastIdx)+">\n")
                avgval = inputvalarray[depth][ind-3] + inputvalarray[depth][ind-2]
                avgval = avgval / 4
                poolDepthValOutput.append(avgval)
                f_in.write(str(lastIdx)+" "+str(avgval)+"\n")
                connect_val.append(avgval)
                #all_in.write(str(lastIdx)+" "+str(avgval)+"\n")
                #f3_in.write(str(lastIdx)+" "+str(avgval)+"\n")
                #f2.write("output "+str(lastIdx))
            elif add_cnt%4 is 3:
                f2.write(str(1)+"> out 1 <"+str(lastIdx)+">\n")
                avgval = inputvalarray[depth][ind-3] + inputvalarray[depth][ind-2] + inputvalarray[depth][ind-1]
                avgval = avgval / 4
                poolDepthValOutput.append(avgval)
                f_in.write(str(lastIdx)+" "+str(avgval)+"\n")
                connect_val.append(avgval)
                #all_in.write(str(lastIdx)+" "+str(avgval)+"\n")
                #f3_in.write(str(lastIdx)+" "+str(avgval)+"\n")
                #f2.write("output "+str(lastIdx))
            f2.write("const-mul-"+inv4+" in 1 <"+str(lastIdx)+"> out 1 <"+str(lastIdx+1)+">\n")
            f2.write("output "+str(lastIdx+1)+"\n")
            f.write("input "+str(lastIdx+1)+"\n")
            #f3.write("input "+str(lastIdx+1)+"\n")
            connect_idx.append(lastIdx+1)
            lastIdx = lastIdx+1
            poolDepthOutput.append(lastIdx)
        
        outarray.append(poolDepthOutput)
        outvalarray.append(poolDepthValOutput)
    print("end pool")
    print("pool output time ", time.time()-start)
    return outarray,outvalarray, lastIdx
#print("pool idx = ",poolDepthOutput[0],"~",poolDepthOutput[len(poolDepthOutput)-1])



outs, out_vals, lastIndex = convolution(cidx, inputs, input_val, input_height=input_h, input_width=input_w, kernel_height=kernel_h, kernel_width=kernel_w,depth_size=depth_size_input)

outs, out_vals, lastIndex = ReLU(lastIndex, outs, out_vals, bit_size_input, depth_size_input)

#outs, out_vals, lastIndex = convolution(lastIndex, outs, out_vals, input_h, input_w, kernel_h, kernel_w, depth_size_input)

#outs, out_vals, lastIndex = ReLU(lastIndex, outs, out_vals, bit_size_input, depth_size_input)
#print(outvals)

outs, out_vals, lastIndex = pooling(lastIndex, outs, out_vals, depth_size_input)
#print(connect_idx)
#print(connect_val)

for i in range(0, len(connect_idx)):
    f.write("cminput "+str(connect_idx[i]+lastIndex)+"\n")
    f2.write("cminput "+str(connect_idx[i]+lastIndex+len(connect_idx))+"\n")
    f_in.write(str(connect_idx[i]+lastIndex)+" "+str(connect_val[i])+"\n")
    f2_in.write(str(connect_idx[i]+lastIndex+len(connect_idx))+" "+str(connect_val[i])+"\n")


f.write("add in "+str(len(connect_idx))+" <")
for i in connect_idx:
    f.write(str(i+lastIndex)+" ")
f.write("> out 1 <"+str(lastIndex+connect_idx[-1]+1)+">\n")
f.write("output "+str(lastIndex+connect_idx[-1]+1)+"\n")

f2.write("add in "+str(len(connect_idx))+" <")
for i in connect_idx:
    f2.write(str(i+lastIndex+len(connect_idx))+" ")
f2.write("> out 1 <"+str(lastIndex+len(connect_idx)+connect_idx[-1]+2)+">\n")
f2.write("output "+str(lastIndex+len(connect_idx)+connect_idx[-1]+2)+"\n")
#print(outvals)
print("1-3layer after value len : ",len(outs))
'''
    if len(sys.argv) >= 8:
        f.write('#Layer2\n')
        depth_size = int(sys.argv[7])
        

        input_height = input_height/2
        input_width = input_width/2


        num_input = input_height* input_width+kernel_height*kernel_width
        print(input_height, input_width, num_input)


        num_output = (input_height + kernel_height -1) * (input_width + kernel_width-1)

        nextLayerKernelIdx = lastIdx
        for depth in range(0, depth_size):
            for i in range(nextLayerKernelIdx + (kernel_height*kernel_width+num_output)*depth, nextLayerKernelIdx + kernel_height*kernel_width + (kernel_height*kernel_width+num_output)*depth):
                f.write("input "+str(i)+"\n") # kernels
                rand = random.randrange(-2,3)#i-(1 + input_height*input_width + (kernel_height*kernel_width+num_output)*depth) #random.randrange(0,256)#i-1
                f_in.write(str(i)+" "+str(rand)+"\n")
                
            f.write("convol in "+str(num_input) + " <")
            for i in poolOutputList[depth]:
                f.write(str(i) + " ")

            for i in range(nextLayerKernelIdx + len(poolOutputList[depth]) + (kernel_height*kernel_width+num_output)*depth, nextLayerKernelIdx+ len(poolOutputList[depth])+(kernel_height*kernel_width) + (kernel_height*kernel_width+num_output)*depth-1):
                f.write(str(i) + " ")
            f.write(str(nextLayerKernelIdx+ len(poolOutputList[depth])+(kernel_height*kernel_width) + (kernel_height*kernel_width+num_output)*depth-1)+"> out "+str(num_output)+" <")

            for i in range(nextLayerKernelIdx+num_input + (kernel_height*kernel_width+num_output)*depth,nextLayerKernelIdx+num_input + num_output + (kernel_height*kernel_width+num_output)*depth-1):
                depth_out.append(i)
                f.write(str(i)+" ")
            depth_out.append(1+num_input + num_output + (kernel_height*kernel_width+num_output)*depth)
            f.write(str(1+num_input + num_output + (kernel_height*kernel_width+num_output)*depth)+"> state 4 <"+str(input_height)+" "+str(input_width)+" "+str(kernel_height)+" "+str(kernel_width)+">\n")
            outs.append(depth_out)




if len(sys.argv) >=7:
    depth_size = int(sys.argv[6])
    for d in range(1, depth_size):
        for i in range(2 + (num_input+num_output)*d, 2+num_input+(num_input+num_output)*d):
            f.write("input "+str(i)+"\n")

            if i < input_height*input_width+2+(num_input+num_output)*d:
                rand = random.randrange(0,256)
                f_in.write(str(i)+" "+str(rand)+"\n")
            else:
                rand = random.randrange(-1, 2)
                f_in.write(str(i)+" "+str(rand)+"\n")
        f.write("convol in "+str(num_input) + " <")
        for i in range(2+(num_input+num_output)*d, 2+num_input+num_output+num_input-1):
            f.write(str(i) + " ")
        f.write(str(num_input+1+num_input+num_output)+"> out "+str(num_output)+" <")

        for i in range(2+num_input+num_input+num_output, 1+num_input+num_output+num_input+num_output):
            f.write(str(i)+" ")
        f.write(str(1+num_input+num_output+num_input+num_output)+"> state 4 <"+str(input_height)+" "+str(input_width)+" "+str(kernel_height)+" "+str(kernel_width)+">\n")
'''		

f.close()
f_in.close()
f2.close()
f2_in.close()
'''
f3.close()
f3_in.close()
f4.close()
f4_in.close()
all_in.close()
'''

'''
os.system("./run_ppzksnark conv nconv.arith nconv.in test.arith test.in > test.result")

f = open("./test.result", "r")
lines = f.readlines()
mem_stat = 0
pk_cnt = 0
vk_cnt = 0
qap_cnt = 0

conv_key_f = open("./conv_key.result","a+")
conv_pro_f = open("./conv_pro.result","a+")
conv_qap_f = open("./conv_qap.result","a+")
conv_key_time=[]
conv_proof_time=[]
conv_qap_time=[]
for line in lines:
    index = line.find("var gadget")
    if index is not -1:
        print line
    index = line.find("(leave) Call to r1cs_gg_ppzksnark_generator")
    if index is not -1:
        words = line.split()[4].split("[")[1].split("s")
        #print "keygen time \t ",float(words[0])
        print "gro gen time : ",float(words[0])
        #print float(words[0])
    index = line.find("(leave) Call to r1cs_conv_ppzksnark_generator")
    if index is not -1:
        words = line.split()[4].split("[")[1].split("s")
        #print "keygen time \t ",float(words[0])
        print "conv gen time : ",float(words[0])
        #print float(words[0])
        conv_key_f.write(words[0]+"\n")
    index = line.find("(leave) Call to LEGO_Keygen")
    if index is not -1:
        words = line.split()[4].split("[")[1].split("s")
        #print "keygen time \t ",float(words[0])
        print "lego gen time : ",float(words[0])
        #print float(words[0])
    index = line.find("(leave) Call to LEGO_prover")
    if index is not -1:
        words = line.split()[4].split("[")[1].split("s")
        #print "keygen time \t ",float(words[0])
        print "lego proof time : ",float(words[0])
        #print float(words[0])
    index = line.find("(leave) Call to r1cs_gg_ppzksnark_prover")
    if index is not -1:
        words = line.split()[4].split("[")[1].split("s")
        #print "Proof time \t ",float(words[0])
        print "Gro prover time : ",float(words[0])
        #print float(words[0])
    index = line.find("(leave) Call to r1cs_conv_ppzksnark_prover")
    if index is not -1:
        words = line.split()[4].split("[")[1].split("s")
        #print "Proof time \t ",float(words[0])
        print "Conv prover time : ",float(words[0])
        #print float(words[0])
        conv_pro_f.write(words[0]+"\n")
    index = line.find("QAP degree")
    if index is not -1:
        words = line.split(":")[1]
        #print "QAP degree \t ", int(words)
        if qap_cnt is 0:
            #print "gro qap deg : ",int(words)
            print int(words)
            qap_cnt = qap_cnt+1
        else :
            #print "conv qap deg : ",int(words)
            print int(words)
            conv_qap_f.write(words)
    index = line.find("PK size in bits")
    if index is not -1:
        words = line.split(":")[1]
        #print "PK \t %.4f" % (float(words)/8/1000000),"[MB]" 
        if pk_cnt is 0:
            print "Gro PK size : %.4f" % (float(words)/8/1000000),"[MB]" 
            #print "%.4f" % (float(words)/8/1000000) 
            pk_cnt = 1
        elif pk_cnt is 1:
            print "Conv PK size : %.4f" % (float(words)/8/1000000),"[MB]" 
            #print "%.4f" % (float(words)/8/1000000) 
            pk_cnt = pk_cnt +1
        else:
            print "Lego PK size : %.4f" % (float(words)/8/1000000),"[MB]" 
            #print "%.4f" % (float(words)/8/1000000) 
            pk_cnt = pk_cnt +1

    index = line.find("VK size in bits")
    if index is not -1:
        words = line.split(":")[1]
        #print "VK \t %.4f" % (float(words)/8/1000000),"[MB]"
        if vk_cnt is 0:
            print "Gro VK size : %.4f" % (float(words)/8/1000000),"[MB]" 
            #print "%.4f" % (float(words)/8/1000000) 
            vk_cnt = 1
        elif vk_cnt is 1:
            print "Conv VK size : %.4f" % (float(words)/8/1000000),"[MB]" 
            #print "%.4f" % (float(words)/8/1000000) 
            vk_cnt = vk_cnt+1
        else:
            print "Lego VK size : %.4f" % (float(words)/8/1000000),"[MB]" 
            #print "%.4f" % (float(words)/8/1000000) 
            vk_cnt = vk_cnt+1
    index = line.find("Peak vsize")
    if index is not -1 and mem_stat is 0:
        words = line.split(":")[1]
        #print "Keygen Mem \t ", int(words),"[MB]"
        print int(words),"[MB]"
        mem_stat = mem_stat +1
    elif index is not -1 and mem_stat is 1:
        words = line.split(":")[1]
        #print "Proof Mem \t ", int(words),"[MB]"
        print int(words),"[MB]"
        mem_stat = mem_stat +1
    
    index = line.find("Proof size in bits")
    if index is not -1:
        words = line.split(":")[1]
        #print "Proof size \t ", int(words),"[bit]"
        print int(words),"[bit]"
    index = line.find("result is")
    if index is not -1:
        words = line.split(":")[1]
        print words
    
'''
    
    

